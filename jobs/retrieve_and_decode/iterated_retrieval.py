print("(Re)/Loading iterated_retrieval.py")

# Standard library
import collections
import copy
import dataclasses
import glob
import importlib
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import *

SCRIPT_DIR = Path(__file__).resolve().parent

# Third party
import beartype
import hydra
import jsonlines
import numpy as np
import omegaconf
import rich
import torch
import transformers
import tqdm

# First Party
import iterated_utils as utils

ROOT_PATH = SCRIPT_DIR.parent.parent
GAR_PATH = ROOT_PATH / "GAR/gar"
sys.path.insert(0, str(GAR_PATH))
DPR_PATH = ROOT_PATH / "DPR"
CONF_PATH = DPR_PATH / "conf"
os.chdir(DPR_PATH)

import train_generator
import dpr.options
import dpr.utils.model_utils
import dense_retriever
import jules_validate_dense_retriever
import utils_gen


LOGGER = logging.getLogger(__name__)
LOGGER.info(f"(Re)loaded {Path(__file__).name}")
LOGGER.info(f"{dense_retriever.__file__ = }")


def build_tokenizers_and_datasets(
    generation_batch_size,
    data_dir,
    max_target_len,
    max_source_len,
):
    SUBSET = "train"

    ###############################################################################
    # Tokenizers
    ###############################################################################
    tokenizer_bart = transformers.AutoTokenizer.from_pretrained(
        "facebook/bart-large"
    )
    tokenizer_bert = transformers.AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    ###############################################################################
    # Build dataloader
    ###############################################################################
    with utils.time_this("Build dataloader"):
        dataset = utils_gen.SummarizationDataset(
                tokenizer_bart,
                type_path=SUBSET,
                data_dir=data_dir,
                max_source_length=max_source_len,
                max_target_length=max_target_len,
            )

        subset = torch.utils.data.Subset(
            dataset,
            list(range(generation_batch_size * 5))
        )
        
        dataloader = torch.utils.data.DataLoader(
            subset,
            batch_size=generation_batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,  # DO NOT CHANGE THIS! It fucks with the metrics. There is no reason to shuffle anyways at inference.
            num_workers=0,
        )

    return (
        dataloader,
        tokenizer_bart,
        tokenizer_bert,
    )


@dataclasses.dataclass
class DecoderConf:
    num_beams: int
    num_return_sequences: int
    max_length: int
    temperature: float = 1
    early_stopping: bool = True



@beartype.beartype
def decode(
    model: train_generator.SummarizationTrainer,
    batch: Mapping[str, torch.Tensor],
    tokenizer: transformers.PreTrainedTokenizer,
    decoding_conf: DecoderConf,
) -> Tuple[np.ndarray, np.ndarray]:

    utils.check_isinstance(decoding_conf, DecoderConf)
    assert vars(decoding_conf).get("output_scores", True), (
        decoding_conf.output_scores
    )

    source_ids, source_mask = utils_gen.trim_batch(
        batch["input_ids"],
        tokenizer.pad_token_id,
        attention_mask=batch["attention_mask"],
    )
    generated_ids = model.model.generate(
        input_ids=source_ids.cuda(),
        attention_mask=source_mask.cuda(),
        output_scores=True,
        **vars(decoding_conf),
    )
    probs = model.model(generated_ids)[0]
    winners = torch.gather(input=probs, dim=2, index=generated_ids.unsqueeze(2)).squeeze(2)
    mask = generated_ids != tokenizer.pad_token_id
    masked_winners = winners * mask
    scores = torch.prod(masked_winners, -1)

    # LOGGER.info(
    #     f"{source_ids.shape =     }\n"
    #     f"{source_mask.shape =    }\n"
    #     f"{scores.shape =         }\n"
    #     f"{winners.shape =        }\n"
    #     f"{generated_ids.shape =  }\n"
    #     f"{probs.shape =          }\n"
    #     f"{decoding_conf =        }\n"
    # )

    shape = (batch["input_ids"].shape[0], decoding_conf.num_return_sequences, -1)

    return generated_ids.cpu().numpy().reshape(*shape), scores.cpu().numpy().reshape(*shape)


def clean_bart_decode(output, tokenizer_bart):
    return output.replace(
         tokenizer_bart.pad_token, ""
    ).replace(
        tokenizer_bart.bos_token, ""
    ).replace(
        tokenizer_bart.eos_token, ""
    ).replace(
        tokenizer_bart.sep_token, ""
    ).strip()


def question_generator(dataloader_, tokenizer_bart, information):
    """
    This is to enforce consistence with the training of the BART models.
    """
    end_ = 0
    for i, data_dict in enumerate(tqdm.notebook.tqdm(
        dataloader_, desc=f"{information} question_generator"
    )):
        question_ids = data_dict["source_ids"]
        question_text = [

            clean_bart_decode(
                tokenizer_bart.decode(question_id),
                tokenizer_bart,
            )

            for question_id in question_ids
        ]

        yield question_text


def write_contexts(
    all_contexts: Dict[str, str],
    context_ids: List[str],
    out_path: str,
):

    # text = []
    # for ids_per_retrieval in context_ids:
    #     text.append([all_contexts[ids] for ids in ids_per_retrieval])

    retrieved = dict(
        # text=text,
        ids=context_ids,
    )

    with jsonlines.open(out_path, "a") as f_out:
        f_out.write(retrieved)


def write_generations(
    generated_text: List[str],
    path: str,
):
    with jsonlines.open(path, "a") as f_out:
        f_out.write(generated_text)


def print_generation(
    title: str,
    input_ids: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
):
    LOGGER.info(f"{title}: {tokenizer.decode(input_ids)}")


def build_models(reader_model_path, query_aug_model_path):
    ###############################################################################
    # Load query model
    ###############################################################################
    with utils.time_this("query_aug_model.load_from_checkpoint"):
        query_aug_model = train_generator.SummarizationTrainer.load_from_checkpoint(
            str(query_aug_model_path)
        )

    ###############################################################################
    # Load inference model
    ###############################################################################
    # with utils.time_this("reader_model.load_from_checkpoint"):
    #     reader_model = train_generator.SummarizationTrainer.load_from_checkpoint(
    #         str(reader_model_path)
    #     )
    return query_aug_model, None