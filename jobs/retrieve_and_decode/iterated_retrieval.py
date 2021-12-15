print("(Re)/Loading iterated_retrieval.py")

# Standard library
import abc
import argparse
import collections
import contextlib
import copy
import dataclasses
import glob
import importlib

from transformers.file_utils import ENV_VARS_TRUE_AND_AUTO_VALUES
from transformers.tokenization_utils_base import ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING

try:
    import ujson as json
except ImportError:
    import json

import logging
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import *

SCRIPT_DIR = Path(__file__).absolute().parent

# Third party
import beartype
import colorama
import hydra
import jsonlines
import more_itertools
import numpy as np
import omegaconf
import rich
import torch
import transformers
import tqdm

# First Party
import iterated_utils as utils
import common_retriever

ROOT_PATH = SCRIPT_DIR.parent.parent
GAR_PATH = ROOT_PATH / "GAR/gar"
sys.path.insert(0, str(GAR_PATH))
DPR_PATH = ROOT_PATH / "DPR"
CONF_PATH = DPR_PATH / "conf"
sys.path.insert(0, str(DPR_PATH))

import train_generator
import dpr.utils.model_utils
import dense_retriever
import utils_gen


LOGGER = logging.getLogger(__name__)
LOGGER.info(f"(Re)loaded {Path(__file__).name}")
LOGGER.info(f"{dense_retriever.__file__ = }")


PathType = Union[str, Path]


class NLI:
    ENTAILMENT_POS = 0
    NEUTRAL_POS = 1
    CONTRADICTION_POS = 2


    def __init__(self, ) -> None:    
        self.hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
        # self.hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli" # Didn't work
        # self.hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
        # self.hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
        # self.hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.hg_model_hub_name
        )
        LOGGER.info(f"Loading model: \"{self.hg_model_hub_name}\"")
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.hg_model_hub_name
        )
        LOGGER.info("Model loaded.")

        self.call_args = []
        self.call_kwargs = dict()
        self.tokenizer_args = []
        self.tokenizer_kwargs = dict(
            pad_to_max_length=True,
            return_tensors="pt",
            return_token_type_ids=True, 
            truncation=True,
        )
        
        
    @classmethod
    def _all(cls, preds):
        return preds[:, cls.ENTAILMENT_POS]


    def __call__(self, contexts, questions, answers) -> Any:
        tokenizer_batch = self.tokenizer.batch_encode_plus(
            contexts, 
            questions + answers, 
            *self.tokenizer_args, 
            **self.tokenizer_kwargs,
        )

        raw_preds = self.model(
            tokenizer_batch, 
            *self.call_args, 
            **self.call_kwargs
        )   
        assert raw_preds.ndims == 2, raw_preds.ndims
        assert raw_preds.shape[1] == 3, raw_preds.shape

        preds = self._all(raw_preds)

        assert preds.ndims == 1, preds.shape

        return preds


def build_tokenizers_and_datasets(
    generation_batch_size,
    data_dir,
    max_target_len,
    max_source_len,
    cv_set
):

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
                type_path=cv_set,
                data_dir=data_dir,
                max_source_length=max_source_len,
                max_target_length=max_target_len,
            )

        # subset = torch.utils.data.Subset(
        #     dataset,
        #     list(range(generation_batch_size * 5))
        # )

        dataloader = torch.utils.data.DataLoader(
            dataset,
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
    max_length: int
    num_return_sequences: int
    repetition_penalty: Optional[float]
    num_beams: Optional[int] = None
    num_beam_groups: Optional[int] = None
    temperature: Optional[float] = None
    early_stopping: bool = True
    do_sample: bool = False
    diversity_penalty: Optional[float] = None
    

    def __post_init__(self):
        self.validate()

    def is_sample(self) -> bool:
        return self.do_sample

    def is_beam_group(self) -> bool:
        return self.num_beam_groups is not None and self.num_beam_groups != 1
        
    def validate(self) -> None:
        if self.repetition_penalty <= 1.0:
            raise ValueError(
                f"{self.repetition_penalty = }. You likely want repetition_penalty to be > 1."
            )

        if not self.is_sample() and self.num_beams < self.num_return_sequences:
            raise ValueError("`num_beams` must be >= `num_return_sequences`.")

        if self.is_beam_group() and not self.diversity_penalty:
            raise ValueError(
                "`diversity_penalty` must be set when using beam groups."
            )
            

        if self.is_sample() and self.is_beam_group():
            raise ValueError(
                f"Sample cannot be used with beam groups."
            )
        if self.is_sample() and self.temperature is None:
            raise ValueError(
                f"Sample cannot be used without a temperature."
            )
        if self.is_sample() and (self.temperature < 0 or math.isclose(self.temperature, 0)):
            raise ValueError(
                f"Temperature must be >= 0"
            )
        
    def to_json(self):
        return dataclasses.asdict(self)


def compute_score_reference(probs, generated_ids, tokenizer):
    probs = probs.cpu().numpy()
    generated_ids = generated_ids.cpu().numpy()
    assert probs.shape[:2] == generated_ids.shape, (
        f"{probs.shape = }, {generated_ids.shape = }"
    )

    scores_per_batch = []
    winner_probs = []
    for batch_idx in range(generated_ids.shape[0]):
        batch_score = 0.0
        seq_len = 0
        seen_one_pad = False
        winner_probs_batch = []
        for seq_pos in range(generated_ids.shape[1]):
            winner = generated_ids[batch_idx, seq_pos]
            if winner != tokenizer.pad_token_id:
                assert not seen_one_pad, (
                    f"{winner = } {generated_ids[batch_idx] = }"
                )
                seq_len += 1
                winner_token_prob = probs[batch_idx, seq_pos, winner]
                batch_score += winner_token_prob
                winner_probs_batch.append(winner_token_prob)
            else:
                seen_one_pad = True
                winner_probs_batch.append(0.)

        winner_probs.append(winner_probs_batch)
        scores_per_batch.append(batch_score / seq_len)

    output = torch.FloatTensor(scores_per_batch).cuda()
    assert output.size() == (generated_ids.shape[0],)
    return output, winner_probs


def compute_score_torch(probs, generated_ids, tokenizer, winner_probs_ref):

    winner_probs_ref = torch.FloatTensor(winner_probs_ref).cuda()    
    winner_probs_torch = torch.take_along_dim(probs, generated_ids.unsqueeze(-1), dim=2,).squeeze()
    
    assert winner_probs_torch.size() == generated_ids.size(), (
        winner_probs_torch.size(), generated_ids.size()
    )

    mask = generated_ids != tokenizer.pad_token_id
    seq_lens = mask.sum(dim=1)
    assert seq_lens.size() == (generated_ids.size(0),)

    masked_winner_probs_torch = winner_probs_torch * mask
    masked_winner_probs_ref = winner_probs_ref * mask

    assert torch.allclose(masked_winner_probs_torch, masked_winner_probs_ref), (
        f"{masked_winner_probs_torch = }", 
        f"{masked_winner_probs_ref = }", 
        f"{mask = }",
        f"{winner_probs_torch.dtype = }", 
        f"{winner_probs_ref.dtype = }",
    )

    scores = torch.sum(masked_winner_probs_torch, 1) / seq_lens.float()
    return scores


# @beartype.beartype
def decode(
    model: train_generator.SummarizationTrainer,
    batch: Mapping[str, torch.Tensor],
    tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast],
    decoding_conf: DecoderConf,
) -> Tuple[np.ndarray, np.ndarray]:

    # utils.check_isinstance(decoding_conf, DecoderConf)
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
    
    start_reference = time.monotonic()
    scores_ref, winner_probs = compute_score_reference(probs, generated_ids, tokenizer)
    end_reference = time.monotonic() - start_reference
    start_torch = time.monotonic()
    scores_torch = compute_score_torch(probs, generated_ids, tokenizer, winner_probs)
    end_torch = time.monotonic() - start_torch
    rich.print(
        f"[cyan bold]Reference: {end_reference}s, "
        f"Torch: {end_torch}s"
    )
    assert torch.allclose(scores_ref, scores_torch), (scores_ref, scores_torch)

    scores = scores_torch

    shape = (batch["input_ids"].shape[0], decoding_conf.num_return_sequences, -1)

    return (
        generated_ids.cpu().numpy().reshape(*shape), 
        scores.cpu().numpy().reshape(*shape),
    )


def clean_bart_decode(output, tokenizer_bart) -> str:
    return output.replace(
        tokenizer_bart.pad_token, ""
    ).replace(
        tokenizer_bart.bos_token, ""
    ).replace(
        tokenizer_bart.eos_token, ""
    ).replace(
        tokenizer_bart.sep_token, ""
    ).strip()


def question_generator(dataloader_, tokenizer_bart, information, oracle_mode) -> Generator[str, None, None]:
    """
    This is to enforce consistence with the training of the BART models.
    """
    for data_dict in tqdm.tqdm(
        dataloader_, desc=f"{information} question_generator"
    ):
        question_ids = data_dict["source_ids"]
        question_text = [
            clean_bart_decode(
                tokenizer_bart.decode(question_id),
                tokenizer_bart,
            )
            for question_id in question_ids
        ]
        target_ids = data_dict["target_ids"]
        target_text = [
            clean_bart_decode(
                tokenizer_bart.decode(target_id),
                tokenizer_bart,
            )
            for target_id in target_ids
        ]
        

        if oracle_mode:
            for target_entry in target_text:
                assert target_entry, target_entry

        yield question_text, target_text


def write_contexts(
    all_contexts: Dict[str, str],
    context_ids: List[str],
    out_path: str,
) -> None:

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
) -> None:
    with jsonlines.open(path, "a") as f_out:
        f_out.write(generated_text)


def print_generation(
    title: str,
    input_ids: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
) -> None:

    LOGGER.info(f"{title}: {tokenizer.decode(input_ids)}")


def build_models(
    reader_model_path, query_aug_model_path: PathType,
    ) -> Tuple[train_generator.SummarizationTrainer, train_generator.SummarizationTrainer]:

    assert isinstance(query_aug_model_path, (str, Path)), type(query_aug_model_path).mro()
    # assert isinstance(reader_model_path, (str, Path)), type(reader_model_path).mro()

    ###############################################################################
    # Load query model
    ###############################################################################
    with utils.time_this("query_aug_model.load_from_checkpoint"):
        query_aug_model = train_generator.SummarizationTrainer.load_from_checkpoint(
            str(query_aug_model_path)
        )

    # ###############################################################################
    # # Load inference model
    # ###############################################################################
    # with utils.time_this("reader_model.load_from_checkpoint"):
    #     reader_model = train_generator.SummarizationTrainer.load_from_checkpoint(
    #         str(reader_model_path)
    #     )

    return query_aug_model, None


###############################################################################
# Specific to selection technique
###############################################################################
@utils.class_checker
@dataclasses.dataclass
class SelectionTechniqueChecksInfo:
    batch_size: int
    num_sequences: int
    n_docs: int
    loop_i: int



class SelectionTechnique(abc.ABC):
    def __init__(self):
        pass


    @final
    def select(
        self,
        *,
        top_retr_ids_np: np.ndarray,
        scores_retr_np: np.ndarray,
        final_num_contexts: int,
        query_scores_batch: np.ndarray,
        checks_info: SelectionTechniqueChecksInfo,
        is_first_loop: bool,
        question_text: List[str],
        retr_text: List[str],
        query_aug_text: List[str],
    ):
        utils.check_shape(top_retr_ids_np.shape, (
            checks_info.batch_size, 
            checks_info.num_sequences, 
            checks_info.n_docs,
        ))

        effective_batch_size, queries_per_question, n_docs = top_retr_ids_np.shape

        output = self._select(
            top_retr_ids_np,
            scores_retr_np,
            final_num_contexts,
            query_scores_batch,
            checks_info,
            is_first_loop,
            question_text,
            retr_text,
            query_aug_text,
            effective_batch_size,
            queries_per_question,
            n_docs,
        )

        # Shape Verification
        try:
            utils.check_shape(
                output.shape, (checks_info.batch_size, final_num_contexts)
            )
        except ValueError as err:
            raise utils.add_to_err(
                err,
                f"\t- {checks_info.batch_size = }\n"
                f"\t- {checks_info.num_sequences = }\n"
                f"\t- {checks_info.n_docs = }\n"
                f"\t- {checks_info.loop_i = }\n"
            )

        return output


    @abc.abstractmethod
    def _select(
        self,
        top_retr_ids_np: np.ndarray,
        scores_retr_np: np.ndarray,
        final_num_contexts: int,
        query_scores_batch: np.ndarray,
        checks_info: SelectionTechniqueChecksInfo,
        is_first_loop: bool,
        question_text: List[str],
        retr_text: List[str],
        query_aug_text: List[str],
        effective_batch_size: int,
        queries_per_question: int,
        n_docs: int,
    ):
        pass


    @final
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.select(*args, **kwds)


class AugTopK(SelectionTechnique):
    def _select(
        self,
        top_retr_ids_np: np.ndarray,
        scores_retr_np: np.ndarray,
        final_num_contexts: int,
        query_scores_batch: np.ndarray,
        checks_info: SelectionTechniqueChecksInfo,
        is_first_loop: bool,
        question_text: List[str],
        retr_text: List[str],
        query_aug_text: List[str],
        effective_batch_size: int,
        queries_per_question: int,
        n_docs: int,
    ):
        if not is_first_loop:    
            assert query_scores_batch is not None
            assert scores_retr_np is not None
            assert (
                query_scores_batch.dtype == np.float32 or 
                query_scores_batch.dtype == np.float64
            ), f"query_scores_batch.dtype = {query_scores_batch.dtype}"
            assert (
                scores_retr_np.dtype == np.float32 or 
                scores_retr_np.dtype == np.float64
            ), f"scores_retr_np.dtype = {scores_retr_np.dtype}"

            
            query_scores_batch = query_scores_batch.squeeze(2)

            utils.check_shape(
                scores_retr_np.shape, 
                (effective_batch_size, queries_per_question, n_docs),
            )
            utils.check_shape(
                query_scores_batch.shape,
                (effective_batch_size, queries_per_question),
            )

            unsqueezed_query_scores_batch = np.expand_dims(query_scores_batch, 2)
            return utils.top_k_max(
                scores=scores_retr_np * unsqueezed_query_scores_batch,
                indices=top_retr_ids_np,
                final_qty=final_num_contexts,
            )

        else:
            return utils.top_k_max(
                scores=scores_retr_np,
                indices=top_retr_ids_np,
                final_qty=final_num_contexts,
            )


class ArgMaxTopK(SelectionTechnique):
    def _select(
        self,
        top_retr_ids_np: np.ndarray,
        scores_retr_np: np.ndarray,
        final_num_contexts: int,
        *args, **kwargs
    ):
        return utils.top_k_max(
            scores=scores_retr_np,
            indices=top_retr_ids_np,
            final_qty=final_num_contexts,
        )


class NLITopK(SelectionTechnique):
    def __init__(self):
        super().__init__()
        self.nli_context = NLI()


    def _select(
        self,
        top_retr_ids_np: np.ndarray,
        scores_retr_np: np.ndarray,
        final_num_contexts: int,
        query_scores_batch: np.ndarray,
        checks_info: SelectionTechniqueChecksInfo,
        is_first_loop: bool,
        question_text: List[str],
        retr_text: List[str],
        query_aug_text: List[str],
        effective_batch_size: int,
        queries_per_question: int,
        n_docs: int,
    ):
        
        nli_scores = self.nli_context(
            context_text=retr_text, 
            quesiton_text=question_text, 
            answer_text=query_aug_text
        )
        return utils.top_k_max(
            scores=scores_retr_np * nli_scores,
            indices=top_retr_ids_np,
            final_qty=final_num_contexts,
        )

SELECTION_TECHNIQUE_TYPES = dict(
        aug_top_k=AugTopK,
        arg_max_top_k=ArgMaxTopK,
        nli_top_k=NLITopK,
    )



class AugmentationModes(utils.StrValueEnum):
    CONCATENATE = "concatenate"
    JUST_USE_AUG = "just_use_aug"


class DatasetType(utils.StrValueEnum):
    ANSWER = "answer"
    SENTENCE = "sentence"
    TITLE = "title"


class InferenceFunctions:
    def __init__(self) -> NoReturn:
        raise RuntimeError("This class is not meant to be instantiated")

    @classmethod
    def prepare_the_queries(
        cls,
        *, 
        question_dataloader, 
        tokenizer_bart, 
        loop_i: int, 
        query_aug_text_all_loops, 
        query_aug_score_all_loops, 
        aug_method, 
        decoding_conf_query_aug, 
        augmentation_mode, 
        tokenizer_bert, 
        output_paths: Dict[str, PathType],
        oracle_mode: bool,
    ):
        if oracle_mode:
            assert loop_i == 0, loop_i

        all_queries_this_loop = []
        questions_batching_generator = question_generator(
            question_dataloader,
            tokenizer_bart,
            f"[{loop_i = }] Preparing the retrieval queries :: ",
            oracle_mode=oracle_mode,
        )

        if loop_i == 0:
            query_batch_generator = (
                None for _ in range(len(question_dataloader))
            )
            query_batch_scores_generator = (
                None for _ in range(len(question_dataloader))
            )

        else:
            query_batch_generator = more_itertools.chunked(
                query_aug_text_all_loops[-1],
                question_dataloader.batch_size,
            )
            query_batch_scores_generator = more_itertools.chunked(
                query_aug_score_all_loops[-1],
                question_dataloader.batch_size,
            )
        
        
        for batch_i, (
            (questions_batch, targets_batch), query_aug_batch, query_aug_batch_scores,
        ) in enumerate(
            more_itertools.zip_equal(
                questions_batching_generator,
                query_batch_generator,
                query_batch_scores_generator,
                ),
            ):

            if loop_i == 0 and not oracle_mode:
                assert query_aug_batch is None
                # The questions are our queries.
                all_queries_this_loop.extend(
                    [[x] for x in questions_batch]
                )
            else:
                if not oracle_mode:
                    query_aug_batch = np.array(query_aug_batch, dtype="object")

                # Use the query augs to augment the question.
                if aug_method == "RETRIEVE_ALL_INDIVIDUALLY":
                    # If we retrieve all queries individually, then
                    # we keep the 1:1 relationship between the score
                    # qty and the query qty
                    if not oracle_mode:
                        utils.check_equal(
                            query_aug_batch.shape[1],
                            decoding_conf_query_aug.num_return_sequences,
                        )
                        utils.check_equal(query_aug_batch.ndim, 2)

                    else:
                        assert query_aug_batch is None, query_aug_batch
                        query_aug_batch = [None for _ in range(len(questions_batch))]

                    for i, (question, query_set, target) in enumerate(
                        more_itertools.zip_equal(
                            questions_batch,
                            query_aug_batch,
                            targets_batch,
                        )
                    ):
                    
                        per_question = []
                        if oracle_mode:
                            if augmentation_mode == AugmentationModes.CONCATENATE:
                                sentence = question + tokenizer_bert.sep_token + target            
                            elif augmentation_mode == AugmentationModes.JUST_USE_AUG:
                                sentence = target
                            else:
                                AugmentationModes.error(augmentation_mode)

                            per_question.append(sentence)
                            all_queries_this_loop.append(per_question)
                        else:
                            for gen in query_set:
                                if augmentation_mode == AugmentationModes.CONCATENATE:
                                    sentence = question + tokenizer_bert.sep_token + gen
                                elif augmentation_mode == AugmentationModes.JUST_USE_AUG:
                                    sentence = gen
                                else:
                                    AugmentationModes.error(augmentation_mode)

                                per_question.append(sentence)
                            all_queries_this_loop.append(per_question)

            write_generations(
                all_queries_this_loop, output_paths["retr_inputs"],
            )

        return all_queries_this_loop

    @classmethod
    def retrieve(
        cls,
        *,
        loop_i: int,
        retriever_batch_size: int,
        decoding_conf_query_aug,
        all_queries_this_loop,
        all_query_augs_this_loop,
        query_aug_score_all_loops,
        aug_method,
        retriever,
        all_passages,
        special_query_token,
        n_docs: int,
        final_num_contexts: int,
        selection_mode,
        output_paths,
        selection_context: SelectionTechnique,
        oracle_mode: bool,
        retrieval_max_size: int,
    ):
        retrieved_this_loop = []
        with utils.time_this("retrieve", no_start=True):
            # If we are at loop_i == 0, the number of queries is 1
            # so the number of retrievals is batch_size * 1, which is
            # num_augs times smaller than it is for loop_i > 0. To
            # compensate, we make the batches larger by a factor of
            # num_augs.
            if loop_i == 0 or oracle_mode:
                effective_batch_size = (
                    retriever_batch_size *
                    (decoding_conf_query_aug.num_return_sequences if not oracle_mode else 1)
                )
                queries_per_question = 1
                all_queries_scores_this_loop = [
                    None for _ in range(len(all_queries_this_loop))
                ]
            else:
                effective_batch_size = retriever_batch_size
                queries_per_question = (
                    decoding_conf_query_aug.num_return_sequences
                )
                all_queries_scores_this_loop = (
                    query_aug_score_all_loops[-1]
                )

            # Make sure we have as many scores as we have queries.
            # This should always be true.
            try:
                utils.check_equal(
                    len(all_queries_this_loop),
                    len(all_queries_scores_this_loop),
                )

            except ValueError as err:
                raise utils.add_to_err(
                    err, (
                        f"{len(all_queries_this_loop) = }\n"
                        f"{np.array(all_queries_this_loop).shape = }\n"
                        f"{len(all_queries_scores_this_loop) = }\n"
                        f"{np.array(all_queries_scores_this_loop).shape = }\n"
                        f"{loop_i = }\n"
                    )
                )


            for batch_i, (query_batch, query_scores_batch, query_augs_batch) in enumerate(
                more_itertools.zip_equal(
                    more_itertools.chunked(
                        tqdm.tqdm(
                            all_queries_this_loop,
                            desc="retrieval all_queries_this_loop",
                        ),
                        effective_batch_size
                    ),
                    more_itertools.chunked(
                        all_queries_scores_this_loop,
                        effective_batch_size,
                    ),
                    more_itertools.chunked(
                        all_query_augs_this_loop,
                        effective_batch_size,
                    )
                )
            ):

                # Retrieve.
                if aug_method == "RETRIEVE_ALL_INDIVIDUALLY":
                    query_batch_np = np.array(
                        query_batch, dtype="object",
                    ).reshape(-1)
                    
                    real_batch_size = len(query_batch)

                    # TODO: Make sure the reshaping makes sense
                    top_ids_and_scores = common_retriever.retrieve(
                        retriever=retriever,
                        all_passages=all_passages,
                        questions=query_batch_np,
                        special_query_token=special_query_token,
                        n_docs=n_docs,
                        retrieval_max_size=retrieval_max_size,
                    )

                    ################################################
                    # Deal with contexts
                    ################################################
                    top_ids, scores_retr = top_ids_and_scores
                    
                    try:
                        top_ids_np = np.array(top_ids).reshape(
                            real_batch_size,
                            queries_per_question,
                            n_docs
                        )
                        scores_retr_np = np.array(scores_retr).reshape(
                            real_batch_size,
                            queries_per_question,
                            n_docs,
                        )
                    except ValueError as err:
                        err = utils.add_to_err(
                            f"\t- {top_ids = }\n"
                            f"\t- {real_batch_size = }\n"
                            f"\t- {effective_batch_size = }\n"
                            f"\t- {queries_per_question = }\n"
                            f"\t- {n_docs = }\n"
                            f"\t- {loop_i = }"
                        )
                        raise err


                    selected_contexts_ids_np = selection_context(
                        top_retr_ids_np=top_ids_np,
                        scores_retr_np=scores_retr_np,
                        final_num_contexts=final_num_contexts,
                        query_scores_batch=np.array(query_scores_batch),
                        checks_info=SelectionTechniqueChecksInfo(
                            batch_size=real_batch_size,
                            num_sequences=queries_per_question,
                            n_docs=n_docs,
                            loop_i=loop_i,
                        ),
                        is_first_loop=loop_i == 0,
                        question_text=query_batch,
                        retr_text=retr_text,
                        query_aug_text=query_augs_batch,   
                        is_first=batch_i == 0,
                    )

                    utils.check_shape(
                        selected_contexts_ids_np.shape,
                        (
                            real_batch_size,
                            final_num_contexts
                        )
                    )

                    retrieved_this_loop.extend(selected_contexts_ids_np)

                else:
                    raise ValueError(aug_method)

                write_contexts(
                    all_contexts=all_passages,
                    context_ids=selected_contexts_ids_np.tolist(),
                    out_path=output_paths["retr_outs"],
                )

        del selected_contexts_ids_np
        del all_queries_this_loop
        del all_queries_scores_this_loop

        return retrieved_this_loop

    @classmethod
    def generate(
        cls,
        *,
        loop_i,
        query_aug_text_all_loops,
        query_aug_score_all_loops,
        retrieved_this_loop,
        question_dataloader,
        generation_batch_size,
        tokenizer_bart,
        all_passages,
        query_aug_model,
        query_aug_input_max_length,
        decoding_conf_query_aug,
        output_paths,
        oracle_mode,
    ):
        LOGGER.info(f"[{loop_i = }] Starting generation.")
        query_aug_text_all_loops.append([])
        query_aug_score_all_loops.append([])

        utils.check_equal(len(query_aug_text_all_loops), loop_i + 1)
        utils.check_equal(len(query_aug_score_all_loops), loop_i + 1)
        tqdm_info = f"[{loop_i = }] Generating with BART models :: "
        num_batchs_generation = np.ceil(
            len(retrieved_this_loop) / question_dataloader.batch_size
        )
        utils.check_equal(
            question_dataloader.batch_size,
            generation_batch_size,
        )
        try:
            utils.check_equal(
                num_batchs_generation,
                len(question_dataloader),
            )
        except ValueError as err:
            err = utils.add_to_err(
                f"\t- {loop_i = }\n"
                , err
            )
            raise err

        for batch_i, ((questions_batch, _), context_batch) in enumerate(
            more_itertools.zip_equal(
                question_generator(
                    question_dataloader, 
                    tokenizer_bart, 
                    tqdm_info, 
                    oracle_mode=oracle_mode,
                ),
                more_itertools.chunked(
                    retrieved_this_loop, generation_batch_size,
                ),
            )
        ):
            utils.check_equal(len(questions_batch), len(context_batch))

            try:
                utils.check_batch_size(
                    len(questions_batch),
                    generation_batch_size,
                    len(question_dataloader.dataset),
                )
                utils.check_batch_size(
                    len(context_batch),
                    generation_batch_size,
                    len(question_dataloader.dataset),
                )
            except RuntimeError as err:
                raise utils.add_to_err(
                    err,
                    f"\t- {loop_i = }\n"
                    f"\t- {batch_i = }\n"
                )

            ###############################################################
            # PREPARE GENERATION INPUTS
            ###############################################################
            # Take the contexts and append them to the questions
            gen_inputs_text = []
            for question, selected_ids in more_itertools.zip_equal(
                questions_batch,
                context_batch,
            ):
                contexts = [
                    all_passages[ids_].text
                    for ids_ in selected_ids
                ]

                generation_input = (
                    question + tokenizer_bart.sep_token +
                    tokenizer_bart.sep_token.join(contexts)
                )

                gen_inputs_text.append(generation_input)

            utils.check_batch_size(
                len(gen_inputs_text),
                generation_batch_size,
                len(question_dataloader.dataset),
            )

            gen_inputs = tokenizer_bart.batch_encode_plus(
                gen_inputs_text,
                return_tensors="pt",
                pad_to_max_length=True,
                max_length=query_aug_input_max_length,
            )

            ###############################################################
            # QUERY_AUG INFERENCE
            ###############################################################

            utils.check_batch_size(
                gen_inputs["input_ids"].shape[0],
                generation_batch_size,
                len(question_dataloader.dataset),
            )

            with torch.cuda.amp.autocast(enabled=False):
                query_aug_ids_batch, query_aug_scores_batch = (
                    decode(
                        model=query_aug_model,
                        batch=gen_inputs,
                        tokenizer=tokenizer_bart,
                        decoding_conf=decoding_conf_query_aug,
                    )
                )

            try:
                utils.check_batch_size(
                    query_aug_ids_batch.shape[0],
                    generation_batch_size,
                    len(question_dataloader.dataset),
                )
                utils.check_batch_size(
                    query_aug_scores_batch.shape[0],
                    generation_batch_size,
                    len(question_dataloader.dataset),
                )

            except RuntimeError as err:
                raise utils.add_to_err(err,
                    f"{query_aug_ids_batch.shape = }\n" +
                    f"{query_aug_scores_batch.shape = }\n" +
                    f"{gen_inputs['input_ids'].shape = }\n"
                )

            query_aug_text_batch = []
            for (
                question, query_aug_input, query_aug_ids_per_question
            ) in more_itertools.zip_equal(
                questions_batch,
                gen_inputs["input_ids"],
                query_aug_ids_batch,
            ):
                texts_per_question = []
                for generation in query_aug_ids_per_question:
                    gen = tokenizer_bart.decode(generation)
                    cleaned = clean_bart_decode(gen, tokenizer_bart)
                    texts_per_question.append(cleaned)
                query_aug_text_batch.append(texts_per_question)

            assert len(query_aug_text_all_loops) == loop_i + 1, (
                len(query_aug_text_all_loops), loop_i + 1
            )
            assert len(query_aug_score_all_loops) == loop_i + 1, (
                len(query_aug_score_all_loops), loop_i + 1
            )

            query_aug_text_batch = np.array(
                query_aug_text_batch, dtype="object"
            )

            # Make sure that query_aug_text_batch are of the expected shape
            utils.check_shape(
                query_aug_text_batch.shape,
                (
                    query_aug_ids_batch.shape[0],
                    decoding_conf_query_aug.num_return_sequences
                )
            )

            # The quantity of query aug to query score should be 1:1
            utils.check_equal(
                query_aug_text_batch.shape[0],
                query_aug_scores_batch.shape[0],
            )

            # Accumulate the query augmentation text by loop
            query_aug_text_all_loops[loop_i].extend(
                query_aug_text_batch
            )

            # Accumulate the query auggmentation generation score per loop
            query_aug_score_all_loops[loop_i].extend(
                query_aug_scores_batch
            )

            ###############################################################
            # READER INFERENCE
            ###############################################################
#                 reader_batch_ids, reader_batch_scores = (
#                     iterated_retrieval.decode(
#                         model=reader_model,
#                         batch=gen_inputs,
#                         tokenizer=tokenizer_bart,
#                         decoding_conf=decoding_conf_reader,
#                     )
#                 )
#                 reader_batch_ids = (
#                     reader_batch_ids.squeeze(1)
#                 )
#                 # Decode the tokens of the batch
#                 reader_text_batch = []
#                 for (
#                     question, query_aug_input, generations_ids, scores
#                 ) in more_itertools.zip_equal(
#                     questions_batch,
#                     gen_inputs["input_ids"],
#                     reader_batch_ids,
#                     reader_batch_scores,
#                 ):
#                     reader_text_batch.append(
#                         ir.clean_bart_decode(
#                             tokenizer_bart.decode(generations_ids),
#                             tokenizer_bart
#                         )
#                     )

            ###############################################################
            # Deal with the generated text: reader inference
            ###############################################################
            # iterated_retrieval.write_generations(
            #     reader_text_batch,
            #     output_paths["reader_outs"],
            # )
            write_generations(
                query_aug_text_batch.tolist(),
                output_paths["q_aug_outs"],
            )
            write_generations(
                gen_inputs_text,
                output_paths["gen_inputs"],
            )

        return query_aug_text_all_loops, query_aug_score_all_loops

###############################################################################
# Inference
###############################################################################
# @beartype.beartype
def inference(
    all_passages: Dict[str, str],
    query_aug_model: train_generator.SummarizationTrainer,
    reader_model: Optional[train_generator.SummarizationTrainer],
    special_query_token: Optional[str],
    retriever: dense_retriever.LocalFaissRetriever,
    question_dataloader: torch.utils.data.DataLoader,
    max_loop_n: int,
    decoding_conf_reader,
    decoding_conf_query_aug,
    query_aug_input_max_length: int,
    n_docs: int,
    out_path: Union[str, Path],
    retriever_batch_size: int,
    aug_method: str,
    final_num_contexts: int,
    generation_batch_size: int,
    selection_mode: str,
    tokenizer_bart: Union[transformers.BartTokenizer, transformers.BartTokenizerFast],
    tokenizer_bert: Union[transformers.BertTokenizer, transformers.BertTokenizerFast],
    augmentation_mode: Union[AugmentationModes, str],
    oracle_mode: bool,
    retrieval_max_size: int,
) -> None:

    selection_context = SELECTION_TECHNIQUE_TYPES[selection_mode]()

    out_path = Path(out_path)

    if oracle_mode:
        assert max_loop_n == 1, max_loop_n

    # Prepare the output files
    prefixes = dict(
        retr_outs="retr_outs_",
        reader_outs="reader_outs_",
        q_aug_outs="q_aug_outs_",
        gen_inputs="gen_inputs_",
        retr_inputs="retr_inputs_",
    )

    for prefix in prefixes.values():
        for path in out_path.glob(f"{prefix}*.jsonl"):
            LOGGER.info(f"Deleting path: {path}")
            os.remove(path)

    with torch.inference_mode(True):
        query_aug_text_all_loops = []
        query_aug_score_all_loops = []

        for loop_i in range(max_loop_n):
            LOGGER.info(f"{loop_i = }")
            output_paths = {}

            for name, prefix in prefixes.items():
                output_paths[name] = out_path / f"{prefix}{loop_i}.jsonl"

            ###################################################################
            # PREPARE THE RETRIEVAL QUERIES
            ###################################################################
            all_queries_this_loop = InferenceFunctions.prepare_the_queries(
                question_dataloader=question_dataloader, 
                tokenizer_bart=tokenizer_bart, 
                loop_i=loop_i, 
                query_aug_text_all_loops=query_aug_text_all_loops, 
                query_aug_score_all_loops=query_aug_score_all_loops, 
                aug_method=aug_method, decoding_conf_query_aug=decoding_conf_query_aug, 
                augmentation_mode=augmentation_mode, 
                tokenizer_bert=tokenizer_bert, 
                output_paths=output_paths,
                oracle_mode=oracle_mode,
            )
            

            ###################################################################
            # RETRIEVE
            ###################################################################
            retrieved_this_loop = InferenceFunctions.retrieve(
                loop_i=loop_i,
                retriever_batch_size=retriever_batch_size,
                decoding_conf_query_aug=decoding_conf_query_aug,
                all_queries_this_loop=all_queries_this_loop,
                all_query_augs_this_loop=query_aug_text_all_loops[loop_i],
                query_aug_score_all_loops=query_aug_score_all_loops,  # This is weird
                aug_method=aug_method,
                retriever=retriever,
                all_passages=all_passages,
                special_query_token=special_query_token,
                n_docs=n_docs,
                final_num_contexts=final_num_contexts,
                selection_mode=selection_mode,
                output_paths=output_paths,
                selection_context=selection_context,
                oracle_mode=oracle_mode,
                retrieval_max_size=retrieval_max_size,
            )
            
            if oracle_mode:
                break

            ###################################################################
            # Generation with the Barts.
            ###################################################################
            # Splitting per bart model would maybe be better.
            query_aug_text_all_loops, query_aug_score_all_loops = InferenceFunctions.generate(
                loop_i=loop_i,
                query_aug_text_all_loops=query_aug_text_all_loops,
                query_aug_score_all_loops=query_aug_score_all_loops,
                retrieved_this_loop=retrieved_this_loop,
                question_dataloader=question_dataloader,
                generation_batch_size=generation_batch_size,
                tokenizer_bart=tokenizer_bart,
                all_passages=all_passages,
                query_aug_model=query_aug_model,
                query_aug_input_max_length=query_aug_input_max_length,
                decoding_conf_query_aug=decoding_conf_query_aug,
                output_paths=output_paths,
                oracle_mode=oracle_mode,
            )


def build_args(
    config_path: Union[str, Path], 
    root_path: Union[str, Path], 
    run_name: str,
    apply_file_modifications: bool = True,
):
    config_path = Path(config_path)
    root_path = Path(root_path)
    config = utils.load_json(config_path)
    LOGGER.info(f"Loaded config from {config_path}")

    RUN_NAME = run_name
    ORACLE_MODE = config.pop("oracle_mode", False)
    # A lot of stuff because unnecessary if `oracle_mode` is on.
    DATALOADER_MAX_SOURCE_LEN = config.pop("dataloader_max_source_len")
    FINAL_NUM_CONTEXTS = config.pop("final_num_contexts")
    N_DOCS = config.pop("n_docs")
    
    SELECTION_MODE = config.pop("selection_mode")
    AUGMENTATION_MODE = config.pop("augmentation_mode")
    QUERY_AUG_MODEL_TYPE = config.pop("query_aug_model_type")

    if ORACLE_MODE:
        MAX_LOOP_N = 1
        DECODING_CONF_QUERY_AUG = None
    else:
        MAX_LOOP_N = config.pop("max_loop_n")
        DECODING_CONF_QUERY_AUG = DecoderConf(**config.pop("decoding_conf_query_aug"))
    
    assert len(config) == 0, config    

    ###############################################################################
    # Likely fixed
    ###############################################################################
    RETRIEVAL_MAX_SIZE = 15
    
    SENTENCE_DATA_DIR = root_path / "GAR" / "data" / "nq-sentence"
    SENTENCE_MODEL = root_path / "GAR/gar/outputs/sentence_with_context/epoch=38-step=3080.ckpt"
    ANSWER_DATA_DIR = root_path / "GAR" / "data" / "nq-answer"
    ANSWER_MODEL = root_path / "GAR/gar/outputs/answer_with_context_1/epoch=17-step=1421.ckpt"
    
    RETRIEVER_BATCH_SIZE = math.ceil(15 / FINAL_NUM_CONTEXTS)
    MAX_SOURCE_LEN = 768
    QUERY_AUG_INPUT_MAX_LEN = 768
    GENERATION_BATCH_SIZE = 5

    if ORACLE_MODE:
        DATALOADER_MAX_TARGET_LEN = 30
        QUERY_AUG_MODEL_PATH = None
        READER_MODEL_PATH = None
        DECODING_CONF_READER = None
        
        if QUERY_AUG_MODEL_TYPE == DatasetType.SENTENCE:
            DATA_DIR = SENTENCE_DATA_DIR  
        elif QUERY_AUG_MODEL_TYPE == DatasetType.ANSWER:
            DATA_DIR = ANSWER_DATA_DIR
        else:
            DatasetType.error(QUERY_AUG_MODEL_TYPE)
    else:
        READER_MODEL_PATH = ANSWER_MODEL
        DATALOADER_MAX_TARGET_LEN = 2  # We don't need the targets. Needs to be 2.

        if QUERY_AUG_MODEL_TYPE == DatasetType.SENTENCE:
            DATA_DIR = SENTENCE_DATA_DIR  
            QUERY_AUG_MODEL_PATH = SENTENCE_MODEL
        elif QUERY_AUG_MODEL_TYPE == DatasetType.ANSWER:
            DATA_DIR = ANSWER_DATA_DIR
            QUERY_AUG_MODEL_PATH = ANSWER_MODEL
        else:
            DatasetType.error(QUERY_AUG_MODEL_TYPE)

        # DON'T CHANGE THIS.
        DECODING_CONF_READER = DecoderConf(
            num_beams=1, # DON'T CHANGE THIS
            max_length=160, # DON'T CHANGE THIS
            # repetition_penalty=2.5, # DON'T CHANGE THIS
            # length_penalty=1.0, # DON'T CHANGE THIS
            num_return_sequences=1, # DON'T CHANGE THIS
            early_stopping=True, # DON'T CHANGE THIS
            repetition_penalty=3.0, # DON'T CHANGE THIS
        )
        # DON'T CHANGE THIS
        

    ###############################################################################
    # Fixed config
    ###############################################################################
    CV_SET = "val"
    DPR_CONF_PATH = root_path / "DPR" / "conf"
    OUTPUT_ROOT = root_path / "jobs" / "retrieve_and_decode" / "iterated_decoding_output"
    assert OUTPUT_ROOT.exists(), OUTPUT_ROOT
    out_path = OUTPUT_ROOT / f"{utils.timestamp()}_{RUN_NAME}"

    ###############################################################################
    # Fixed logic
    ###############################################################################
    if not ORACLE_MODE:
        assert isinstance(DECODING_CONF_QUERY_AUG.temperature, (float, type(None))), (
            type(DECODING_CONF_QUERY_AUG.temperature).mro()
        )
        assert ANSWER_MODEL.exists(), ANSWER_MODEL
        assert READER_MODEL_PATH.exists(), READER_MODEL_PATH
        assert QUERY_AUG_MODEL_PATH.exists(), QUERY_AUG_MODEL_PATH
        assert SENTENCE_MODEL.exists(), SENTENCE_MODEL


    assert N_DOCS <= FINAL_NUM_CONTEXTS, f"{N_DOCS = }, {FINAL_NUM_CONTEXTS = }"
    assert isinstance(RETRIEVER_BATCH_SIZE, int), type(RETRIEVER_BATCH_SIZE).mro()
    assert DATA_DIR.exists(), DATA_DIR
    assert ANSWER_DATA_DIR.exists(), ANSWER_DATA_DIR
    assert SENTENCE_DATA_DIR.exists(), SENTENCE_DATA_DIR

    AUG_METHOD = "RETRIEVE_ALL_INDIVIDUALLY"

    dpr_cfg = common_retriever.build_cfg(str(DPR_CONF_PATH))

    args = dict(
        conf_path=DPR_CONF_PATH,
        data_dir=DATA_DIR,
        query_aug_model_path=QUERY_AUG_MODEL_PATH,
        reader_model_path=READER_MODEL_PATH,
        dataloader_max_target_len=DATALOADER_MAX_TARGET_LEN,
        dataloader_max_source_len=DATALOADER_MAX_SOURCE_LEN,
        generation_batch_size=GENERATION_BATCH_SIZE,
        max_loop_n=MAX_LOOP_N,
        n_docs=N_DOCS,
        max_source_len=MAX_SOURCE_LEN,
        query_aug_input_max_len=QUERY_AUG_INPUT_MAX_LEN,
        decoding_conf_reader=DECODING_CONF_READER,
        decoding_conf_query_aug=DECODING_CONF_QUERY_AUG,
        out_path=out_path,
        retriever_batch_size=RETRIEVER_BATCH_SIZE,
        aug_method=AUG_METHOD,
        final_num_contexts=FINAL_NUM_CONTEXTS,
        cv_set=CV_SET,
        selection_mode=SELECTION_MODE,
        augmentation_mode=AUGMENTATION_MODE,
        oracle_mode=ORACLE_MODE,
        retrieval_max_size=RETRIEVAL_MAX_SIZE,
    )

    if apply_file_modifications:
        if out_path.exists():
            shutil.rmtree(out_path)
        out_path.mkdir()

        json_output_config = dict(
            indent=2,
        )

        # We write it instead of copying it in case it's been modified
        utils.save_json(
            config, 
            out_path / "config.json", 
            **json_output_config,
        )
        utils.save_json(
            args,
            out_path / "notebook_args.json",
            **json_output_config
        )
        utils.save_json(
            omegaconf.OmegaConf.to_container(dpr_cfg),
            out_path / "dpr_config.json",
            **json_output_config
        )

    return argparse.Namespace(**args), dpr_cfg


if __name__ == "__main__":
    ROOT_PATH = Path("/home/mila/g/gagnonju/IteratedDecoding/")
    args, dpr_cfg = build_args(ROOT_PATH, apply_file_modifications=False)
    def default(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, DecoderConf):
            return vars(obj)
        else:
            raise ValueError(type(obj).mro())

    print(json.dumps(vars(args), default=default, indent=4))
