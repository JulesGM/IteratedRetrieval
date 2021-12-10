import json
import logging
import os
from pathlib import Path
import sys

import colorama
import fire
import horovod.torch as hvd
import numpy as np
import torch
import transformers
import tqdm

LOGGER = logging.getLogger(__name__)

ROOT_PATH = Path("/home/mila/g/gagnonju/IteratedDecoding/")

format_info = (
    "[%(levelname)s] (%(asctime)s) "
    "{%(name)s.%(funcName)s:%(lineno)d}:\n"
)

logging_format = (
    colorama.Fore.CYAN +
    format_info +
    colorama.Style.RESET_ALL +
    "%(message)s"
)
logging.basicConfig(
    format=logging_format,
    level=logging.WARNING,
    force=True,
)

SCRIPT_DIR = Path(__file__).absolute().parent

sys.path.insert(0, str(SCRIPT_DIR / "retrieve_and_decode"))
import iterated_retrieval as ir

DPR_PATH = ROOT_PATH / "DPR"
sys.path.insert(0, str(DPR_PATH))
import utils_gen

GAR_PATH = ROOT_PATH / "GAR/gar"
sys.path.insert(0, str(GAR_PATH))
import train_generator


def main(distributed=True):

    if distributed:
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())

        # if hvd.rank() == 0:
        #     print("Loading args.")
        #     args, _ = ir.build_args(ROOT_PATH, apply_file_modifications=False)
        #     print("Done loading args.")
        # else:
        #     args = None
        # args = hvd.broadcast_object(args, root_rank=0)
        
    args, _ = ir.build_args(ROOT_PATH, apply_file_modifications=False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "facebook/bart-large"
    )

    dataset = utils_gen.SummarizationDataset(
        tokenizer,
        type_path=args.cv_set,
        data_dir=args.data_dir,
        max_source_length=args.dataloader_max_source_len,
        max_target_length=args.dataloader_max_target_len,
    )

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas=hvd.size(), 
            rank=hvd.rank()
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.generation_batch_size, sampler=sampler,
        )

    else:    
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            collate_fn=dataset.collate_fn,
            batch_size=args.generation_batch_size, 
        )

    model = train_generator.SummarizationTrainer.load_from_checkpoint(
        str(args.query_aug_model_path)
    ).cuda()

    if distributed:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    with torch.inference_mode():
        for batch in tqdm.tqdm(dataloader):
            batch = dict(input_ids=batch["source_ids"], attention_mask=batch["source_mask"])
            output = ir.decode(
                model,
                batch,
                tokenizer,
                args.decoding_conf_query_aug,
            ) 

            if distributed:
                all_outputs = hvd.allgather(output)
                print(type(all_outputs))

            import pdb; pdb.set_trace()


if __name__ == "__main__":
    fire.Fire(main)
