#!/usr/bin/env python
# coding: utf-8


CONFIG_RAN_ALL_THE_WAY = False

###############################################################################
# Imports
###############################################################################
# Standard library
import argparse
import collections
import dataclasses
import logging
import math
import numpy as np
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import *

# Third party
import beartype
import colorama
import fire
import more_itertools
import omegaconf
import rich
import torch
import tqdm

import transformers
assert int(transformers.__version__.split('.')[0]) >= 4, transformers.__version__

# First Party
import iterated_utils as utils
import iterated_retrieval as ir
import iterated_retrieval
import common_retriever

assert "condaless" in sys.executable, sys.executable

LOGGER = logging.getLogger(__name__)


def main(config_path: str):
    config_path = Path(config_path)
    assert config_path.exists(), f"\nconfig_path does not exist: {config_path}"

    ROOT_PATH = Path("/home/mila/g/gagnonju/IteratedDecoding/")

    ###############################################################################
    # Logging
    ###############################################################################

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
    logging.getLogger(
        "transformers.configuration_utils"
    ).setLevel(logging.WARN)
    logging.getLogger(
        "transformers.tokenization_utils"
    ).setLevel(logging.WARN)
    logging.getLogger(
        "transformers.modeling_utils"
    ).setLevel(logging.WARN)
    logging.getLogger(
        "common_retriever"
    ).setLevel(logging.INFO)
    logging.getLogger(
        "dense_retriever"
    ).setLevel(logging.INFO)


    ###############################################################################
    # CONFIG
    ###############################################################################
    args, dpr_cfg = ir.build_args(config_path, ROOT_PATH, apply_file_modifications=True)

    rich.print(f"args:\n{vars(args)}")

    (
        dataloader, tokenizer_bart, tokenizer_bert,
    ) = iterated_retrieval.build_tokenizers_and_datasets(
        generation_batch_size=args.generation_batch_size,
        data_dir=args.data_dir,
        max_target_len=args.dataloader_max_target_len,
        max_source_len=args.dataloader_max_source_len,
        cv_set=args.cv_set,

    )

    retriever, all_passages, special_query_token = common_retriever.build_retriever(
        dpr_cfg,
        ROOT_PATH / "jobs" / "retrieve_and_decode" / "cache" 
    )
    retriever.index.index = common_retriever.faiss_to_gpu(
        retriever.index.index,
    )

    query_aug_model, reader_model = ir.build_models(
        reader_model_path=args.reader_model_path,
        query_aug_model_path=args.query_aug_model_path,
    )

    ir.inference(
        all_passages=all_passages,
        query_aug_model=query_aug_model.cuda(),
        reader_model=reader_model.cuda() if reader_model else None,
        special_query_token=special_query_token,
        retriever=retriever,
        selection_technique=ir.selection_technique,
        question_dataloader=dataloader,
        max_loop_n=args.max_loop_n,
        query_aug_input_max_length=args.max_source_len,
        decoding_conf_query_aug=args.decoding_conf_query_aug,
        decoding_conf_reader=args.decoding_conf_reader,
        n_docs=args.n_docs,
        out_path=args.out_path,
        retriever_batch_size=args.retriever_batch_size,
        aug_method=args.aug_method,
        final_num_contexts=args.final_num_contexts,
        generation_batch_size=args.generation_batch_size,
        selection_mode=args.selection_mode,
        tokenizer_bart=tokenizer_bart,
        tokenizer_bert=tokenizer_bert,
        augmentation_mode=args.augmentation_mode,
    )


if __name__ == "__main__":
    fire.Fire(main)