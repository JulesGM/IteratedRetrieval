print("(Re)/Loading common_retriever.py")

# Standard library
import collections
import copy
import glob
import importlib
import logging
import os
import pathlib
try:
    import ujson as json
except ImportError:
    import json
    
from pathlib import Path
import re
import rich
import time
import sys

SCRIPT_DIR = Path(__file__).resolve().parent

# Third party
import faiss
import hydra
import jsonlines
import numpy as np
import omegaconf
import rich
import torch
import transformers

# First Party
import iterated_utils as utils

ROOT_PATH = SCRIPT_DIR.parent.parent
DPR_PATH = ROOT_PATH  / "DPR"
CONF_PATH = DPR_PATH / "conf"
sys.path.insert(0, str(DPR_PATH))

import dpr.options
import dpr.utils.model_utils
import dense_retriever
import jules_validate_dense_retriever

sys.path.insert(0, str(ROOT_PATH / "GAR" / "gar"))              
import utils_gen


LOGGER = logging.getLogger(__name__)
LOGGER.info(f"(Re)loaded {Path(__file__).name}")
LOGGER.info(f"{dense_retriever.__file__ = }")

###########################################################
# Check package versions
###########################################################
_parse_version_version_string_pat = re.compile(
    r"^([0-9]+\.[0-9]+\.[0-9]+)"
)


def _parse_version(version_string):
    version_string = _parse_version_version_string_pat.match(
        version_string
    ).group(1)
    return tuple(map(int, version_string.split(".")))


def check_version(module_name, lb=None, ub=None):
    assert lb or ub, (lb, ub)
    LOGGER.debug(f"Importing {module_name}...")
    module = importlib.__import__(module_name)
    LOGGER.debug(f"Done importing {module_name}.")
    version = _parse_version(module.__version__)
    if lb:
        assert lb <= version, (lb, version)
    if ub:
        assert version < ub, (ub, version)


LOGGER.info("Checking versions...")
check_version("torch", (1, 5, 0))
# check_version("transformers", (3, 0, 0), (3, 1, 0))
check_version("tqdm", (4, 27, 0))
check_version("spacy", (2, 1, 8))
check_version("hydra", (1, 0, 0))
check_version("omegaconf", (2, 0, 1))
LOGGER.info("All version checks passed.")


###########################################################
# Main task
###########################################################
def prep_cfg(cfg, verbose=False):
    #######################################################
    # Complete and validate CFG
    #######################################################
    jules_validate_dense_retriever.validate(
        {k: getattr(cfg, k) for k in dir(cfg)},
        dense_retriever.SCHEMA_PATH,
        verbose,
    )
    cfg = dpr.options.setup_cfg_gpu(cfg)

    assert cfg.out_file, cfg.out_file
    assert Path(cfg.out_file).parent.exists(), cfg.out_file

    if verbose:
        LOGGER.info("CFG (after gpu  configuration):")
        LOGGER.info("%s", omegaconf.OmegaConf.to_yaml(cfg))


LoadDataReturn = collections.namedtuple(
    "LoadDataReturn",
    [
        "questions",
        "question_answers",
        "special_query_token",
    ]
)


def load_data(cfg):
    cfg = copy.copy(cfg)

    prep_cfg(cfg)

    questions = []
    question_answers = []

    if not cfg.qa_dataset:
        LOGGER.warning("Please specify qa_dataset to use")
        return

    ds_key = cfg.qa_dataset
    LOGGER.info("qa_dataset: %s", ds_key)

    LOGGER.info(
        "load_data: hydra.utils.instantiate(cfg.datasets[ds_key])"
    )
    qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    qa_src.load_data()
    assert not qa_src.selector, qa_src.selector
    LOGGER.info("Using custom representation token selector")

    for ds_item in qa_src.data:
        question, answers = ds_item.query, ds_item.answers
        questions.append(question)
        question_answers.append(answers)

    return LoadDataReturn(
        questions=questions,
        question_answers=question_answers,
        special_query_token=qa_src.special_query_token,
    )


def build_args(root_path):
    DPR_CONF_PATH = root_path / "DPR" / "conf"
    
    try:
        hydra.initialize_config_dir(config_dir=str(DPR_CONF_PATH))
    except ValueError as err:
        message = (
            "GlobalHydra is already initialized, call "
            "GlobalHydra.instance().clear() if you want to re-initialize"
        )
        if message not in err.args[0]:
            raise err


    dpr_cfg = hydra.compose(
        config_name="dense_retriever",
        overrides=["out_file=/tmp/"],
    )

    return dpr_cfg


def load_passages(cfg):
    cfg = copy.copy(cfg)
    prep_cfg(cfg)
    #######################################################
    # Prepare sources
    #######################################################
    LOGGER.debug("hydra.utils.instantiate")
    all_passages = {}
    id_prefixes = []
    ctx_sources = []
    for ctx_src in cfg.ctx_datatsets:
        ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
        id_prefixes.append(ctx_src.id_prefix)
        ctx_sources.append(ctx_src)

    LOGGER.debug("ctx_src.load_data_to")
    all_passages = {}
    for ctx_src in ctx_sources:
        ctx_src.load_data_to(all_passages)

    LOGGER.info("Done loading passages.")
    return all_passages, id_prefixes


###########################################################
# Build the retriever
###########################################################
def make_retriever(cfg, id_prefixes):
    cfg = copy.copy(cfg)
    prep_cfg(cfg)

    #######################################################
    # Prepare models
    #######################################################
    LOGGER.info("Loading encoders.")
    saved_state = dense_retriever.load_states_from_checkpoint(
        cfg.model_file
    )
    dpr.options.set_cfg_params_from_state(
        saved_state.encoder_params, 
        cfg,
    )

    tensorizer, encoder, _ = dense_retriever.init_biencoder_components(
        cfg.encoder.encoder_model_type, cfg, inference_only=True
    )

    encoder_path = cfg.encoder_path
    if encoder_path:
        LOGGER.debug("Selecting encoder: %s", encoder_path)
        encoder = getattr(encoder, encoder_path)
    else:
        LOGGER.debug("Selecting standard question encoder")
        encoder = encoder.question_model

    LOGGER.debug(
        "dpr.utils.model_utils.setup_for_distributed_mode"
    )
    encoder, _ = dpr.utils.model_utils.setup_for_distributed_mode(
        encoder,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16
    )
    encoder.eval()

    # load weights from the model file
    LOGGER.debug("dpr.utils.model_utils.get_model_obj")
    model_to_load = dpr.utils.model_utils.get_model_obj(encoder)
    encoder_prefix = (
        encoder_path if encoder_path else "question_model"
    ) + "."
    prefix_len = len(encoder_prefix)

    LOGGER.debug("Encoder state prefix %s", encoder_prefix)
    question_encoder_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith(encoder_prefix)
    }
    # TODO: long term HF state compatibility fix
    LOGGER.debug("model_to_load.load_state_dict")
    model_to_load.load_state_dict(question_encoder_state, strict=False)
    vector_size = model_to_load.get_out_size()
    LOGGER.debug("Encoder vector_size=%d", vector_size)

    #######################################################
    # Load Index & Prepare retriever
    #######################################################
    index_path = cfg.index_path
    #------------------------------------------------------
    # Instantiate the index and create a retriever
    #------------------------------------------------------
    LOGGER.debug(
        "Loading index. "
        "hydra.utils.instantiate(cfg.indexers[cfg.indexer])"
    )
    index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
    LOGGER.debug("Index class %s ", type(index))
    index_buffer_sz = index.buffer_size
    index.init_index(vector_size)
    LOGGER.debug("dense_retriever.LocalFaissRetriever")
    retriever = dense_retriever.LocalFaissRetriever(
        encoder,
        cfg.batch_size,
        tensorizer,
        index,
    )
    LOGGER.debug("Loaded retriever.")

    #------------------------------------------------------
    # Index all passages
    #------------------------------------------------------
    ctx_files_patterns = cfg.encoded_ctx_files

    LOGGER.debug(f"ctx_files_patterns: {ctx_files_patterns}")
    if ctx_files_patterns:
        assert len(ctx_files_patterns) == len(id_prefixes), (
            "ctx len={} pref leb={}".format(
                len(ctx_files_patterns),
                len(id_prefixes)
            )
        )
    else:
        assert index_path, (
            "Either encoded_ctx_files or index_path parameter "
            "should be set."
        )

    input_paths = []
    path_id_prefixes = []
    for i, pattern in enumerate(ctx_files_patterns):
        pattern_files = glob.glob(pattern)
        pattern_id_prefix = id_prefixes[i]
        input_paths.extend(pattern_files)
        path_id_prefixes.extend([pattern_id_prefix] * len(pattern_files))

    if index_path and index.index_exists(index_path):
        LOGGER.info(
            f"retriever.index.deserialize {index_path},"
            " takes 11 min"
        )
        retriever.index.deserialize(index_path)
    else:
        LOGGER.info(f"retriever.index_encoded_data {input_paths}")
        retriever.index_encoded_data(
            input_paths,
            index_buffer_sz,
            path_id_prefixes=path_id_prefixes,
        )
        if index_path:
            LOGGER.info("retriever.index.serialize")
            retriever.index.serialize(index_path)

    LOGGER.info(f"Emb. files id prefixes: {path_id_prefixes}")
    return retriever


def retrieve(
    retriever,
    all_passages,
    questions,
    special_query_token,
    n_docs,
):
    batch_size = len(questions)

    if len(all_passages) == 0:
        raise RuntimeError("No passages data found.")

    #######################################################
    # Get top k results.
    #######################################################

    start = time.monotonic()
    questions_tensor = retriever.generate_question_vectors(
        questions,
        query_token=special_query_token,
    )
    delta = time.monotonic() - start
    LOGGER.debug(
        f"Bert total time was {delta:0.2f}s. "
        f"Bert has {batch_size / delta:0.2f} seq/sec for tensor "
        f"{questions_tensor.size()} and inner batch size "
        f"{retriever.batch_size}"
    )
    LOGGER.debug(f"get_top_docs: Starting.")
    start = time.monotonic()
    top_ids_and_scores = retriever.get_top_docs(
        questions_tensor.numpy(),
        n_docs,
    )
    delta = time.monotonic() - start
    LOGGER.debug(
        f"Retrieval took {delta:0.2f} sec. "
        f"Retrieval at {batch_size / delta:0.2f} input "
        f"vectors/s with {n_docs} neighbors each."
    )
    return top_ids_and_scores


def build_retriever(cfg):
    with utils.time_this("common_retriever.load_passages (~6 min)"):
        all_passages, id_prefixes = load_passages(
            cfg,
        )

    with utils.time_this("common_retriever.make_retriever (~11 min.)"):
        retriever = (
            make_retriever(cfg, id_prefixes)
        )

    with utils.time_this("common_retriever.load_data"):
        questions, question_answers, special_query_token = (
            load_data(cfg)
        )

        n_docs = cfg.n_docs
    return retriever, all_passages, special_query_token


def faiss_to_gpu(index):
    if isinstance(index, faiss.swigfaiss.IndexShards):
        LOGGER.info("Index is alerady sharded on GPUs.")
        return index

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True

    LOGGER.info(f"{torch.cuda.device_count()} GPUs")

    with utils.time_this("moving to all GPUs, ~20 sec with 4 A100"):
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    return index

