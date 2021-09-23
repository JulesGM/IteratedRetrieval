# Standard library
import collections
import copy
import glob
import importlib
import logging
import os
import pathlib
import re
import rich
import time

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

# Third party
import hydra
import rich

# First Party
BASE_PATH = pathlib.Path("/home/mila/g/gagnonju/IteratedDecoding/DPR")
CONF_PATH = BASE_PATH/"conf"

import common_retriever

os.chdir(BASE_PATH)
import dpr.options
import dpr.utils.model_utils

import dense_retriever
import jules_validate_dense_retriever

LOGGER = logging.getLogger(__name__)

###################################################################
# Check package versions
###################################################################
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
check_version("transformers", (3, 0, 0), (3, 1, 0))
check_version("tqdm", (4, 27, 0))
check_version("spacy", (2, 1, 8))
check_version("hydra", (1, 0, 0))
check_version("omegaconf", (2, 0, 1))
LOGGER.info("All version checks passed.")


###########################################################################
# Main task
###########################################################################
def prep_cfg(cfg, verbose=False):
    ###########################################################################
    # Complete and validate CFG
    ###########################################################################
    jules_validate_dense_retriever.validate(
        {k: getattr(cfg, k) for k in dir(cfg)}, 
        dense_retriever.SCHEMA_PATH,
        verbose,
    )
    cfg = dpr.options.setup_cfg_gpu(cfg)

    assert cfg.out_file, cfg.out_file
    assert pathlib.Path(cfg.out_file).parent.exists(), cfg.out_file

    if verbose:
        LOGGER.info("CFG (after gpu  configuration):")
        LOGGER.info("%s", OmegaConf.to_yaml(cfg))
    
    
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

def load_passages(cfg):
    cfg = copy.copy(cfg)        
    prep_cfg(cfg)
    ###########################################################################
    # Prepare sources
    ###########################################################################
    LOGGER.info("Loading passages.")
    all_passages = {}
    id_prefixes = []
    ctx_sources = []
    for ctx_src in cfg.ctx_datatsets:
        ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
        id_prefixes.append(ctx_src.id_prefix)
        ctx_sources.append(ctx_src)

    all_passages = {}
    for ctx_src in ctx_sources:
        ctx_src.load_data_to(all_passages)
    LOGGER.info("Done loading passages.")

    return all_passages, id_prefixes

###########################################################################
# Build the retriever
###########################################################################
def make_retriever(cfg, id_prefixes):
    cfg = copy.copy(cfg)        
    prep_cfg(cfg)

    ###########################################################################
    # Prepare models
    ###########################################################################
    saved_state = dense_retriever.load_states_from_checkpoint(cfg.model_file)
    dpr.options.set_cfg_params_from_state(saved_state.encoder_params, cfg)

    tensorizer, encoder, _ = dense_retriever.init_biencoder_components(
        cfg.encoder.encoder_model_type, cfg, inference_only=True
    )

    encoder_path = cfg.encoder_path
    if encoder_path:
        LOGGER.info("Selecting encoder: %s", encoder_path)
        encoder = getattr(encoder, encoder_path)
    else:
        LOGGER.info("Selecting standard question encoder")
        encoder = encoder.question_model

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
    model_to_load = dpr.utils.model_utils.get_model_obj(encoder)
    LOGGER.info("Loading saved model state ...")

    encoder_prefix = (
        encoder_path if encoder_path else "question_model") + "."
    prefix_len = len(encoder_prefix)

    LOGGER.info("Encoder state prefix %s", encoder_prefix)
    question_encoder_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith(encoder_prefix)
    }
    # TODO: long term HF state compatibility fix
    model_to_load.load_state_dict(question_encoder_state, strict=False)
    vector_size = model_to_load.get_out_size()
    LOGGER.info("Encoder vector_size=%d", vector_size)

    
    ###########################################################################
    # Load Index & Prepare retriever
    ###########################################################################
    index_path = cfg.index_path
    #------------
    ## Instantiate the index and create a retriever
    #------------
    LOGGER.info(f"Loading index.")
    index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
    LOGGER.info("Index class %s ", type(index))
    index_buffer_sz = index.buffer_size
    index.init_index(vector_size)
    LOGGER.info(f"Done loading index.")
    LOGGER.info(f"Loading retriever.")
    retriever = dense_retriever.LocalFaissRetriever(
        encoder, 
        cfg.batch_size, 
        tensorizer, 
        index,
    )
    LOGGER.info(f"Loaded retriever.")
            
    #------------
    ## Index all passages
    #------------
    ctx_files_patterns = cfg.encoded_ctx_files

    LOGGER.info("ctx_files_patterns: %s", ctx_files_patterns)
    if ctx_files_patterns:
        assert len(ctx_files_patterns) == len(id_prefixes), (
            "ctx len={} pref leb={}".format(
            len(ctx_files_patterns), 
            len(id_prefixes)
            )
        )
    else:
        assert index_path, (
            "Either encoded_ctx_files or index_path parameter should be set."
        )

        
    input_paths = []
    path_id_prefixes = []
    for i, pattern in enumerate(ctx_files_patterns):
        pattern_files = glob.glob(pattern)
        pattern_id_prefix = id_prefixes[i]
        input_paths.extend(pattern_files)
        path_id_prefixes.extend([pattern_id_prefix] * len(pattern_files))
        
      
    if index_path and index.index_exists(index_path):
        LOGGER.info("Index path: %s", index_path)
        retriever.index.deserialize(index_path)
    else:
        LOGGER.info("Reading all passages data from files: %s", input_paths)
        retriever.index_encoded_data(
            input_paths, 
            index_buffer_sz, 
            path_id_prefixes=path_id_prefixes,
        )
        if index_path:
            retriever.index.serialize(index_path)
            
    LOGGER.info("Embeddings files id prefixes: %s", path_id_prefixes)
    return retriever


def retrieve(
    retriever, 
    all_passages, 
    questions, 
    special_query_token, 
    n_docs,
):
    if len(all_passages) == 0:
        raise RuntimeError("No passages data found.")
        
    ###########################################################################
    # Get top k results.
    ###########################################################################
    LOGGER.info("Using special token %s", special_query_token)
    
    start = time.monotonic()
    questions_tensor = retriever.generate_question_vectors(
        questions, 
        query_token=special_query_token,
    )
    delta = time.monotonic() - start
    LOGGER.info(f"{len(questions) / delta} tok/sec for batch size {retriever.batch_size}")
    
    LOGGER.info(f"get_top_docs: Starting.")
    top_ids_and_scores = retriever.get_top_docs(
        questions_tensor.numpy(), 
        n_docs,
    )
    LOGGER.info("get_top_docs: Done.")
    
    return top_ids_and_scores