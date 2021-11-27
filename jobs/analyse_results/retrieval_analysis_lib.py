import asyncio
import base64
import collections
import functools
import io
import json
import logging
import math
import os
from pathlib import *
import pickle
import re
import shutil
import sys
import time
from typing import *

import beartype
import bs4
import colorama
import fire
import hydra
from IPython.display import display, HTML
import jinja2
import jsonlines as jsonl
from mpl_toolkits import mplot3d
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker
import more_itertools
import nest_asyncio
import numpy as np
import omegaconf
import retrieval_analysis_lib as ra
import pandas as pd
import rich
import rich.console

sys.path.insert(0, "/home/mila/g/gagnonju/IteratedDecoding/DPR/")
import dense_retriever

sys.path.insert(0, "/home/mila/g/gagnonju/IteratedDecoding/jobs/retrieve_and_decode")
import common_retriever
import iterated_utils as utils
print("Done with all imports")

LOGGER = logging.getLogger(__name__)
ROOT = Path("/home/mila/g/gagnonju/IteratedDecoding/")
_NUM_ENTRIES = 5

functools.lru_cache(maxsize=1)
def _get_loop_i_extractor():
    """Lazily compile the pattern, and only do it once."""
    return re.compile(r"retr_outs_(\w+)\.jsonl")


def prep_retrievals_object(input_folder):
    assert input_folder.exists(), input_folder
    retr_outs = list(input_folder.glob("retr_outs_*.jsonl"))
    retr_outs_and_loop_i = [(int(_get_loop_i_extractor().match(path.name).group(1)), path) for path in retr_outs]
    retr_outs_and_loop_i.sort(key=lambda pair: pair[0])
    retr_outs_and_loop_i = retr_outs_and_loop_i

    retrievals = collections.defaultdict(list)

    for loop_i, path in retr_outs_and_loop_i:
        assert loop_i not in retrievals
        with jsonl.open(path) as fin:
            for entry in fin:
                assert isinstance(entry, dict), type(entry).mro()
                assert len(entry) == 1, len(entry)
                retrievals[loop_i].extend(entry["ids"])

    loop_i_values_observed = set(more_itertools.unzip(retr_outs_and_loop_i)[0])
    loop_i_values_expected = set(range(len(retr_outs_and_loop_i)))
    assert loop_i_values_observed == loop_i_values_expected, retr_outs_and_loop_i

    # Check that the number of retrievals per question is the same
    first = None
    for k, v in retrievals.items():
        for entry in v:
            if first is None:
                first = len(entry)
            else:
                assert len(entry) == first, (len(entry, first))
    
    return retrievals
    

def prep_directories(input_folder, root, output_dir_name):
    out_file_directory = root / "jobs" / "analyse_results" / "retrieval_analysis_outputs"
    out_file = out_file_directory / f"{output_dir_name}"


    # Create the output directory, delete it if it already exists
    if out_file.exists():
        shutil.rmtree(out_file)
    os.mkdir(out_file)
    os.mkdir(out_file / "input_dir")

    # Copy the input files to the output directory
    input_copy_dir = out_file / "input_dir" / input_folder.name
    shutil.copytree(input_folder, input_copy_dir)
    
    return out_file, input_copy_dir


def _clean_contents(contents):
    output = [str(x) for x in contents if (x and str(x).strip())]
    return "\n".join(output)


def panda_df_to_html_table(
    pd_actual_values: pd.DataFrame, 
    float_format_fn: str
) -> str:

    pd_html = pd_actual_values.to_html(float_format=float_format_fn)
    pd_bs4 = bs4.BeautifulSoup(pd_html, "html.parser")
    return str(pd_bs4.table)


@functools.lru_cache(maxsize=1)
def _build_untab_pat():
    return re.compile(r"^    ")


def untab_once(text):
    return _build_untab_pat().sub("", text, flags=re.MULTILINE)   


def untab_n_times(text, n):
    for _ in range(n):
        text = untab_once(text)
    return text


def _format_float(value):
    return f"{value:0.0%}"


def plot_retrieval_accuracies(np_actual_values):
    plt.figure(figsize=(10, 10))
    plt.tight_layout(pad=0)
    MODE = "2D"
    if MODE == "3D":
        ax = plt.axes(
            projection="3d", 
            proj_type="ortho"
        )
    else:
        ax = plt.gca()

    for y in range(np_actual_values.shape[0]):
        if MODE == "3D":
            ax.plot3D(
                np.arange(np_actual_values.shape[1]), 
                np.ones(np_actual_values.shape[1]) * y, 
                np_actual_values[y],
                label=str(y)
            )
        else:
            ax.plot(
                np.arange(np_actual_values.shape[1]), 
                np_actual_values[y],
                label=str(y)
            )

    plt.xlabel("Value of K in 'Top-K'")
    ticker = matplotlib.ticker.FuncFormatter(lambda x, pos: f"{x:0.0%}")

    if MODE == "3D":
        ax.view_init(elev=0, azim=270)
        ax.set_zlabel("Cumulative Accuracy")
        plt.yticks([])
        ax.zaxis.set_major_formatter(ticker)
    else:
        ax.set_ylabel("Cumulative Accuracy")
        plt.yticks(np.linspace(0, 1, 11))
        ax.yaxis.set_major_formatter(ticker)
        plt.legend()

    plt.legend()

    ########################################################################
    # Convert the plot to a base64 encoded string for the html file
    ########################################################################
    fake_file = io.BytesIO()
    plt.savefig(fake_file, format="png", dpi=600)
    fake_file.seek(0)
    encoded = base64.b64encode(fake_file.read()).decode("utf-8")
    plt.close()
    return encoded


def prep_accuracies_table_html(accuracies):
    max_num_entries_per_line = 15
    num_splits = math.ceil(accuracies.shape[1] / max_num_entries_per_line)
    pd_splits = np.array_split(
        pd.DataFrame(accuracies), num_splits, axis=1,
    )[::-1]
    tables_gen = lambda: (
        panda_df_to_html_table(x, _format_float) for x in pd_splits
    )
    return "\n".join(
        f"<p>{table}</p><br />" for table in tables_gen()
    )


def prep_display_args_dict(args):
    console = rich.console.Console(record=True)
    
    filtered_args = {
        k: v for k, v in args.items()
        if k not in {
            "generation_batch_size",
            "conf_path",
            "dataloader_max_target_len",
            "max_loop_n",
            "max_source_ken",
            "query_aug_input_max_len",
            "retriever_batch_size",
            "out_path",
            "cv_set",
            "aug_method",
        }
    }
    console.print(filtered_args)
    args_html = console.export_html()
    args_bs4 = bs4.BeautifulSoup(args_html, "html.parser")
    args_style = _clean_contents(args_bs4.head.style.contents)
    args_body= _clean_contents(args_bs4.body.contents)
    return args_body, args_style


def jsonl_to_table(path, id_, description, N):
    import itertools
    with jsonl.open(path) as fin:
        data = [x for x in itertools.islice(fin, N)]
    
    html = (
f"""<p>
    <button id='{id_}_button'>{description}</button>
    <table id='{id_}_table' style='display: none'>
        """)

    for line in data:
        html += "\t\t\t<tr><td>" + str(line) + "</td></tr>\n"
    html += (
""" 
    </table>
</p>""")

    script = f"""
$("#{id_}_button").click(
        function() {{ 
            flip_visibility("#{id_}_table"); 
            
            console.log(`flipped: ${{$("#{id_}_table").css("display")}}`);
        }}
);"""

    return html, script

def _by_it_no(path):
    path = Path(path)
    return int(path.stem.split("_")[-1])

def make_hideable_tables(input_dir: Union[Path, str]):
    input_dir = Path(input_dir)
    
    data_names = ["gen_inputs", "q_aug_outs", "retr_inputs"]
    data_descr = dict(
        gen_inputs="Generated Inputs", 
        q_aug_outs="Augmented Queries", 
        retr_inputs="Retriever Inputs"
    )
    paths = {key: [] for key in data_names}

    for name in data_names:
        paths[name] = sorted(input_dir.glob(f"{name}_*.jsonl"), key=_by_it_no) 
    
    inputs = []
    for data_name in data_names:
        for i, path in enumerate(paths[data_name]):
            idx = _by_it_no(path)
            assert i == idx, (i, idx)
            special_descr = ""

            if i == 0 and data_name == "gen_inputs":
                special_descr = " = Just Questions"

            inputs.append(
                    dict(
                        path=path, 
                        data_name=data_name, 
                        description=f"{data_descr[data_name]} {idx}{special_descr}"
                )
            )

    inputs_converted = [
        jsonl_to_table(
                path=x["path"], 
                id_=x["data_name"], 
                description=x["description"], 
                N=_NUM_ENTRIES,
            ) for x in inputs
    ]

    inputs_html_list, inputs_scripts_list = zip(*inputs_converted)
    inputs_html = "\n".join(inputs_html_list)
    inputs_scripts = "\n".join(inputs_scripts_list)

    return inputs_html, inputs_scripts


@beartype.beartype
def display_results(
    results: dict, 
    output_dir: Union[Path, str], 
    args: Dict[str, Any], 
    input_dir: Union[Path, str]
) -> None:

    input_dir = Path(input_dir)
    results_tuple = sorted(results.items(), key=lambda pair: pair[0])
    actual_keys = set(list(zip(*results_tuple))[0])
    actual_values = list(zip(*results_tuple))[1]
    expected_keys = set(range(len(results_tuple)))
    assert actual_keys == expected_keys, (actual_keys, expected_keys)
    retrieval_accuracies = np.array(actual_values)

    ########################################################################
    # Draw and encode the retrieval accuracies plot
    ########################################################################
    encoded = plot_retrieval_accuracies(retrieval_accuracies)

    ########################################################################
    # Convert the results to a dataframe
    ########################################################################
    accuracies_table_html = prep_accuracies_table_html(retrieval_accuracies)

    ########################################################################
    # Prepare the displaying of the args dict
    ########################################################################
    args_body, args_style = prep_display_args_dict(args)
    
    ########################################################################
    # Make hidable tables and scripts
    ########################################################################
    tables_html, tables_scripts = make_hidable_tables()
    
    ########################################################################
    # Write the html file
    ########################################################################
    

    template_text = (
        ROOT / "jobs" / "analyse_results" / "output_template.jinja2"
    ).read_text().strip()
    template = jinja2.Template(template_text)
    html = template.render(
        args_body=args_body,
        encoded=encoded,
        table_html=accuracies_table_html,
        args_style=args_style,
        inputs_html=inputs_html,
        inputs_scripts=inputs_scripts,
    )
    
    output_file = output_dir / "results.html"
    output_file.write_text(html)
    print(f"Wrote the HTML: {output_file}")


@beartype.beartype
def load_questions_and_answers(
    cfg: omegaconf.DictConfig, ds_key: str
) -> Tuple[List[Any], List[Any]]:
    questions = []
    question_answers = []
    qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    qa_src.load_data()

    for ds_item in qa_src.data:
        question, answers = ds_item.query, ds_item.answers
        questions.append(question)
        question_answers.append(answers)

    return questions, question_answers

def analyse(
    cfg: omegaconf.DictConfig, 
    input_folder_name: str, 
    retrieve_and_decode_output: Union[str, Path], 
    question_answers: List[str], 
    passages: Dict[str, str],
):
    output_dir_name = input_folder_name
    input_folder = retrieve_and_decode_output / input_folder_name
    assert input_folder.exists(), input_folder

    ########################################################################
    # Start working.
    ########################################################################
    out_file, input_copy_dir = prep_directories(input_folder, ROOT, output_dir_name)
    retrievals = prep_retrievals_object(input_copy_dir)


    ########################################################################
    # Validate the retrievals
    ########################################################################
    results = {}
    for loop_i, retrieved in sorted(retrievals.items(), key=lambda pair: pair[0]):
        if len(question_answers) != len(retrieved):
            rich.print(f"Breaking at {loop_i} because the lengths don't match.")
            break
        
        with (out_file / f"validate_{loop_i}.txt").open("w") as output_file:
            output = dense_retriever.validate(
                passages=passages,
                answers=question_answers,
                result_ctx_ids=list(zip(retrieved, [None for _ in retrieved])),
                workers_num=os.cpu_count(),
                match_type=cfg.match,
                output_file=output_file,
            )
            results[loop_i] = output
    return results, out_file, input_copy_dir, input_folder


def _segment(sentence):
    post = {x for x in sentence.lower().strip().split()}
    return post


def compute_gen_distance(args, input_folder):
    """Compute the distance between each of the generated segments"""
    # Load the generated segments
    input_folder = Path(input_folder)
    assert input_folder.exists(), input_folder
    targets = list(input_folder.glob("q_aug_outs_*.jsonl"))
    assert targets
    LOGGER.info(targets)

    all_self_bleus = []

    for file in targets:
        with jsonl.open(input_folder / file) as f:
            all_lines = list(f)
            rich.print(f"[bold green]{len(all_lines)} in file `{file.name}`")
            self_bleus_per_file = []
            line_lengths = collections.Counter((len(x) for x in all_lines))
            rich.print(f"[bold green]{line_lengths = }")
            for all_entries in all_lines:
                for entry in all_entries:
                    # print(f"{entry = }")
                    all_words = set()
                    sets = []
                    self_bleus = []
                    segmented = [_segment(x) for x in entry]

                    for i, gen_words in enumerate(segmented):
                        sets.append(gen_words)
                        all_words |= gen_words

                    for i, gen_words in enumerate(segmented):
                        for set_ in sets:
                            good_words = gen_words & set_
                            self_bleu = len(good_words) / len(gen_words)
                            self_bleus.append(self_bleu)

                    self_bleu = np.mean(self_bleus)
                    # if self_bleu < 0.25:
                    #     print(f"{self_bleu = :0.0%}")
                    #     print(f"{entry = }")
                    #     print(f"{all_words = }")
                    #     print(f"#" * 80)

                    self_bleus_per_file.append(self_bleu)
                    # print(f"{all_words = }")
                    # print(f"{self_bleu = :0.0%}")
                    # print("#" * 80)
            # return

        all_self_bleus.extend(self_bleus_per_file)
        rich.print(f"[bold blue]{np.mean(self_bleus_per_file) = : 0.2%}")
    
    points = np.linspace(0, 1, 10000)
    all_self_bleus = np.array(all_self_bleus, dtype=np.float64)
    cum = [np.mean(all_self_bleus >= x) for x in points]
    plt.plot(points, cum)
    mean = np.mean(all_self_bleus)
    median = sorted(all_self_bleus)[len(all_self_bleus) // 2]
    plt.plot([mean, mean], [0, 1], color="red")
    plt.plot([median, median], [0, 1], color="purple")

    plt.xlabel("threshold")
    plt.ylabel("Fraction >=")
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    plt.title(f"{args['decoding_conf_query_aug']['temperature'] = }")


def main(input_folder_name):
    format_info = (
        "[%(levelname)s] (%(asctime)s) "
        "{%(name)s->%(funcName)s:%(lineno)d}:\n"
    )

    logging_format = (
        colorama.Fore.CYAN +
        format_info +
        colorama.Style.RESET_ALL +
        "%(message)s"
    )

    logging.basicConfig(
        format=logging_format,
        level=logging.INFO,
        force=True,
    )

    logging.getLogger(
        "common_retriever"
    ).setLevel(logging.WARN)
    logging.getLogger(
        "dense_retriever"
    ).setLevel(logging.WARN)
    logging.getLogger(
        "dpr.data.download_data"
    ).setLevel(logging.WARN)
    logging.getLogger(
        "dpr.data.qa_validation"
    ).setLevel(logging.WARN)

    logging.getLogger(
        "transformers.configuration_utils"
    ).setLevel(logging.WARN)
    logging.getLogger(
        "transformers.tokenization_utils"
    ).setLevel(logging.WARN)
    logging.getLogger(
        "transformers.modeling_utils"
    ).setLevel(logging.WARN)


    ########################################################################
    # Parse the files
    ########################################################################
    ROOT = Path("/home/mila/g/gagnonju/IteratedDecoding/")
    RETRIEVE_AND_DECODE_ROOT = ROOT / "jobs" / "retrieve_and_decode"
    RETRIEVE_AND_DECODE_OUTPUT = RETRIEVE_AND_DECODE_ROOT / "iterated_decoding_output"


    DPR_CONF_PATH = ROOT / "DPR" / "conf"
    cfg = common_retriever.build_cfg(DPR_CONF_PATH)

    with (ROOT / "jobs" / "cache" / "all_passages.pkl").open("rb") as fin:
        passages = pickle.load(fin)

    questions, question_answers = load_questions_and_answers(cfg, "nq_dev")

    results, out_file, input_copy_dir, input_folder = analyse(
        cfg=cfg,
        input_folder_name=input_folder_name, 
        retrieve_and_decode_output=RETRIEVE_AND_DECODE_OUTPUT,
        question_answers=question_answers,
        passages=passages,
    )

    args = utils.load_json(input_copy_dir / "notebook_args.json")
    display_results(results, out_file, args, input_folder)
    rich.print("[bold green]DONE!")

    # compute_gen_distance(args, input_folder)



if __name__ == "__main__":
    fire.Fire(main)