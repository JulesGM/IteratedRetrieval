#!/usr/bin/env python
# coding: utf-8

import collections
import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import sys
import time

from pathlib import Path

import fire
import tqdm


@functools.lru_cache()
def parse_num(path):
    return re.match(
        r"rouge_([0-9]+)\.txt", 
        path.name,
    ).group(1)


def build_entries(path, glob_pattern):
    path = Path(path)
    paths_and_numbers = [
        (path, parse_num(path))
        for path in path.glob(glob_pattern)
    ]
    paths_and_numbers.sort(key=lambda x: int(x[1]))
    
    entries = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: dict(low=[], mid=[], high=[])
        )
    )
    
    start = time.monotonic()
    for path, number in tqdm.tqdm(paths_and_numbers, desc="reading from files."):
        with open(path) as f:
            for line in f:
                fields = line.strip().split(",")
                if "rouge" not in fields[0]:
                    continue
                metric_name, recall_precision_f = fields[0].split("-", 1)
                
                low = float(fields[1])
                mid = float(fields[2])
                high = float(fields[3])
                
#                 print(f"{metric_name}-{recall_precision_f}: {low} {mid} {high}")
                entries[metric_name][recall_precision_f]["low"].append(low)
                entries[metric_name][recall_precision_f]["mid"].append(mid)
                entries[metric_name][recall_precision_f]["high"].append(high)
            
    return entries

def main(directory, ):
    by_metric = build_entries(directory, "rouge_*.txt")
    plt.clf()
    
    for metric_name, by_rpf in by_metric.items():
        fig, axes = plt.subplots(1, 3)
        for ax, (rpf_name, per_confidence_interval) in zip(axes, by_rpf.items()):
            ax.title.set_text(f"{metric_name}-{rpf_name}")
            ax.plot(per_confidence_interval["mid"], label=f"mid")
            x = np.arange(len(per_confidence_interval["low"]))
            ax.fill_between(
                x, 
                per_confidence_interval["low"], 
                per_confidence_interval["high"],
                alpha=0.2, 
                edgecolor='#1B2ACC', 
                facecolor='#089FFF',
                linewidth=4, 
                antialiased=True,
            )

#                 plt.scatter(np.arange(len(value)), value, s=3)
                
        fig.show()
        fig.set_size_inches(20, 5)

if __name__ == "__main__"    :
    fire.Fire(main)



