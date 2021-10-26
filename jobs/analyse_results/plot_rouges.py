#!/usr/bin/env python
# coding: utf-8

import collections
import colored_traceback.auto
import functools
import os
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import scipy
import scipy.ndimage

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
    assert path.exists, f"\nPath doesn't exist:\n{str(path) = }"
    paths = list(path.glob(glob_pattern))
    assert paths, f"\nDidn't match any files:\n{str(path) = }\n{glob_pattern = }"
    
    paths_and_numbers = [
        (path, parse_num(path))
        for path in paths
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

                entries[metric_name][recall_precision_f]["low"].append(low)
                entries[metric_name][recall_precision_f]["mid"].append(mid)
                entries[metric_name][recall_precision_f]["high"].append(high)

    return entries


def exponential_smoothing(points, k):
    so_far = points[0]
    output = [so_far]
    for point in points[1:]:
        so_far = so_far * k + (1 - k) * point
        output.append(so_far)
    return output
    

def conv_lin_smoothing(points, k):
    return np.convolve(points, np.ones((k,)) * 1 / k, mode="valid")
    

def conv_norm_smoothing(points, k):
    return scipy.ndimage.gaussian_filter1d(points, k, mode="nearest")

    
def main(directory, smoothing_param=0, smoothing_type="exponential"):
    if smoothing_type == "exponential":
        smoothing_fn = lambda points: exponential_smoothing(points, smoothing_param)
    elif smoothing_type == "conv-lin" or smoothing_type == "conv":
        smoothing_fn = lambda points: conv_lin_smoothing(points, smoothing_param)
    elif smoothing_type == "conv-norm" or smoothing_type == "conv-gauss":
        smoothing_fn = lambda points: conv_norm_smoothing(points, smoothing_param)
    else:
        raise ValueError(smoothing_type)
    
    by_metric = build_entries(directory, "rouge_*.txt")
    plt.clf()
    for metric_name, by_rpf in by_metric.items():
        fig, axes = plt.subplots(1, 3)
        for ax, (rpf_name, per_confidence_interval) in zip(axes, by_rpf.items()):
            ax.title.set_text(f"{metric_name}-{rpf_name}")
            smoothed_mid = smoothing_fn(per_confidence_interval["mid"])
            argmax = np.argmax(smoothed_mid)
            ax.plot(smoothed_mid, label=f"mid")
            x = np.arange(len(smoothed_mid))
            min_ = np.min(smoothing_fn(per_confidence_interval["low"]))
            ax.fill_between(
                x,
                smoothing_fn(per_confidence_interval["low"]), 
                smoothing_fn(per_confidence_interval["high"]),
                alpha=0.2, 
                edgecolor='#1B2ACC', 
                facecolor='#089FFF',
                linewidth=4, 
                antialiased=True,
            )
            diff_y = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
            diff_x = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
            ax.plot([x[argmax], x[argmax]], [min_, smoothed_mid[argmax]], color="red")
            ax.text(x[argmax] + diff_x * 0.02, min_ - diff_y * 0.015, str(x[argmax]), color="red")
            
#           plt.scatter(np.arange(len(value)), value, s=3)

        fig.show()
        fig.set_size_inches(20, 5)
    
    
        
if __name__ == "__main__":
    fire.Fire(main)



