import wandb
import csv
import os
from typing import Dict
from collections import deque
from datetime import datetime
import argparse


class RunningAverage:
    def __init__(self, size):
        self.size = size
        #滚动平均的窗口大小，即用于计算平均值的最近几个值的数量。size 的值会传递到类的 size 属性
        self.q = deque()
        #双端队列（deque），用于存储最近的 size 个值。deque 允许在两端高效地进行元素的添加和删除
        self.sum = 0
        #跟踪当前队列中所有值的总和

    def add(self, val):
        self.q.append(val)#将新的值 val 加入队列 q 的尾部
        self.sum += val#将值 val 加到当前队列中所有值的总和 sum 上
        if len(self.q) > self.size:
            self.sum -= self.q.popleft()# 如果队列的长度超过了预定的 size，则移除队列中的最旧值（即队列的头部元素）

    def mean(self):
        # Avoid divide by 0
        return self.sum / max(len(self.q), 1)


def timestamp():
    return datetime.now().strftime("%B %d, %H:%M:%S")


def wandb_init(config, group_keys, **kwargs) -> None:
    wandb.init(
        project=config["project_name"],
        group="_".join(
            [f"{key}={val}" for key, val in config.items() if key in group_keys]
        ),
        config=config,
        **kwargs,
    )


class CSVLogger:
    """Logger to write results to a CSV. The log function matches that of Weights and Biases.

    Args:
        path: path for the csv results file
    """

    def __init__(self, path: str, args: argparse.Namespace):
        self.results_path = path + "_results.csv"
        self.losses_path = path + "_losses.csv"
        #通过拼接传入的路径和后缀，生成保存结果和损失数据的文件路径。
        self.envs = args.envs
        # If we have a checkpoint, we don't want to overwrite
        if not os.path.exists(self.results_path):
            head_row = ["Hours", "Step"]
            for env in self.envs:
                head_row += [
                    f"{env}/SuccessRate",
                    f"{env}/EpisodeLength",
                    f"{env}/Return",
                ]
            with open(self.results_path, "w") as file:
                writer = csv.writer(file)
                writer.writerow(head_row)
        if not os.path.exists(self.losses_path):
            with open(self.losses_path, "w") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Hours",
                        "Step",
                        "TD Error",
                        "Grad Norm",
                        "Max Q Value",
                        "Mean Q Value",
                        "Min Q Value",
                        "Max Target Value",
                        "Mean Target Value",
                        "Min Target Value",
                    ]
                )

    def log(self, results: Dict[str, str], step: int):
        results_row = [results["losses/hours"], step]
        for env in self.envs:
            results_row += [
                results[f"{env}/SuccessRate"],
                results[f"{env}/EpisodeLength"],
                results[f"{env}/Return"],
            ]
        with open(self.results_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(results_row)
        with open(self.losses_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    results["losses/hours"],
                    step,
                    results["losses/TD_Error"],
                    results["losses/Grad_Norm"],
                    results["losses/Max_Q_Value"],
                    results["losses/Mean_Q_Value"],
                    results["losses/Min_Q_Value"],
                    results["losses/Max_Target_Value"],
                    results["losses/Mean_Target_Value"],
                    results["losses/Min_Target_Value"],
                ]
            )


def get_logger(
    policy_path: str, args: argparse.Namespace, wandb_kwargs: Dict[str, str]
):
    if args.disable_wandb:
        logger = CSVLogger(policy_path, args)
    else:
        wandb_init(
            vars(args),
            [
                "model",
                "obs_embed",
                "a_embed",
                "in_embed",
                "context",
                "layers",
                "bag_size",
                "gate",
                "identity",
                "history",
                "pos",
            ],
            **wandb_kwargs,
        )
        logger = wandb
    return logger
