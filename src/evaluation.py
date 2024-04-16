import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

from metrics import exIoU, exAcc, joint_iou, joint_accuracy, mean_iou
from config import *


@dataclass
class EvaluationResult:
    iou_activated_per_class: dict[str, exIoU]
    iou_pred_per_class: dict[str, exIoU]
    accuracy: exAcc

    def to_str(self, description: str):
        iou_activated = "".join(f"\t\t {class_name}: {iou.iou:.4f}\n" for class_name, iou in self.iou_activated_per_class.items())
        iou_pred = "".join(f"\t\t {class_name}: {iou.iou:.4f}\n" for class_name, iou in self.iou_pred_per_class.items())
        res_str = (
            f"Evaluation result ({description}):\n"
            f"\tmean IoU (activated): {mean_iou(self.iou_activated_per_class):.4f}\n"
            f"\tmean IoU (prediction): {mean_iou(self.iou_pred_per_class):.4f}\n"
            f"\taccuracy: {self.accuracy.accuracy:.4f}\n"
            f"\tIoU (activated) per class:\n"
            f"{iou_activated}"
            f"\tIoU (prediction) per class:\n"
            f"{iou_pred}\n"
        )
        return res_str


def evaluate_dataset(eval_results: list[EvaluationResult]) -> EvaluationResult:
    total_iou_activated_per_class = dict()
    total_iou_pred_per_class = dict()

    for class_name in squeezed_codes2labels.values():
        total_iou_activated_per_class[class_name] = joint_iou([eval_res.iou_activated_per_class[class_name] for eval_res in eval_results])
        total_iou_pred_per_class[class_name] = joint_iou([eval_res.iou_pred_per_class[class_name] for eval_res in eval_results])

    total_accuracy = joint_accuracy([eval_res.accuracy for eval_res in eval_results])

    return EvaluationResult(
        iou_activated_per_class=total_iou_activated_per_class,
        iou_pred_per_class=total_iou_pred_per_class,
        accuracy=total_accuracy
    )


def plot_single_class_data(data: list[float], data_name: str, exp_path: Path):
    n_epochs = len(data)
    fig = plt.figure(figsize=(12, 6))
    
    x = [x + 1 for x in range(n_epochs)]
    y = [data[i] for i in range(n_epochs)]
    plt.plot(x, y)

    plt.ylabel(f"{data_name}", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    fig.savefig(exp_path / f"{data_name}.png")


def plot_multi_class_data(data: dict[str, list[float]], data_name: str, exp_path: Path):
    n_epochs = len(list(data.values())[0])
    fig = plt.figure(figsize=(12, 6))

    for class_name, values in data.items():
        x = [x + 1 for x in range(n_epochs)]
        y = [values[i] for i in range(n_epochs)]
        plt.plot(x, y, color=labels2colors[class_name])

    plt.ylabel(f"{data_name}", fontsize=20)
    plt.xlabel("epoch", fontsize=20)    
    plt.legend([class_name for class_name in data], loc="center right", fontsize=15)
    fig.savefig(exp_path / f"{data_name}.png")


def show_evaluation_over_epoch(epoch_eval_results: list[EvaluationResult], exp_path: Path) -> None:
    mean_ious_activated = [mean_iou(eval_res.iou_activated_per_class) for eval_res in epoch_eval_results]
    mean_ious_prediction = [mean_iou(eval_res.iou_pred_per_class) for eval_res in epoch_eval_results]
    accuracies = [eval_res.accuracy.accuracy for eval_res in epoch_eval_results]

    plot_single_class_data(mean_ious_activated, "mean_iou_activated", exp_path)
    plot_single_class_data(mean_ious_prediction, "mean_iou_prediction", exp_path)
    plot_single_class_data(accuracies, "accuracy", exp_path)

    ious_activated_per_class = dict()
    ious_pred_per_class = dict()
    for class_name in squeezed_codes2labels.values():
        ious_activated_per_class[class_name] = [eval_res.iou_activated_per_class[class_name].iou for eval_res in epoch_eval_results]
        ious_pred_per_class[class_name] = [eval_res.iou_pred_per_class[class_name].iou for eval_res in epoch_eval_results]

    plot_multi_class_data(ious_activated_per_class, "iou_activated_per_class", exp_path)
    plot_multi_class_data(ious_pred_per_class, "iou_pred_per_class", exp_path)
