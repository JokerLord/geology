from dataclasses import dataclass

from metrics import exIoU, exAcc, joint_iou, joint_accuracy
from config import squeezed_codes2labels


@dataclass
class EvaluationResult:
    iou_activated_per_class: dict[str, exIoU]
    iou_pred_per_class: dict[str, exIoU]
    accuracy: exAcc


    def _mean_iou(self, iou_per_class: dict[str, exIoU], weights=None) -> float:
        """
        Calculates average IoU metric over all classes

        Arguments:
            iou_per_class (dict[str, exIoU]): Dictionary of extended IoU metrics for each class
            weights (list, Optional): List of weights for each class. Default: None
        Returns:
            mean_iou (float): Weighted mean IoU metric over classes
        """

        if weights is None:
            return sum(iou.iou for iou in iou_per_class.values()) / len(iou_per_class)
        else:
            """ Not implemented yet """
            pass

    def to_str(self, description: str):
        iou_activated = "".join(f"\t\t {class_name}: {iou.iou:.4f}\n" for class_name, iou in self.iou_activated_per_class.items())
        iou_pred = "".join(f"\t\t {class_name}: {iou.iou:.4f}\n" for class_name, iou in self.iou_pred_per_class.items())
        res_str = (
            f"Evaluation result ({description}):\n"
            f"\tmean IoU (activated): {self._mean_iou(self.iou_activated_per_class):.4f}\n"
            f"\tmean IoU (prediction): {self._mean_iou(self.iou_pred_per_class):.4f}\n"
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
