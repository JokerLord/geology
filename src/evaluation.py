from dataclasses import dataclass

from metrics import exIoU, exAcc


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


def evaluate_dataset(file: object, eval_results: list[EvaluationResult], description: str) -> None:
    pass

    # avg_iou_activated_per_class = dict()
    # avg_iou_pred_per_class = dict()
    # for code in range(len(present_class_codes)):
    #     class_name = squeezed_codes2labels[code]

    #     ious_activated = [x["iou_activated_per_class"][class_name] for x in val_outputs]
    #     avg_iou_activated_per_class[class_name] = sum(ious_activated) / len(ious_activated)

    #     ious_pred = [x["iou_pred_per_class"][class_name] for x in val_outputs]
    #     avg_iou_pred_per_class[class_name] = sum(ious_pred) / len(ious_pred)