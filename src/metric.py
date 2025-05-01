from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        # One-vs-rest confusion matrix components
        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: shape (B, C) or (B,)
        targets: shape (B,)
        """
        if preds.ndim == 2:
            preds = preds.argmax(dim=1)

        for cls in range(self.num_classes):
            pred_is_cls = preds == cls
            target_is_cls = targets == cls

            tp = (pred_is_cls & target_is_cls).sum()
            fp = (pred_is_cls & ~target_is_cls).sum()
            fn = (~pred_is_cls & target_is_cls).sum()

            self.tp[cls] += tp
            self.fp[cls] += fp
            self.fn[cls] += fn

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return f1  # tensor of shape (num_classes,)

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        if preds.ndim == 2:
            preds = preds.argmax(dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape} vs target {target.shape}")
        
        # [TODO] Count the number of correct prediction
        correct = (preds == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
