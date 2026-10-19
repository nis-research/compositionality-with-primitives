"""Group robustness evaluator: worst-group accuracy over (class, spurious-attr) subgroups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GroupRobustnessMetrics:
    """Metrics for worst-group / subgroup robustness evaluation.

    Subgroups are defined by the Cartesian product of class labels and spurious
    attribute (group) labels: subgroup (y, s) = {images with class y, group s}.

    Fields
    ------
    avg_acc:          Overall average accuracy across all images.
    worst_group_acc:  Minimum per-subgroup accuracy (worst-case robustness).
    gap:              avg_acc − worst_group_acc.
    group_accs:       Per-subgroup accuracy array, shape [n_classes, n_groups].
                      Entry [i, j] is the accuracy for class i, group j.
                      Zero for empty subgroups.
    """

    avg_acc: float
    worst_group_acc: float
    gap: float
    group_accs: np.ndarray  # [n_classes, n_groups]

    def asdict(self) -> dict:
        return {
            "avg_acc": self.avg_acc,
            "worst_group_acc": self.worst_group_acc,
            "gap": self.gap,
            "group_accs": self.group_accs.tolist(),
        }

    def __str__(self) -> str:
        lines = [
            f"  avg_acc:          {self.avg_acc:6.3f}%",
            f"  worst_group_acc:  {self.worst_group_acc:6.3f}%",
            f"  gap:              {self.gap:6.3f}%",
        ]
        n_cls, n_grp = self.group_accs.shape
        for ci in range(n_cls):
            for gi in range(n_grp):
                lines.append(
                    f"    class={ci}, group={gi}: {self.group_accs[ci, gi]:6.3f}%"
                )
        return "\n".join(lines)


class GroupRobustnessEvaluator:
    """Evaluates worst-group accuracy for two-factor (class × group) subgroups.

    Usage::

        evaluator = GroupRobustnessEvaluator()
        metrics = evaluator.evaluate(preds, targets_class, targets_group)
        print(metrics)
    """

    def evaluate(
        self,
        preds: np.ndarray,
        targets_class: np.ndarray,
        targets_group: np.ndarray,
    ) -> GroupRobustnessMetrics:
        """Evaluate group robustness.

        Parameters
        ----------
        preds:
            Predicted class indices, shape [N].
        targets_class:
            True class labels, shape [N].
        targets_group:
            Spurious attribute / subgroup labels, shape [N].

        Returns
        -------
        GroupRobustnessMetrics
        """
        preds = np.asarray(preds, dtype=int)
        targets_class = np.asarray(targets_class, dtype=int)
        targets_group = np.asarray(targets_group, dtype=int)

        classes = np.unique(targets_class)
        groups = np.unique(targets_group)

        class2idx = {c: i for i, c in enumerate(classes)}
        group2idx = {g: i for i, g in enumerate(groups)}

        n_cls = len(classes)
        n_grp = len(groups)

        correct_by = np.zeros((n_cls, n_grp))
        total_by = np.zeros((n_cls, n_grp))

        correct = preds == targets_class
        for i in range(len(preds)):
            ci = class2idx[int(targets_class[i])]
            gi = group2idx[int(targets_group[i])]
            correct_by[ci, gi] += correct[i]
            total_by[ci, gi] += 1

        group_accs = np.zeros((n_cls, n_grp))
        worst_acc = float("inf")
        total_correct = 0.0
        total_n = 0.0

        for ci in range(n_cls):
            for gi in range(n_grp):
                n = total_by[ci, gi]
                if n == 0:
                    continue
                acc = correct_by[ci, gi] / n * 100
                group_accs[ci, gi] = acc
                total_correct += correct_by[ci, gi]
                total_n += n
                if acc < worst_acc:
                    worst_acc = acc

        avg_acc = total_correct / total_n * 100 if total_n > 0 else 0.0
        worst_acc = worst_acc if worst_acc != float("inf") else 0.0

        return GroupRobustnessMetrics(
            avg_acc=avg_acc,
            worst_group_acc=worst_acc,
            gap=avg_acc - worst_acc,
            group_accs=group_accs,
        )
