"""CZSL evaluator: pair/attr/obj accuracy, seen/unseen split, HM, AUC."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


def _harmonic_mean(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if a + b > 0 else 0.0


@dataclass
class CZSLMetrics:
    """All CZSL metrics from a single evaluation run.

    Fields
    ------
    pair_acc:         % images where predicted (attr, obj) == GT pair.
    attr_acc:         % images where attr of predicted pair == GT attr.
    obj_acc:          % images where obj of predicted pair == GT obj.
    seen_acc:         pair_acc restricted to seen-pair test images (bias=0).
    unseen_acc:       pair_acc restricted to unseen-pair test images (bias=0).
    hm:               harmonic mean of seen_acc and unseen_acc (bias=0).
    seen_only_acc:    accuracy on seen images when unseen candidates are suppressed.
    unseen_only_acc:  accuracy on unseen images when seen candidates are suppressed.
    best_hm:          best HM across the bias sweep (= hm when sweep is skipped).
    best_bias:        unseen-side additive bias achieving best_hm.
    auc:              area under the seen-vs-unseen accuracy curve (×100).

    Note
    ----
    seen_only_acc and unseen_only_acc are NaN when called from
    evaluate_from_predictions(), since no pair-level logits are available.
    """

    pair_acc: float
    attr_acc: float
    obj_acc: float
    seen_acc: float
    unseen_acc: float
    hm: float
    seen_only_acc: float
    unseen_only_acc: float
    best_hm: float
    best_bias: float
    auc: float

    def asdict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        def _fmt(v: float) -> str:
            return f"{v:6.2f}%" if not (v != v) else "   N/A "  # nan check

        return "\n".join(
            [
                f"  pair_acc:         {_fmt(self.pair_acc)}",
                f"  attr_acc:         {_fmt(self.attr_acc)}",
                f"  obj_acc:          {_fmt(self.obj_acc)}",
                f"  seen_acc  (b=0):  {_fmt(self.seen_acc)}",
                f"  unseen_acc (b=0): {_fmt(self.unseen_acc)}",
                f"  hm        (b=0):  {_fmt(self.hm)}",
                f"  seen_only_acc:    {_fmt(self.seen_only_acc)}",
                f"  unseen_only_acc:  {_fmt(self.unseen_only_acc)}",
                f"  best_hm:          {_fmt(self.best_hm)}",
                f"  best_bias:        {self.best_bias:8.4f}",
                f"  auc:              {self.auc:6.2f}",
            ]
        )


class CZSLEvaluator:
    """Evaluates CZSL metrics from pre-computed pair logits.

    Construction::

        evaluator = CZSLEvaluator(train_pairs)
        # or from a CompositionDatasetEmbeddings:
        evaluator = CZSLEvaluator.from_dataset(train_dataset)

    Evaluation::

        # Full evaluation with bias sweep (pair-level logits available):
        metrics = evaluator.evaluate(logits, candidate_pairs, test_pairs)

        # Direction-only: predict attr and obj independently (no bias sweep):
        metrics = evaluator.evaluate_from_predictions(attr_preds, obj_preds, test_pairs)

    Parameters
    ----------
    train_pairs:
        Seen (training) (attr, obj) pairs, used to split test images into
        seen / unseen groups.
    """

    def __init__(self, train_pairs: list[tuple[str, str]]):
        self.seen_pairs: frozenset[tuple[str, str]] = frozenset(
            tuple(p) for p in train_pairs
        )

    @classmethod
    def from_dataset(cls, dataset) -> CZSLEvaluator:
        """Construct from any object with a .train_pairs attribute."""
        return cls(dataset.train_pairs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_pair(
        self,
        logits: torch.Tensor,
        candidate_pairs: list[tuple[str, str]],
        test_pairs: list[tuple[str, str]],
        n_bias_steps: int = 20,
    ) -> CZSLMetrics:
        """Evaluate all CZSL metrics from pair-level logits.

        Parameters
        ----------
        logits:
            Float tensor of shape [N_test, N_candidates]. Higher values indicate
            greater similarity. Logit scaling is the caller's responsibility.
        candidate_pairs:
            (attr, obj) pair for each logit column — length N_candidates.
        test_pairs:
            Ground-truth (attr, obj) for each test image — length N_test.
        n_bias_steps:
            Approximate number of bias values sampled for the seen/unseen sweep.
        """
        candidate_pairs = [tuple(p) for p in candidate_pairs]
        test_pairs = [tuple(p) for p in test_pairs]

        # ---- index structures ----
        pair2col = {p: i for i, p in enumerate(candidate_pairs)}
        seen_mask = torch.BoolTensor([p in self.seen_pairs for p in candidate_pairs])

        attrs = {a for a, _ in candidate_pairs}
        objs = {o for _, o in candidate_pairs}
        attr2idx = {a: i for i, a in enumerate(attrs)}
        obj2idx = {o: i for i, o in enumerate(objs)}
        col2ao = torch.LongTensor(
            [(attr2idx[a], obj2idx[o]) for a, o in candidate_pairs]
        )

        pair_true = torch.LongTensor([pair2col[p] for p in test_pairs])
        attr_true = col2ao[pair_true, 0]
        obj_true = col2ao[pair_true, 1]

        is_seen = torch.BoolTensor([p in self.seen_pairs for p in test_pairs])
        seen_ids = is_seen.nonzero(as_tuple=True)[0]
        unseen_ids = (~is_seen).nonzero(as_tuple=True)[0]

        # ---- bias=0 predictions ----
        pair_preds = logits.argmax(dim=1)
        attr_preds = col2ao[pair_preds, 0]
        obj_preds = col2ao[pair_preds, 1]

        pair_acc = (pair_preds == pair_true).float().mean().item() * 100
        attr_acc = (attr_preds == attr_true).float().mean().item() * 100
        obj_acc = (obj_preds == obj_true).float().mean().item() * 100
        seen_acc = _subset_acc(pair_preds, pair_true, seen_ids)
        unseen_acc = _subset_acc(pair_preds, pair_true, unseen_ids)
        hm = _harmonic_mean(seen_acc, unseen_acc)

        # ---- endpoints ----
        s_logits = logits.clone()
        s_logits[:, ~seen_mask] -= 1e3
        seen_only_acc = _subset_acc(s_logits.argmax(dim=1), pair_true, seen_ids)

        u_logits = logits.clone()
        u_logits[:, seen_mask] -= 1e3
        unseen_only_acc = _subset_acc(u_logits.argmax(dim=1), pair_true, unseen_ids)

        # ---- bias sweep ----
        if len(unseen_ids) == 0 or not seen_mask.any():
            best_hm, best_bias, auc = hm, 0.0, 0.0
        else:
            best_hm, best_bias, auc = self._bias_sweep(
                logits, pair_true, seen_mask, seen_ids, unseen_ids, n_bias_steps
            )

        return CZSLMetrics(
            pair_acc=pair_acc,
            attr_acc=attr_acc,
            obj_acc=obj_acc,
            seen_acc=seen_acc,
            unseen_acc=unseen_acc,
            hm=hm,
            seen_only_acc=seen_only_acc,
            unseen_only_acc=unseen_only_acc,
            best_hm=best_hm,
            best_bias=best_bias,
            auc=auc,
        )

    def evaluate_independent(
        self,
        attr_preds: list[str],
        obj_preds: list[str],
        test_pairs: list[tuple[str, str]],
    ) -> CZSLMetrics:
        """Evaluate from independent attr and obj predictions.

        Use this for direction-only methods that independently classify attrs and
        objs without constructing pair-level logits.  Pair accuracy is the joint
        event (attr correct AND obj correct).  Bias-sweep fields (seen_only_acc,
        unseen_only_acc, best_bias, auc) are not applicable and are set to NaN /
        0.0; best_hm equals hm at bias=0.

        Parameters
        ----------
        attr_preds:  Predicted attribute name per test image.
        obj_preds:   Predicted object name per test image.
        test_pairs:  Ground-truth (attr, obj) per test image.
        """
        test_pairs = [tuple(p) for p in test_pairs]
        n = len(test_pairs)
        true_attrs = [a for a, _ in test_pairs]
        true_objs = [o for _, o in test_pairs]

        attr_ok = [p == t for p, t in zip(attr_preds, true_attrs)]
        obj_ok = [p == t for p, t in zip(obj_preds, true_objs)]
        pair_ok = [a and o for a, o in zip(attr_ok, obj_ok)]

        attr_acc = sum(attr_ok) / n * 100
        obj_acc = sum(obj_ok) / n * 100
        pair_acc = sum(pair_ok) / n * 100

        seen_ids = [i for i, p in enumerate(test_pairs) if p in self.seen_pairs]
        unseen_ids = [i for i, p in enumerate(test_pairs) if p not in self.seen_pairs]

        seen_acc = (
            sum(pair_ok[i] for i in seen_ids) / len(seen_ids) * 100 if seen_ids else 0.0
        )
        unseen_acc = (
            sum(pair_ok[i] for i in unseen_ids) / len(unseen_ids) * 100
            if unseen_ids
            else 0.0
        )
        hm = _harmonic_mean(seen_acc, unseen_acc)

        return CZSLMetrics(
            pair_acc=pair_acc,
            attr_acc=attr_acc,
            obj_acc=obj_acc,
            seen_acc=seen_acc,
            unseen_acc=unseen_acc,
            hm=hm,
            seen_only_acc=float("nan"),
            unseen_only_acc=float("nan"),
            best_hm=hm,
            best_bias=0.0,
            auc=0.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bias_sweep(
        logits: torch.Tensor,
        pair_true: torch.Tensor,
        seen_mask: torch.Tensor,
        seen_ids: torch.Tensor,
        unseen_ids: torch.Tensor,
        n_steps: int,
    ) -> tuple[float, float, float]:
        """Return (best_hm %, best_bias, auc %).

        Bias values are sampled from the distribution of
        (max_seen_score - correct_unseen_score) over unseen test images,
        following the standard CZSL bias-sweep protocol.
        """
        correct_scores = logits[torch.arange(len(logits)), pair_true][unseen_ids]
        max_seen_scores = logits[unseen_ids][:, seen_mask].max(dim=1)[0]
        score_diff = max_seen_scores - correct_scores

        sorted_diffs, _ = torch.sort(score_diff)
        step = max(len(sorted_diffs) // n_steps, 1)
        bias_candidates = set(sorted_diffs[::step].tolist())
        bias_candidates.add(0.0)
        bias_list = sorted(bias_candidates)

        seen_accs, unseen_accs = [], []
        for bias in bias_list:
            biased = logits.clone()
            biased[:, ~seen_mask] += bias
            preds = biased.argmax(dim=1)
            correct = preds == pair_true
            seen_accs.append(
                correct[seen_ids].float().mean().item() if len(seen_ids) > 0 else 0.0
            )
            unseen_accs.append(correct[unseen_ids].float().mean().item())

        s = np.array(seen_accs)
        u = np.array(unseen_accs)

        order = np.argsort(u)
        auc = float(np.trapz(s[order], u[order]) * 100)

        hms = 2 * s * u / (s + u + 1e-8)
        best_idx = int(hms.argmax())
        best_hm = float(hms[best_idx] * 100)
        best_bias = float(bias_list[best_idx])

        return best_hm, best_bias, auc


def _subset_acc(preds: torch.Tensor, targets: torch.Tensor, ids: torch.Tensor) -> float:
    if len(ids) == 0:
        return 0.0
    return (preds[ids] == targets[ids]).float().mean().item() * 100
