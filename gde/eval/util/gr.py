import torch
import numpy as np

from gde.dataset import CompositionDatasetEmbeddings
from gde.eval import GroupRobustnessMetrics, GroupRobustnessEvaluator


def gr_from_pair_logits(
    logits: torch.Tensor,
    candidate_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]],
    train_dataset: CompositionDatasetEmbeddings,
) -> GroupRobustnessMetrics:
    pair_preds = logits.argmax(dim=1).tolist()
    pred_objs = [candidate_pairs[i][1] for i in pair_preds]
    return gr_from_obj_preds(pred_objs, test_pairs, train_dataset)


def gr_from_obj_preds(
    obj_preds: list[str],
    test_pairs: list[tuple],
    train_dataset: CompositionDatasetEmbeddings,
) -> GroupRobustnessMetrics:
    obj2idx = {o: i for i, o in enumerate(train_dataset.objs)}
    attr2idx = {a: i for i, a in enumerate(train_dataset.attrs)}

    preds = np.array([obj2idx[o] for o in obj_preds])
    targets_class = np.array([obj2idx[o] for _, o in test_pairs])
    targets_group = np.array([attr2idx[a] for a, _ in test_pairs])

    return GroupRobustnessEvaluator().evaluate(preds, targets_class, targets_group)
