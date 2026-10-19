from typing import Literal, Optional, Iterable

import numpy as np

from gde.config import logger

from .cc import CrossCorrelationEstimator, CCAEstimator

from ..dataset import CompositionDatasetEmbeddings
from ..eval.util.weights import build_weights
from ..factorizer import LDE, HyperbolicGDE, GDE

ESTIMATORS = {"cce": CrossCorrelationEstimator, "cca": CCAEstimator}
DEFAULT_ESTIMATORS = ["cce", "cca"]
ESTIMATOR_TYPE = Literal["cce", "cca"]


def _to_numpy(x) -> np.ndarray:
    """Coerce torch tensors (or anything array-like) to a numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def estimate(
    est_cls,
    x,
    y,
    geometry=None,
    normalize=None,
    resolved_model_name=None,
    dataset_name=None,
    name=None,
):
    try:
        mi = float(est_cls().estimate(x, y))
    except Exception as e:
        logger.error(
            f"Error computing metrics for {name=} "
            f"({geometry=}/{normalize=}) "
            f"with model={resolved_model_name} "
            f"on dataset {dataset_name}:\n{e}"
        )
        mi = 0.0
    return mi


def mutual_information(
    dataset_name,
    resolved_model_name,
    geometry,
    normalize=False,
    stats_gde=None,
    estimators: Optional[Iterable[ESTIMATOR_TYPE]] = None,
):
    if estimators is None:
        estimators = [*DEFAULT_ESTIMATORS]

    if stats_gde is None:
        logger.warning(f"Computing statistics from scratch")
        emb_ds_train = CompositionDatasetEmbeddings(
            dataset=dataset_name, phase="train", model_name=resolved_model_name
        )

        if geometry == "euclidean":
            factorizer_cls = LDE
        elif geometry == "spherical":
            factorizer_cls = GDE
        elif geometry == "hyperbolic":
            factorizer_cls = HyperbolicGDE
        else:
            raise ValueError(f"Unsupported geometry: {geometry}")

        train_embs, train_pairs = emb_ds_train.load_all_image_embs(normalize=normalize)
        weights_img = build_weights(
            train_embs,
            train_pairs,
            geometry,
            emb_ds_train,
            None,
            use_im_text_sim=True,
            compute_robust_text_embeddings=False,
        )

        factorizer_img = factorizer_cls(train_embs, train_pairs, weights=weights_img)
        dummy_pairs = [(f"{a}::{o}", "-") for (a, o) in train_pairs]
        factorizer_all = factorizer_cls(train_embs, dummy_pairs, weights=weights_img)

        img = {
            "objs": factorizer_img.objs,
            "attrs": factorizer_img.attrs,
            "objs_iw": factorizer_img.obj_IW,
            "attrs_iw": factorizer_img.attr_IW,
            "pairs": [lab.split("::") for lab in factorizer_all.attrs],
            "pairs_iw": factorizer_all.attr_IW,
        }
        origin = factorizer_all.obj_IW

        stats_gde = {"image": img, "origin": origin}
    else:
        logger.success(f"Using cached statistics!")
        img = stats_gde["image"]

    objects, attributes, obj_attr_pairs = img["objs"], img["attrs"], img["pairs"]
    object_ideal_words = _to_numpy(img["objs_iw"])
    attribute_ideal_words = _to_numpy(img["attrs_iw"])
    pair_ideal_words = _to_numpy(img["pairs_iw"])

    object_index = {_name: i for i, _name in enumerate(objects)}
    attribute_index = {_name: i for i, _name in enumerate(attributes)}

    x_obj_attr = np.stack(
        [object_ideal_words[object_index[o]] for a, o in obj_attr_pairs]
    )
    y_obj_attr = np.stack(
        [attribute_ideal_words[attribute_index[a]] for a, o in obj_attr_pairs]
    )

    x_obj_pair = np.stack(
        [object_ideal_words[object_index[o]] for a, o in obj_attr_pairs]
    )
    y_obj_pair = pair_ideal_words

    x_attr_pair = np.stack(
        [attribute_ideal_words[attribute_index[a]] for a, o in obj_attr_pairs]
    )
    y_attr_pair = pair_ideal_words

    results = {}
    for est_key in estimators:
        est_cls = ESTIMATORS[est_key]
        try:
            oa_mi = estimate(
                est_cls,
                x_obj_attr,
                y_obj_attr,
                geometry,
                normalize,
                resolved_model_name,
                dataset_name,
                "obj-attr",
            )
            op_mi = estimate(
                est_cls,
                x_obj_pair,
                y_obj_pair,
                geometry,
                normalize,
                resolved_model_name,
                dataset_name,
                "obj-pair",
            )
            ap_mi = estimate(
                est_cls,
                x_attr_pair,
                y_attr_pair,
                geometry,
                normalize,
                resolved_model_name,
                dataset_name,
                "attr-pair",
            )
        except Exception as e:
            logger.error(
                f"Error computing metrics for "
                f"({geometry=}/{normalize=}) "
                f"with model={resolved_model_name} "
                f"on dataset {dataset_name}: {e}"
            )
            oa_mi = op_mi = ap_mi = 0.0
        results[est_key] = {
            "object_attribute": oa_mi,
            "object_pair": op_mi,
            "attribute_pair": ap_mi,
        }
    return stats_gde, results
