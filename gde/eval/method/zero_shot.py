from itertools import product
from typing import Callable

from torch import Tensor

from gde.config import logger
from gde.eval import CZSLEvaluator
from gde.dataset.base import DATASETS
from gde.dataset import CompositionDatasetEmbeddings

from gde.eval.util.similarity import (
    pw_cosine_similarity,
    pw_spherical_similarity,
    pw_lorentzian_similarity,
    GEOMETRIES,
)
from gde.eval.util.gr import gr_from_pair_logits
from gde.whiten import DIM_RED_MODELS
from gde.whiten.base import ManifoldWhiten


def primitive_classification(
    attrs_objs,
    logit_scale: float,
    similarity: Callable[[Tensor, Tensor], Tensor],
    test_im_embeddings,
    prim_attr_embs,
    prim_obj_embs,
) -> tuple[Tensor, list[tuple[str, str]]]:
    # perform independent evaluation of primitives

    attr_scores = logit_scale * similarity(test_im_embeddings, prim_attr_embs)
    obj_scores = logit_scale * similarity(test_im_embeddings, prim_obj_embs)
    combined_scores = attr_scores.unsqueeze(-1) + obj_scores.unsqueeze(  # (B, N, 1)
        -2
    )  # (B, 1, M)  # (B, N, M)

    # flatten to (B, N*M) matching list(product(attrs, objs))
    combined_scores = combined_scores.flatten(1)
    combined_probs = combined_scores.softmax(dim=-1)

    attrs, objs = attrs_objs

    combined_pairs = list(product(attrs, objs))

    return combined_probs, combined_pairs


def run_zero_shot(
    dataset_name: DATASETS,
    model_name: str,
    geometry: GEOMETRIES = "spherical",
    logit_scale: float = min(1.0 / 0.07, 100),
    run_czsl=True,
    run_group_robustness=True,
    run_pair_based=True,
    run_primitive_based=True,
    normalize: bool = False,
    *args,
    **kwargs,
) -> dict:
    emb_ds_train = CompositionDatasetEmbeddings(
        dataset=dataset_name, phase="train", model_name=model_name
    )
    emb_ds_test = CompositionDatasetEmbeddings(
        dataset=dataset_name, phase="test", model_name=model_name
    )

    test_im_embeddings, test_pairs = emb_ds_test.load_all_image_embs(
        normalize=normalize
    )

    (text_prim_attr_embeddings, attrs), (text_prim_obj_embeddings, objs) = (
        emb_ds_train.load_text_primitive_embs(normalize=normalize)
    )

    logger.info(
        f"Text Primitive embeddings shape: \n"
        f"  Attr: {text_prim_attr_embeddings.shape}\n"
        f"  Obj: {text_prim_obj_embeddings.shape}"
    )

    if geometry == "euclidean":
        similarity = pw_cosine_similarity
    elif geometry == "spherical":
        similarity = pw_spherical_similarity
    elif geometry == "hyperbolic":
        similarity = pw_lorentzian_similarity
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")

    all_results = {}

    evaluator = CZSLEvaluator(emb_ds_train.train_pairs)

    if run_pair_based and run_czsl:
        all_results["pair"] = {"open_world": {}, "close_world": {}}
        for open_world in [False, True]:
            key = "open_world" if open_world else "close_world"
            text_embeddings, text_pairs = emb_ds_test.load_all_text_embs(
                open_world, normalize=normalize
            )

            logger.info(
                f"Image embeddings shape: {test_im_embeddings.shape}\n"
                f"Text embeddings shape: {text_embeddings.shape}"
            )

            logits = (
                logit_scale * similarity(test_im_embeddings, text_embeddings)
            ).softmax(dim=-1)

            all_results["pair"][key]["czsl"] = evaluator.evaluate_pair(
                logits, candidate_pairs=text_pairs, test_pairs=test_pairs
            )

    if run_primitive_based:
        all_results["primitive"] = {"text": {}}

        combined_probs, combined_pairs = primitive_classification(
            (attrs, objs),
            logit_scale,
            similarity,
            test_im_embeddings,
            text_prim_attr_embeddings,
            text_prim_obj_embeddings,
        )

        if run_czsl:
            all_results["primitive"]["text"]["czsl"] = evaluator.evaluate_pair(
                combined_probs, combined_pairs, test_pairs
            )

        if run_group_robustness:
            all_results["primitive"]["text"]["gr"] = gr_from_pair_logits(
                combined_probs,
                combined_pairs,
                test_pairs,
                emb_ds_train,
            )

    return all_results
