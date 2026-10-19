from itertools import product

import torch
from torch import Tensor

from gde.config import logger
from gde.eval import CZSLEvaluator
from gde.dataset.base import DATASETS
from gde.dataset import CompositionDatasetEmbeddings

from gde.factorizer import LDE, GDE, HyperbolicGDE
from gde.eval.util.similarity import (
    pw_cosine_similarity,
    pw_spherical_similarity,
    pw_lorentzian_similarity,
    GEOMETRIES,
)
from gde.eval.util.weights import build_weights
from gde.eval.util.gr import gr_from_obj_preds
from gde.whiten import DIM_RED_MODELS
from gde.whiten.base import ManifoldWhiten


def primitive_classification(
    factorizer: LDE | GDE | HyperbolicGDE,
    logit_scale: float,
    test_im_embeddings,
    prim_attr_embs,
    prim_obj_embs,
) -> tuple[Tensor, list[tuple[str, str]]]:
    # perform independent evaluation of primitives
    vector_on_tangent = factorizer.to_tangent_at(test_im_embeddings)

    attr_scores = logit_scale * pw_cosine_similarity(vector_on_tangent, prim_attr_embs)
    obj_scores = logit_scale * pw_cosine_similarity(vector_on_tangent, prim_obj_embs)
    # (B, 1, M)  # (B, N, M)
    combined_scores = attr_scores.unsqueeze(-1) + obj_scores.unsqueeze(-2)

    # flatten to (B, N*M) matching list(product(attrs, objs))
    combined_scores = combined_scores.flatten(1)
    combined_probs = combined_scores.softmax(dim=-1)

    combined_pairs = list(product(factorizer.attrs, factorizer.objs))

    return combined_probs, combined_pairs


def run_decomposition(
    dataset_name: DATASETS,
    model_name: str,
    geometry: GEOMETRIES = "spherical",
    logit_scale: float = min(1.0 / 0.07, 100),
    compute_robust_text_embeddings: bool = False,
    run_czsl=True,
    run_group_robustness=True,
    run_pair_based=True,
    run_primitive_based=True,
    normalize: bool = False,
    use_whitening: bool = False,
    *args,
    **kwargs,
) -> dict:
    emb_ds_train = CompositionDatasetEmbeddings(
        dataset=dataset_name, phase="train", model_name=model_name
    )

    text_embs, text_pairs = emb_ds_train.load_text_embs(
        emb_ds_train.full_pairs, True, normalize=normalize
    )
    train_embs, train_pairs = emb_ds_train.load_all_image_embs(normalize=normalize)

    if use_whitening:
        # geometry = "euclidean"
        dim_red_cls = DIM_RED_MODELS[geometry]
        # text_whitener: ManifoldWhiten = dim_red_cls()
        img_whitener: ManifoldWhiten = dim_red_cls()

        # t_emb, _ = emb_ds_train.load_all_text_embs(True, normalize=normalize)
        # text_whitener.fit(t_emb)
        img_whitener.fit(train_embs)

        # text_embs = text_whitener.project(text_embs)
        train_embs = img_whitener.project(train_embs)

    if geometry == "euclidean":
        factorizer_cls = LDE
    elif geometry == "spherical":
        factorizer_cls = GDE
    elif geometry == "hyperbolic":
        factorizer_cls = HyperbolicGDE
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")

    if geometry == "euclidean":
        similarity = pw_cosine_similarity
    elif geometry == "spherical":
        similarity = pw_spherical_similarity
    elif geometry == "hyperbolic":
        similarity = pw_lorentzian_similarity
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")

    weights_text = build_weights(text_embs, text_pairs, geometry, None, None, False)
    factorizer_text = factorizer_cls(text_embs, text_pairs, weights=weights_text)

    text_prim_attr_embs, text_prim_obj_embs = (
        factorizer_text.attr_IW,
        factorizer_text.obj_IW,
    )

    logger.info(
        f"Text Primitive embeddings shape: \n"
        f"  Attr: {factorizer_text.attr_IW.shape}\n"
        f"  Obj: {factorizer_text.obj_IW.shape}"
    )

    pair_embeds, _ = emb_ds_train.load_text_embs(train_pairs, train=True)
    if use_whitening:
        pair_embeds = img_whitener.project(pair_embeds)

    weights_img = build_weights(
        train_embs,
        train_pairs,
        geometry,
        pair_embeds,
        factorizer_text if not use_whitening else None,
        use_im_text_sim=not use_whitening,
        compute_robust_text_embeddings=compute_robust_text_embeddings
        and not use_whitening,
    )
    factorizer_img = factorizer_cls(train_embs, train_pairs, weights=weights_img)
    img_prim_attr_embs, img_prim_obj_embs = (
        factorizer_img.attr_IW,
        factorizer_img.obj_IW,
    )

    logger.info(
        f"Image Primitive embeddings shape: \n"
        f"  Attr: {factorizer_img.attr_IW.shape}\n"
        f"  Obj: {factorizer_img.obj_IW.shape}"
    )

    emb_ds_test = CompositionDatasetEmbeddings(
        dataset=dataset_name, phase="test", model_name=model_name
    )

    test_im_embeddings, test_pairs = emb_ds_test.load_all_image_embs(
        normalize=normalize
    )
    if use_whitening:
        test_im_embeddings = img_whitener.project(test_im_embeddings)

    all_results = {}

    evaluator = CZSLEvaluator(emb_ds_train.train_pairs)
    if run_pair_based and run_czsl:
        all_results["pair"] = {"open_world": {}, "close_world": {}}

        for open_world in [False, True]:
            key = "open_world" if open_world else "close_world"
            text_embeddings, text_pairs = emb_ds_test.load_all_text_embs(
                open_world, normalize=normalize
            )
            if use_whitening:
                text_embeddings = img_whitener.project(text_embeddings)

            logger.info(
                f"Image embeddings shape: {test_im_embeddings.shape}\n"
                f"Text embeddings shape: {text_embeddings.shape}"
            )

            decomposed_text_iw = factorizer_text.compute_ideal_words_approximation(
                text_pairs
            )
            logits_w_text_iw = (
                logit_scale * similarity(test_im_embeddings, decomposed_text_iw)
            ).softmax(dim=-1)

            decomposed_image_iw = factorizer_img.compute_ideal_words_approximation(
                text_pairs
            )
            logits_w_image_iw = (
                logit_scale * similarity(test_im_embeddings, decomposed_image_iw)
            ).softmax(dim=-1)

            all_results["pair"][key] = {
                "text": {
                    "czsl": evaluator.evaluate_pair(
                        logits_w_text_iw,
                        candidate_pairs=text_pairs,
                        test_pairs=test_pairs,
                    )
                },
                "image": {
                    "czsl": evaluator.evaluate_pair(
                        logits_w_image_iw,
                        candidate_pairs=text_pairs,
                        test_pairs=test_pairs,
                    )
                },
            }

    if run_primitive_based:

        primitive_result = {}

        for modality, fact, attr_emb, obj_emb in [
            ("text", factorizer_text, text_prim_attr_embs, text_prim_obj_embs),
            ("image", factorizer_img, img_prim_attr_embs, img_prim_obj_embs),
        ]:
            metrics = {}

            if run_czsl:
                probs, pairs = primitive_classification(
                    fact,
                    logit_scale,
                    test_im_embeddings,
                    attr_emb,
                    obj_emb,
                )
                metrics["czsl"] = evaluator.evaluate_pair(probs, pairs, test_pairs)

            if run_group_robustness:
                tangents = fact.to_tangent_at(test_im_embeddings)
                obj_idxs = (
                    (logit_scale * pw_cosine_similarity(tangents, fact.obj_IW))
                    .softmax(dim=-1)
                    .argmax(dim=-1)
                )
                pred_objs = [fact.objs[i] for i in obj_idxs]
                metrics["gr"] = gr_from_obj_preds(pred_objs, test_pairs, emb_ds_train)

            if metrics:
                primitive_result[modality] = metrics

        all_results["primitive"] = primitive_result

    return all_results
