from gde.dataset import CompositionDatasetEmbeddings
from gde.eval.util.weights import build_weights
from gde.factorizer import (
    PrimitiveEuclidean,
    PrimitiveSphere,
    PrimitiveHyperbolic,
)
from gde.whiten import DIM_RED_MODELS
from gde.whiten.base import ManifoldWhiten


def compute_statistics(
    dataset_name, resolved_model_name, geometry, normalize=False, use_whitening=False
):
    emb_ds_train = CompositionDatasetEmbeddings(
        dataset=dataset_name, phase="train", model_name=resolved_model_name
    )

    if geometry == "euclidean":
        factorizer_cls = PrimitiveEuclidean
    elif geometry == "spherical":
        factorizer_cls = PrimitiveSphere
    elif geometry == "hyperbolic":
        factorizer_cls = PrimitiveHyperbolic
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")

    train_embs, train_pairs = emb_ds_train.load_all_image_embs(normalize=normalize)

    if use_whitening:
        dim_red_cls = DIM_RED_MODELS[geometry]
        img_whitener: ManifoldWhiten = dim_red_cls()
        img_whitener.fit(train_embs)
        train_embs = img_whitener.project(train_embs)

    pair_embeds, _ = emb_ds_train.load_text_embs(
        train_pairs, train=True, normalize=normalize
    )
    weights_img = build_weights(
        train_embs,
        train_pairs,
        geometry,
        pair_embeds,
        None,
        use_im_text_sim=not use_whitening,
        compute_robust_text_embeddings=False,
    )
    factorizer_img = factorizer_cls(train_embs, train_pairs, weights=weights_img)

    dummy_pairs = [(f"{a}::{o}", "-") for (a, o) in train_pairs]
    factorizer_all = factorizer_cls(train_embs, dummy_pairs, weights=weights_img)

    return {
        "image": {
            "objs": factorizer_img.objs,
            "attrs": factorizer_img.attrs,
            "objs_iw": factorizer_img.obj_IW,
            "attrs_iw": factorizer_img.attr_IW,
            "pairs": [lab.split("::") for lab in factorizer_all.attrs],
            "pairs_iw": factorizer_all.attr_IW,
        },
        "origin": factorizer_all.obj_IW,
    }
