import json
import argparse

from typing import get_args
from itertools import product

from gde.dataset.base import DATASETS
from gde.config import logger, RESULTS_DIR
from gde.model import GRITModels, OpenCLIP, Talk2DiNo
from gde.eval import CZSLMetrics, GroupRobustnessMetrics
from gde.eval.method import (
    run_zero_shot,
    run_decomposition,
    run_primitive_only,
)

MODEL_REGISTRY = {
    "openclip": OpenCLIP,
    "talk2dino": Talk2DiNo,
    "grit": GRITModels,
}


_CZSL_FIELDS = [
    ("pair_acc", "Pair acc"),
    ("attr_acc", "Attr acc"),
    ("obj_acc", "Obj acc"),
    ("seen_acc", "Seen  (b=0)"),
    ("unseen_acc", "Unseen (b=0)"),
    ("hm", "HM    (b=0)"),
    ("seen_only_acc", "S* seen-only"),
    ("unseen_only_acc", "U* unseen-only"),
    ("best_hm", "Best HM"),
    ("best_bias", "Best bias"),
    ("auc", "AUC"),
]

_GR_FIELDS = [
    ("avg_acc", "Avg acc"),
    ("worst_group_acc", "Worst-group acc"),
    ("gap", "Gap"),
]


def _serialise(result) -> dict:
    """Recursively convert CZSLMetrics / GroupRobustnessMetrics to plain dicts."""
    if isinstance(result, CZSLMetrics):
        return result.asdict()
    if isinstance(result, GroupRobustnessMetrics):
        return result.asdict()
    if isinstance(result, dict):
        return {k: _serialise(v) for k, v in result.items()}
    return result


def _fmt(v, field: str) -> str:
    if v is None or (isinstance(v, float) and v != v):  # NaN
        return "   N/A"
    if field == "best_bias":
        return f"{v:8.4f}"
    return f"{v:6.2f}%"


def _print_metrics_block(
    label: str,
    czsl: dict | None,
    gr: dict | None,
    indent: str = "",
) -> None:
    w = 26
    print(f"{indent}{label}")

    if czsl is not None:
        print(f"{indent}  {'CZSL':-<{w}}")
        for key, name in _CZSL_FIELDS:
            val = czsl.get(key)
            print(f"{indent}    {name:<20} {_fmt(val, key)}")

    if gr is not None:
        print(f"{indent}  {'Group Robustness':-<{w}}")
        for key, name in _GR_FIELDS:
            val = gr.get(key)
            print(f"{indent}    {name:<20} {_fmt(val, key)}")


def print_results(results: dict) -> None:
    for method, result in results.items():
        print(f"\n+-- {method}")

        if "error" in result:
            print(f"|   ERROR: {result['error']}")
            continue

        czsl = result.get("czsl")
        gr = result.get("gr")

        if czsl is None and gr is None:
            # primitive / pair / etc. nested case
            if any(isinstance(v, dict) for v in result.values()):
                print_results(result)
                print("+" + "-" * 60)
        else:
            _print_metrics_block(
                "",
                czsl if isinstance(czsl, dict) else None,
                gr if isinstance(gr, dict) else None,
                indent="|  ",
            )

        print("+" + "-" * 60)


def compute_selective(
    model_names,
    architectures,
    pretrained,
    dataset_names,
    apply_normalisation=False,
    apply_dimred=False,
):
    logger.info("Will run evaluation with the following filters:")
    logger.info(f"  Models: {model_names}")
    logger.info(f"  Architectures: {architectures if architectures else 'ALL'}")
    logger.info(f"  Pretraining: {pretrained if pretrained else 'ALL'}")
    logger.info(f"  Datasets: {dataset_names if dataset_names else 'ALL'}")

    for dataset_name in dataset_names:
        run_czsl = dataset_name not in ["waterbirds", "celebA"]
        run_group_robustness = dataset_name in ["waterbirds", "celebA"]
        run_pair_based = dataset_name not in ["waterbirds", "celebA"]
        run_primitive_based = True

        for model_name, model_cls in [(n, MODEL_REGISTRY[n]) for n in model_names]:
            valid_archs = get_args(model_cls.ARCH_OPTIONS)
            valid_pres = get_args(model_cls.PRETRAINING_OPTIONS)
            normalize = apply_normalisation and (model_name == "talk2dino")
            use_dimred = apply_dimred

            archs = (
                [a for a in valid_archs if a in architectures]
                if architectures
                else list(valid_archs)
            )
            pres = (
                [p for p in valid_pres if p in pretrained]
                if pretrained
                else list(valid_pres)
            )

            if not archs:
                logger.warning(
                    f"No valid architectures selected for {model_cls.__name__} "
                    f"on dataset {dataset_name}. Skipping."
                )
                continue

            if not pres:
                logger.warning(
                    f"No valid pretraining options selected for {model_cls.__name__} "
                    f"on dataset {dataset_name}. Skipping."
                )
                continue

            logger.info(
                f"Running {model_cls.__name__} on {dataset_name} "
                f"with architectures={archs} and pretraining={pres}"
            )

            for arch, pre in product(archs, pres):
                try:
                    # Geometry selection
                    if model_name == "grit":
                        if pre in ["meru", "hycoclip"]:
                            geometry = "hyperbolic"
                        else:
                            geometry = "spherical"
                    elif model_name == "openclip":
                        geometry = "spherical"
                    elif model_name == "talk2dino":
                        if normalize:
                            geometry = "spherical"
                        else:
                            geometry = "euclidean"
                    else:
                        geometry = "unresolved"
                        raise ValueError(f"Unrecognized {model_name=}")

                    resolved_model_name = model_cls.get_name(arch, pre)

                    logger.info(
                        f"Evaluating model={resolved_model_name} "
                        f"({geometry=}/{normalize=}/{use_dimred=}) on dataset={dataset_name} "
                    )

                    # Create output directory
                    if not (normalize or use_dimred):
                        save_to_dir = RESULTS_DIR / dataset_name / resolved_model_name
                    else:
                        name_file = f"{resolved_model_name}"
                        if normalize:
                            name_file += "_normalized"
                        if use_dimred:
                            name_file += "_dimred"

                        save_to_dir = RESULTS_DIR / dataset_name / name_file
                    save_to_dir.mkdir(parents=True, exist_ok=True)

                    zero_shot_path = save_to_dir / "zero_shot.json"
                    if not zero_shot_path.exists() and not use_dimred:
                        zs_result = run_zero_shot(
                            dataset_name,
                            resolved_model_name,
                            geometry,
                            run_czsl=run_czsl,
                            run_group_robustness=run_group_robustness,
                            run_pair_based=run_pair_based,
                            run_primitive_based=run_primitive_based,
                            normalize=normalize,
                        )

                        s_zs_result = _serialise(zs_result)

                        print_results(s_zs_result)

                        with open(zero_shot_path, "w") as f:
                            json.dump(s_zs_result, f, indent=4)

                    logger.info(f"Saved zero-shot results to {zero_shot_path}")

                    decomposition_path = save_to_dir / "decomposition.json"
                    if not decomposition_path.exists():
                        compute_robust_text_embeddings = dataset_name in [
                            "waterbirds",
                            "celebA",
                        ]

                        decomp_result = run_decomposition(
                            dataset_name,
                            resolved_model_name,
                            geometry,
                            compute_robust_text_embeddings=compute_robust_text_embeddings,
                            run_czsl=run_czsl,
                            run_group_robustness=run_group_robustness,
                            run_pair_based=run_pair_based,
                            run_primitive_based=run_primitive_based,
                            normalize=normalize,
                            use_whitening=use_dimred,
                        )

                        s_decomp_result = _serialise(decomp_result)

                        print_results(s_decomp_result)

                        with open(decomposition_path, "w") as f:
                            json.dump(s_decomp_result, f, indent=4)

                    logger.info(f"Saved decomposition results to {decomposition_path}")

                    primitive_only_path = save_to_dir / "primitive_only.json"
                    if not primitive_only_path.exists():
                        prim_result = run_primitive_only(
                            dataset_name,
                            resolved_model_name,
                            geometry,
                            compute_robust_text_embeddings=False,
                            run_czsl=run_czsl,
                            run_group_robustness=run_group_robustness,
                            run_pair_based=run_pair_based,
                            run_primitive_based=run_primitive_based,
                            normalize=normalize,
                            use_whitening=use_dimred,
                        )

                        s_prim_result = _serialise(prim_result)

                        print_results(s_prim_result)

                        with open(primitive_only_path, "w") as f:
                            json.dump(s_prim_result, f, indent=4)

                    logger.info(
                        f"Saved primitive only results to {primitive_only_path}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error computing metrics for "
                        f"{model_cls.__name__} ({geometry=}/{normalize=}/{use_dimred=}) "
                        f"with arch={arch}, pretraining={pre} "
                        f"on dataset {dataset_name}: {e}"
                    )
                    raise e


def _build_parser() -> argparse.ArgumentParser:
    all_archs = sorted(
        {arch for cls in MODEL_REGISTRY.values() for arch in get_args(cls.ARCH_OPTIONS)}
    )
    all_pres = sorted(
        {
            pre
            for cls in MODEL_REGISTRY.values()
            for pre in get_args(cls.PRETRAINING_OPTIONS)
        }
    )

    parser = argparse.ArgumentParser(
        description="Compute VLM metrics on composition datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        metavar="MODEL",
        help=f"Model class(es) to run. Choices: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--architecture",
        nargs="+",
        choices=all_archs,
        metavar="ARCH",
        help=f"Architecture(s) to filter per model. Choices: {all_archs}",
    )
    parser.add_argument(
        "--pretraining",
        nargs="+",
        choices=all_pres,
        metavar="PRETRAINING",
        help=f"Pretraining(s) to filter per model. Choices: {all_pres}",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=list(get_args(DATASETS)),
        metavar="DATASET",
        help=f"Dataset(s) to run. Choices: {list(get_args(DATASETS))}",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=f"If to normalize the embeddings when loaded.",
    )
    parser.add_argument(
        "--dimred",
        action="store_true",
        help=f"If to dimred the embeddings.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    compute_selective(
        model_names=args.model or list(MODEL_REGISTRY.keys()),
        architectures=args.architecture,
        pretrained=args.pretraining,
        dataset_names=args.dataset or list(get_args(DATASETS)),
        apply_normalisation=args.normalize,
        apply_dimred=args.dimred,
    )
