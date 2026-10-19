import json
import argparse

from typing import get_args
from itertools import product

import torch

from gde.dataset.base import DATASETS
from gde.config import logger, RESULTS_DIR
from gde.model import GRITModels, OpenCLIP, Talk2DiNo
from gde.statistics import (
    compute_statistics,
    geometric_variance,
    mutual_information,
    radius_stratum,
)

MODEL_REGISTRY = {
    "openclip": OpenCLIP,
    "talk2dino": Talk2DiNo,
    "grit": GRITModels,
}


def compute_selective(
    model_names,
    architectures,
    pretrained,
    dataset_names,
    apply_normalisation=False,
    use_whitening=False,
):
    logger.info("Will run evaluation with the following filters:")
    logger.info(f"  Models: {model_names}")
    logger.info(f"  Architectures: {architectures if architectures else 'ALL'}")
    logger.info(f"  Pretraining: {pretrained if pretrained else 'ALL'}")
    logger.info(f"  Datasets: {dataset_names if dataset_names else 'ALL'}")

    for dataset_name in dataset_names:
        for model_name, model_cls in [(n, MODEL_REGISTRY[n]) for n in model_names]:
            valid_archs = get_args(model_cls.ARCH_OPTIONS)
            valid_pres = get_args(model_cls.PRETRAINING_OPTIONS)
            normalize = apply_normalisation and (model_name == "talk2dino")

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
                        f"({geometry=}/{normalize=}) on dataset={dataset_name}"
                    )

                    if not normalize and not use_whitening:
                        save_to_dir = RESULTS_DIR / dataset_name / resolved_model_name
                    else:
                        name_file = f"{resolved_model_name}"
                        if normalize:
                            name_file += "_normalized"
                        if use_whitening:
                            name_file += "_dimred"

                        save_to_dir = RESULTS_DIR / dataset_name / name_file

                    save_to_dir = save_to_dir / "statistics"
                    save_to_dir.mkdir(parents=True, exist_ok=True)

                    statistics_path = save_to_dir / "statistics.pt"
                    if not statistics_path.exists():
                        stats = compute_statistics(
                            dataset_name,
                            resolved_model_name,
                            geometry,
                            normalize=normalize,
                            use_whitening=use_whitening,
                        )
                        torch.save(
                            stats,
                            statistics_path,
                        )
                    else:
                        stats = torch.load(statistics_path)

                    logger.info(f"Saved statistics to {statistics_path}")

                    geometric_var_path = save_to_dir / "geometric_variance.json"
                    if not geometric_var_path.exists():
                        var = geometric_variance(stats, geometry)
                        with open(geometric_var_path, "w") as f:
                            json.dump(var, f)

                    logger.info(f"Saved geometric variance to {geometric_var_path}")

                    mutual_information_path = save_to_dir / "mutual_information.json"
                    statistics_gde_path = save_to_dir / "statistics_gde.pt"

                    if not mutual_information_path.exists():
                        if statistics_gde_path.exists():
                            stats_gde = torch.load(statistics_gde_path)
                        else:
                            stats_gde = None

                        stats_gde, mi = mutual_information(
                            dataset_name,
                            resolved_model_name,
                            geometry,
                            normalize,
                            stats_gde=stats_gde,
                        )
                        with open(mutual_information_path, "w") as f:
                            json.dump(mi, f)

                        if not statistics_gde_path.exists():
                            torch.save(
                                stats_gde,
                                statistics_gde_path,
                            )

                    logger.info(
                        f"Saved mutual information to {mutual_information_path}"
                    )
                    logger.info(f"Saved stats gde to {statistics_gde_path}")

                    if geometry == "hyperbolic" and not use_whitening:
                        radius_stratum_path = save_to_dir / "radius_stratum.json"
                        if not radius_stratum_path.exists():
                            var = radius_stratum(
                                dataset_name, resolved_model_name, stats
                            )
                            with open(radius_stratum_path, "w") as f:
                                json.dump(var, f)

                        logger.info(f"Saved radius stratum to {radius_stratum_path}")

                except Exception as e:
                    logger.error(
                        f"Error computing metrics for "
                        f"{model_cls.__name__} ({geometry=}/{normalize=}) "
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
        "--arch",
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
        "--whiten",
        action="store_true",
        help=f"If to whiten the embeddings when loaded.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    compute_selective(
        model_names=args.model or list(MODEL_REGISTRY.keys()),
        architectures=args.arch,
        pretrained=args.pretraining,
        dataset_names=args.dataset or list(get_args(DATASETS)),
        apply_normalisation=args.normalize,
        use_whitening=args.whiten,
    )
