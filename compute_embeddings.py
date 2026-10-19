import argparse

from typing import get_args
from itertools import product

import torch
import torchvision.transforms as T

from gde.config import logger
from gde.model import GRITModels, OpenCLIP, Talk2DiNo
from gde.dataset.base import CompositionDataset, DATASETS

MODEL_REGISTRY = {
    "openclip": OpenCLIP,
    "talk2dino": Talk2DiNo,
    "grit": GRITModels,
}


def get_default_transform():
    return T.Compose(
        [
            T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def compute_all():
    for dataset_name in get_args(DATASETS):
        dataset = CompositionDataset(
            dataset_name, phase="all", transform=get_default_transform()
        )
        for model_cls_name, model_cls in MODEL_REGISTRY.items():
            for arch, pre in product(
                get_args(model_cls.ARCH_OPTIONS),
                get_args(model_cls.PRETRAINING_OPTIONS),
            ):
                try:
                    model = model_cls(architecture=arch, pretraining=pre)
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    model.to(device)
                    model.compute_embeddings(
                        dataset, dataset_name, project=model_cls_name != "talk2dino"
                    )
                except Exception as e:
                    logger.error(
                        f"Error computing embeddings for {model_cls.__name__} with arch={arch}, pretraining={pre} on dataset {dataset_name}: {e}"
                    )


def compute_selective(model_names, architectures, pretrained, dataset_names):
    for dataset_name in dataset_names:
        dataset = CompositionDataset(
            dataset_name, phase="all", transform=get_default_transform()
        )
        for model_name, model_cls in [(n, MODEL_REGISTRY[n]) for n in model_names]:
            valid_archs = get_args(model_cls.ARCH_OPTIONS)
            valid_pres = get_args(model_cls.PRETRAINING_OPTIONS)

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

            logger.info(
                f"Running {model_cls.__name__} on {dataset_name} with architectures {archs} and pretraining {pres}"
            )

            for arch, pre in product(archs, pres):
                try:
                    model = model_cls(architecture=arch, pretraining=pre)
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    model.to(device)
                    model.compute_embeddings(
                        dataset, dataset_name, project=model_name != "talk2dino"
                    )
                except Exception as e:
                    logger.error(
                        f"Error computing embeddings for {model_cls.__name__} with arch={arch}, pretraining={pre} on dataset {dataset_name}: {e}"
                    )


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
        description="Compute VLM embeddings on composition datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compute embeddings for every model × architecture × pretraining × dataset combination.",
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
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.all:
        compute_all()
    else:
        compute_selective(
            model_names=args.model or list(MODEL_REGISTRY.keys()),
            architectures=args.architecture,
            pretrained=args.pretraining,
            dataset_names=args.dataset or list(get_args(DATASETS)),
        )
