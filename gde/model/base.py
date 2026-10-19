import abc
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from tqdm import tqdm

from gde.dataset import CompositionDataset
from gde.config import EMBEDDING_DIR, logger
from gde.dataset.prompts import get_text_prompts


def _chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class EmbeddingsModel(torch.nn.Module, abc.ABC):
    """
    Abstract base for vision-language models that produce image and text embeddings.

    Concrete subclasses must implement:
        encode_image(images: Tensor, project: bool) -> Tensor
        encode_text(token_ids: Tensor, project: bool) -> Tensor

    For compute_embeddings to work, subclasses must also set:
        self.tokenizer   — callable(list[str]) -> Tensor[B, context_length]
        self.preprocess  — callable(PIL.Image) -> Tensor
        self.architecture, self.pretraining  — strings used for output path
    """

    OBJ_TEMPLATE = "A picture of {}"
    ATTR_TEMPLATE = "A picture of something {}"

    def __init__(self, architecture: str, pretraining: str):
        super().__init__()
        self.architecture = architecture
        self.pretraining = pretraining

    @staticmethod
    def get_name(arch, pretraining) -> str:
        """Return a human-readable name for this model, e.g. "CLIP ViT-L/14 (OpenAI)"""
        raise NotImplementedError

    @abc.abstractmethod
    def encode_image(self, images: torch.Tensor, project: bool = True) -> torch.Tensor:
        """Encode a batch of preprocessed images."""
        raise NotImplementedError

    @abc.abstractmethod
    def encode_text(self, raw_text: List[str], project: bool = True) -> torch.Tensor:
        """Encode a batch of tokenised text. token_ids: [B, context_length]."""
        raise NotImplementedError

    def forward(
        self, images: torch.Tensor, raw_text: List[str]
    ) -> dict[str, torch.Tensor]:
        img_proj = self.encode_image(images, project=True)
        text_proj = self.encode_text(raw_text, project=True)
        return {"img_proj": img_proj, "text_proj": text_proj}

    # ------------------------------------------------------------------
    # Embedding pre-computation
    # ------------------------------------------------------------------

    def _out_dir(self, dataset_name: str) -> Path:
        out = (
            EMBEDDING_DIR
            / dataset_name
            / self.get_name(self.architecture, self.pretraining)
        )
        out.mkdir(parents=True, exist_ok=True)
        return out

    @torch.no_grad()
    def compute_text_embeddings(
        self,
        dataset: CompositionDataset,
        dataset_name: str,
        chunk_size: int = 128,
        project=True,
    ) -> Path:
        """
        Tokenise every (attr, obj) prompt, encode with the text tower, and write to
        EMBEDDING_DIR/<arch>/<pretraining>/<dataset>/text_embeddings.pt

        Saved dict: {"embeddings": Tensor[N, D], "pairs": list[tuple], "prompt template": str}
        """
        out_path = self._out_dir(dataset_name) / "text_embeddings.pt"
        if out_path.exists():
            logger.info("Text embeddings already exist at {}. Skipping.", out_path)
            return out_path

        prompts, pairs, template = get_text_prompts(dataset, dataset_name)

        text_embeddings: list[torch.Tensor] = []
        for chunk in tqdm(
            _chunks(prompts, chunk_size),
            total=(len(prompts) + chunk_size - 1) // chunk_size,
            desc="Computing text embeddings",
        ):
            emb = self.encode_text(chunk, project=project)
            text_embeddings.append(emb.cpu())

        torch.save(
            {
                "embeddings": torch.cat(text_embeddings),
                "pairs": pairs,
                "prompt template": template,
            },
            out_path,
        )
        logger.info("Saved {} text embeddings -> {}", len(pairs), out_path)
        return out_path

    @torch.no_grad()
    def compute_image_embeddings(
        self,
        dataset: CompositionDataset,
        dataset_name: str,
        chunk_size: int = 64,
        project=True,
    ) -> Path:
        """
        Encode every image in the dataset and write to
        EMBEDDING_DIR/<arch>/<pretraining>/<dataset>/image_embeddings.pt

        Saved dict: {"image_ids": list[str], "embeddings": Tensor[N, D]}
        """
        out_path = self._out_dir(dataset_name) / "image_embeddings.pt"
        if out_path.exists():
            logger.info("Image embeddings already exist at {}. Skipping.", out_path)
            return out_path

        all_images, *_ = zip(*dataset.data)
        all_images = list(all_images)

        image_embeddings: list[torch.Tensor] = []
        for img_chunk in tqdm(
            _chunks(all_images, chunk_size),
            total=(len(all_images) + chunk_size - 1) // chunk_size,
            desc="Computing image embeddings",
        ):
            imgs = [dataset.loader(p) for p in img_chunk]
            imgs = torch.stack(imgs).to(self.device)
            emb = self.encode_image(imgs, project=project)
            image_embeddings.append(emb.cpu())

        torch.save(
            {
                "image_ids": all_images,
                "embeddings": torch.cat(image_embeddings),
            },
            out_path,
        )
        logger.info("Saved {} image embeddings -> {}", len(all_images), out_path)
        return out_path

    @torch.no_grad()
    def compute_primitive_embeddings(
        self,
        dataset: CompositionDataset,
        dataset_name: str,
        chunk_size: int = 128,
        project: bool = True,
    ) -> Path:
        """Encode each attribute and object separately and save to disk.

        Prompts:
            attrs → ATTR_TEMPLATE.format(attr)
            objs  → OBJ_TEMPLATE.format(obj)

        Args:
            dataset: source of attrs/objs lists.
            dataset_name: used to build the output path.
            project: L2-normalise the embeddings before saving.
            chunk_size: text encoding batch size.

        Returns:
            Path to the saved .pt file.
        """
        out_path = self._out_dir(dataset_name) / "primitive_embeddings.pt"

        if out_path.exists():
            logger.info("Primitive embeddings already exist at {}. Skipping.", out_path)
            return out_path

        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.eval()

        attrs = dataset.attrs
        objs = dataset.objs
        logger.info(
            "Generating primitive embeddings | {} attrs, {} objs | project={}",
            len(attrs),
            len(objs),
            project,
        )

        def _encode(prompts: List[str]) -> torch.Tensor:
            chunks_out: list[torch.Tensor] = []
            for chunk in tqdm(
                _chunks(prompts, chunk_size),
                total=(len(prompts) + chunk_size - 1) // chunk_size,
                leave=False,
            ):
                with torch.no_grad():
                    emb = self.encode_text(chunk, project=False)
                chunks_out.append(emb.cpu())
            embs = torch.cat(chunks_out)
            if project:
                embs = F.normalize(embs, dim=-1)
            return embs

        attr_prompts = [self.ATTR_TEMPLATE.format(a.replace("_", " ")) for a in attrs]
        obj_prompts = [self.OBJ_TEMPLATE.format(o.replace("_", " ")) for o in objs]

        logger.info("Encoding {} attribute prompts...", len(attr_prompts))
        attr_embs = _encode(attr_prompts)
        logger.info("Encoding {} object prompts...", len(obj_prompts))
        obj_embs = _encode(obj_prompts)

        torch.save(
            {
                "attr_template": self.ATTR_TEMPLATE,
                "obj_template": self.OBJ_TEMPLATE,
                "attrs": list(attrs),
                "objs": list(objs),
                "attr_embs": attr_embs,
                "obj_embs": obj_embs,
            },
            out_path,
        )

        logger.info(
            "Saved primitive embeddings -> {} | attr_embs: {} | obj_embs: {}",
            out_path,
            tuple(attr_embs.shape),
            tuple(obj_embs.shape),
        )

        return out_path

    def compute_embeddings(
        self, dataset: CompositionDataset, dataset_name: str, project: bool
    ) -> dict[str, Path]:
        """Compute and persist both text and image embeddings."""
        self.eval()
        return {
            "text": self.compute_text_embeddings(
                dataset, dataset_name, project=project
            ),
            "image": self.compute_image_embeddings(
                dataset, dataset_name, project=project
            ),
            "primitives": self.compute_primitive_embeddings(
                dataset, dataset_name, project=project
            ),
        }
