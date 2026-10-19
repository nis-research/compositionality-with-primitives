from pathlib import Path
from typing import Literal, List

import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoModel

from gde.config import logger
from gde.dataset import CompositionDataset
from gde.model.base import EmbeddingsModel, _chunks


def _pair_prompt(attr: str, obj: str) -> str:
    return f"{attr.replace('.', ' ').lower()} {obj.replace('.', ' ').lower()}"


class Talk2DiNo(EmbeddingsModel):

    hf_user = "lorebianchi98"

    arch = "VIT"
    base_string = "Talk2DINO"

    ARCH_OPTIONS = Literal["base", "large"]
    PRETRAINING_OPTIONS = Literal["v2", "v3"]

    def __init__(
        self,
        architecture: ARCH_OPTIONS = "base",
        pretraining: PRETRAINING_OPTIONS = "v2",
    ):
        from clip import clip

        super().__init__(architecture, pretraining)
        self.size = architecture

        model_name = self.base_string

        if pretraining == "v3":
            model_name = f"{model_name}{pretraining}"
        elif pretraining != "v2":
            raise ValueError(f"Invalid variant: {pretraining}")

        model_name = f"{model_name}-{self.arch}"

        if architecture == "base":
            model_name = f"{model_name}B"
        elif architecture == "large":
            model_name = f"{model_name}L"
        else:
            raise ValueError(f"Invalid size: {architecture}")

        self.architecture = "talk2dino"
        self.variant = f"{pretraining}"
        self.pretraining = f"{pretraining}_{architecture}"

        self.model = AutoModel.from_pretrained(
            f"{self.hf_user}/{model_name}",
            trust_remote_code=True,
        )
        self.model.clip_model = clip.load("ViT-B/16", device="cpu")[0]

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def get_name(arch, pretraining) -> str:
        return f"talk2dino_{pretraining}_{arch}"

    def encode_text(self, raw_text: List[str], project: bool = True) -> torch.Tensor:
        feats = self.model.encode_text(raw_text)
        if project:
            return F.normalize(feats, dim=-1)
        return feats

    def encode_image(self, images: torch.Tensor, project: bool = True) -> torch.Tensor:
        if self.variant == "v2":
            feats = self.model.model.forward_features(images)["x_norm_patchtokens"]
        else:
            feats = self.model.model.forward_features(images)[
                :, 5:, :
            ]  # one cls token + 4 register tokens
        if project:
            return F.normalize(feats, dim=-1)
        return feats

    @torch.no_grad()
    def compute_image_embeddings(
        self,
        dataset: CompositionDataset,
        dataset_name: str,
        chunk_size: int = 64,
        project: bool = True,
    ) -> Path:
        out_path = self._out_dir(dataset_name) / "image_embeddings.pt"
        if out_path.exists():
            logger.info("Image embeddings already exist at {}. Skipping.", out_path)
            return out_path

        items = [(img, attr, obj) for img, attr, obj in dataset.data]
        image_embeddings: list[torch.Tensor] = []

        for chunk in tqdm(
            _chunks(items, chunk_size),
            total=(len(items) + chunk_size - 1) // chunk_size,
            desc="Computing image embeddings",
        ):
            img_paths, attrs, objs = zip(*chunk)
            prompts = [_pair_prompt(a, o) for a, o in zip(attrs, objs)]

            imgs = [dataset.loader(p) for p in img_paths]
            imgs = torch.stack(imgs).to(self.device)

            # patch_tokens: [B, P, D]
            patch_tokens = self.encode_image(imgs, project=False)

            # text_emb: [B, D] — one embedding per image matching its (attr, obj) label
            text_emb = self.encode_text(prompts, project=False)

            # cosine similarity heatmap: [B, P]
            patch_norm = F.normalize(patch_tokens, dim=-1)
            text_norm = F.normalize(text_emb, dim=-1)
            sim = torch.einsum("bd,bpd->bp", text_norm, patch_norm)

            # softmax weights over patch dimension: [B, P]
            weights = torch.softmax(sim, dim=-1)

            # weighted mean of patch tokens: [B, D]
            img_emb = torch.einsum("bp,bpd->bd", weights, patch_tokens)

            if project:
                img_emb = F.normalize(img_emb, dim=-1)

            image_embeddings.append(img_emb.cpu())

        all_images = [img for img, _, _ in items]
        all_image_emb = torch.cat(image_embeddings)  # [N, D]
        torch.save({"image_ids": all_images, "embeddings": all_image_emb}, out_path)
        logger.info("Saved {} image embeddings -> {}", len(all_images), out_path)
        return out_path


if __name__ == "__main__":
    _m = Talk2DiNo()
    _m.to("cuda")

    _text = ["red ball", "woman"]
    _img = torch.rand(2, 3, 224, 224, device="cuda")

    print(_m.encode_image(_img, project=True).shape)
    print(_m.encode_text(_text, project=True).shape)
