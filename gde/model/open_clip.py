from typing import List, Literal

import torch

from gde.model.base import EmbeddingsModel


class OpenCLIP(EmbeddingsModel):
    """
    Thin nn.Module wrapper around open_clip for spherical (L2-normalised)
    image/text embeddings.

    forward() returns {"img_proj": ..., "text_proj": ...} — both L2-normalised.
    compute_text_embeddings() / compute_image_embeddings() write pre-computed
    features to EMBEDDING_DIR in the layout expected by
    CompositionDatasetEmbeddings.
    """

    ARCH_OPTIONS = Literal["ViT-B-16", "ViT-L-14"]
    PRETRAINING_OPTIONS = Literal["openai"]

    def __init__(
        self,
        architecture: ARCH_OPTIONS,
        pretraining: PRETRAINING_OPTIONS,
        device: str | torch.device = "cpu",
    ):
        import open_clip

        super().__init__(architecture, pretraining)

        arch_string = (
            f"{architecture}-quickgelu" if pretraining == "openai" else architecture
        )

        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=arch_string,
            pretrained=pretraining,
            device=device,
            # OpenAI weights were trained with QuickGELU; keep parity.
            quick_gelu=(pretraining == "openai"),
        )
        self.clip = clip_model
        self.tokenizer = open_clip.get_tokenizer(architecture)

    @staticmethod
    def get_name(arch, pretraining):
        return f"{pretraining}_{arch}"

    @property
    def device(self) -> torch.device:
        return next(self.clip.parameters()).device

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def encode_image(self, images: torch.Tensor, project: bool = True) -> torch.Tensor:
        return self.clip.encode_image(images, normalize=project).float()

    def encode_text(self, raw_text: List[str], project: bool = True) -> torch.Tensor:
        token_ids = self.tokenizer(raw_text).to(self.device)
        return self.clip.encode_text(token_ids, normalize=project).float()
