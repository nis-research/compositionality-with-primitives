import math
import urllib.request

from pathlib import Path
from typing import List, Literal

import torch

from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import gde.geometry.lorentz as L

from gde.model.base import EmbeddingsModel
from gde.config import CHECKPOINTS_DIR, logger
from gde.model.hycoclip.tokenizer import Tokenizer
from gde.model.hycoclip.encoders import build_timm_vit, TransformerTextEncoder


def _download_checkpoint(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=dest.name
    ) as bar:

        def _reporthook(block_count: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                bar.total = total_size
            bar.update(block_count * block_size - bar.n)

        urllib.request.urlretrieve(url, dest, reporthook=_reporthook)


class CLIPBaseline(torch.nn.Module):
    """
    Re-implementation of the CLIP model that uses an image-text contrastive
    loss as a training objective and embeds images and text in a Euclidean space.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
    ):
        """
        Args:
            visual: ConvNet or ViT image encoder to compute image features.
            textual: Transformer-based encoder to compute text features.
            embed_dim: Size of the visual and textual embedding vectors for
                computing pairwise similarity matrix.
        """
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.embed_dim = embed_dim

        self.visual_proj = nn.Linear(visual.width, embed_dim, bias=False)
        self.textual_proj = nn.Linear(textual.width, embed_dim, bias=False)

        nn.init.normal_(self.visual_proj.weight, std=visual.width**-0.5)
        nn.init.normal_(self.textual_proj.weight, std=textual.width**-0.5)

        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        # Tokenizer lives inside the model so encode_text is self-contained.
        self.tokenizer = Tokenizer(context_length=textual.context_length)

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def encode_image(self, images: torch.Tensor, project: bool = True) -> torch.Tensor:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in [0, 1].
            project: L2-normalise features onto the unit hypersphere.
        """
        image_feats = self.visual(images)
        image_feats = self.visual_proj(image_feats)
        if project:
            image_feats = F.normalize(image_feats, dim=-1)
        return image_feats

    def encode_text(self, raw_text: List[str], project: bool = True) -> torch.Tensor:
        """
        Args:
            raw_text: Pre-tokenised tensor of shape [B, context_length] (long).
                       Produced by self.tokenizer or any compatible tokenizer.
            project: L2-normalise features onto the unit hypersphere.
        """
        tokens = self.tokenizer(raw_text).to(self.device)

        # shape: (batch_size, context_length, textual.width)
        text_feats = self.textual(tokens)

        # Extract features at the [EOT] position. EOT has the highest token ID
        # in the vocabulary, so argmax reliably finds it even with zero-padding.
        eos_indices = tokens.argmax(dim=-1)
        batch_idxs = torch.arange(text_feats.shape[0], device=self.device)
        text_feats = text_feats[batch_idxs, eos_indices]
        text_feats = self.textual_proj(text_feats)

        if project:
            text_feats = F.normalize(text_feats, dim=-1)

        return text_feats


class MERU(CLIPBaseline):
    """
    Implementation of MERU model that embeds images and text in a hyperbolic space.

    Reference: MERU paper (https://arxiv.org/abs/2304.09172)
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
    ):
        super().__init__(visual, textual, embed_dim)

        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        self.entail_weight = entail_weight

        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())

    def encode_image(self, images: torch.Tensor, project: bool = True) -> torch.Tensor:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in [0, 1].
            project: Lift features onto the Hyperboloid via exponential map.
        """
        image_feats = super().encode_image(images, project=False)
        if project:
            image_feats = image_feats * self.visual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                image_feats = L.exp_map0(image_feats, self.curv.exp())
        return image_feats

    def encode_text(self, raw_text: List[str], project: bool = True) -> torch.Tensor:
        """
        Args:
            raw_text: Pre-tokenised tensor of shape [B, context_length] (long).
            project: Lift features onto the Hyperboloid via exponential map.
        """
        text_feats = super().encode_text(raw_text, project=False)
        if project:
            text_feats = text_feats * self.textual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                text_feats = L.exp_map0(text_feats, self.curv.exp())
        return text_feats


class HyCoCLIP(MERU):
    """
    HyCoCLIP model: extends MERU to embed images and texts hierarchically in
    a hyperbolic space using localized box information.
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
    ):
        super().__init__(
            visual, textual, embed_dim, curv_init, learn_curv, entail_weight
        )


class GRITModels(EmbeddingsModel):

    textual_arch = "L12_W512"
    context_length = 77
    vocab_size = 49408

    HUGGING_FACE_BASE_LINK = "https://huggingface.co/avik-pal/hycoclip/resolve/main"
    CHECKPOINT_FORMAT = "{pretraining}_vit_{size}.pth"
    SIZE_MAP = {"small": "s", "base": "b"}
    CHECKPOINT_BASE_DIR = "grit"

    ARCH_OPTIONS = Literal["small", "base"]
    PRETRAINING_OPTIONS = Literal["clip", "meru", "hycoclip"]

    def __init__(
        self,
        architecture: ARCH_OPTIONS,
        pretraining: PRETRAINING_OPTIONS,
    ):
        super().__init__(architecture, pretraining)
        if pretraining == "clip":
            cls = CLIPBaseline
        elif pretraining == "meru":
            cls = MERU
        elif pretraining == "hycoclip":
            cls = HyCoCLIP
        else:
            raise ValueError(f"Invalid pretraining: {pretraining}")

        if architecture == "small":
            arch = "vit_small_mocov3_patch16_224"
        elif architecture == "base":
            arch = "vit_base_patch16_224"
        else:
            raise ValueError(f"Invalid size: {architecture}")

        visual_model = build_timm_vit(arch=arch)
        textual_model = TransformerTextEncoder(
            arch=self.textual_arch,
            vocab_size=self.vocab_size,
            context_length=self.context_length,
        )

        self.model = cls(visual_model, textual_model, 512)

        # try to load checkpoints from the CHECKPOINTS_DIR, otherwise download from Hugging Face
        checkpoint_name = self.CHECKPOINT_FORMAT.format(
            pretraining=pretraining, size=self.SIZE_MAP[architecture]
        )
        checkpoint_path = CHECKPOINTS_DIR / self.CHECKPOINT_BASE_DIR / checkpoint_name
        checkpoint_link = f"{self.HUGGING_FACE_BASE_LINK}/{checkpoint_name}"
        if not checkpoint_path.exists():
            logger.info(f"Downloading checkpoint from {checkpoint_link}")
            _download_checkpoint(checkpoint_link, checkpoint_path)

        _, unexpected = self.model.load_state_dict(
            torch.load(checkpoint_path, weights_only=False)["model"], strict=False
        )
        assert all([k in ["pixel_mean", "pixel_std"] for k in unexpected])

    def encode_image(self, images: torch.Tensor, project: bool = True) -> torch.Tensor:
        return self.model.encode_image(images, project)

    def encode_text(self, raw_text: List[str], project: bool = True) -> torch.Tensor:
        return self.model.encode_text(raw_text, project)

    @property
    def device(self):
        return self.model.device

    @staticmethod
    def get_name(arch, pretraining) -> str:
        return f"grit_{pretraining}_{arch}"


if __name__ == "__main__":
    _m = GRITModels("small", "meru")

    _rand_img = torch.randn(2, 3, 224, 224)
    print(_m.encode_image(_rand_img).shape)

    _rand_text = ["a ball", "a fan"]
    print(_m.encode_text(_rand_text).shape)
