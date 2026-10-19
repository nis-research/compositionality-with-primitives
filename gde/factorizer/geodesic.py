import torch

from gde.config import logger
from gde.registry import registry
from gde.geometry.sphere import (
    calculate_intrinstic_mean,
    logarithmic_map,
    exponential_map,
    parallel_transport,
)
from gde.factorizer.base import (
    CompositionalFactorizer,
    compute_attr_obj_means,
    compute_group_means,
)
import gde.factorizer.lorentz_geopt as LG


@registry.register("factorizer", "GDE")
class GDE(CompositionalFactorizer):

    def compute_ideal_words(self, embeddings, all_pairs_gt, weights):
        intrinsic_mean = calculate_intrinstic_mean(
            embeddings, weights, init="normalized mean"
        )  # mu

        # 1) Map embedding to the tangent space T_muS^n
        embeddings_T = logarithmic_map(intrinsic_mean, embeddings)

        # 2) Compute IW on the tangent space
        v_c, attr_IW, obj_IW = compute_attr_obj_means(
            embeddings_T, all_pairs_gt, weights=weights
        )
        logger.debug("v_c norm: {:.6f}", torch.norm(v_c, p=2).item())
        # assert torch.norm(v_c, p=2) < 1e-4  # should be v_c=0

        return intrinsic_mean, attr_IW, obj_IW

    def combine_ideal_words(self, *ideal_words, context=None):
        if context is None:
            original_contex = True
            context = self.context
        else:
            original_contex = False

        # 3) Combine ideal words in the tangent plane
        ideal_words = torch.stack(ideal_words)
        embs_approx_T = torch.sum(ideal_words, dim=0)

        # 4) Map the obtained approximation back to the sphere
        if original_contex:
            embs_approx = exponential_map(context, embs_approx_T)
        else:
            # If context is not mu, we need to transport embs_approx_T from T_muS^n to T_contextS^n
            embs_approx_T_transported = parallel_transport(
                self.context, context, embs_approx_T
            )
            embs_approx = exponential_map(context, embs_approx_T_transported)

        return embs_approx

    def get_denoised_pair(self):
        unique_pairs = list(set(self.all_pairs_gt))
        embs_T = logarithmic_map(self.context, self.embs_for_IW)
        denoised_pair_T = compute_group_means(embs_T, self.all_pairs_gt, unique_pairs)
        denoised_pair = exponential_map(self.context, denoised_pair_T)
        return unique_pairs, denoised_pair

    def to_tangent_at(self, embeddings):
        return logarithmic_map(self.context, embeddings)


@registry.register("factorizer", "hGDE")
class HyperbolicGDE(CompositionalFactorizer):
    curv = torch.tensor(0.1)

    def compute_ideal_words(self, embeddings, all_pairs_gt, weights):
        x = embeddings
        t = torch.sqrt(1 / self.curv + torch.sum(x**2, dim=-1, keepdim=True))
        embeddings = torch.cat((t, x), dim=1)

        intrinsic_mean = LG.calculate_intrinstic_mean(embeddings, k=1 / self.curv)

        # 1) Map embedding to the tangent space T_muH
        embeddings_T = LG.logmap(intrinsic_mean, embeddings, k=1 / self.curv)

        # 2) Compute IW on the tangent space
        v_c, attr_IW, obj_IW = compute_attr_obj_means(
            embeddings_T, all_pairs_gt, weights=weights
        )
        logger.debug("v_c norm: {:.6f}", torch.norm(v_c, p=2).item())
        self.v_c = v_c

        return intrinsic_mean, attr_IW, obj_IW

    def combine_ideal_words(self, *ideal_words, context=None):
        if context is None:
            context = self.context
        else:
            raise NotImplementedError("Non-mu context not implemented")

        # 3) Combine ideal words in the tangent plane
        ideal_words = torch.stack(ideal_words)
        embs_approx_T = torch.sum(ideal_words, dim=0)

        # 4) Map the obtained approximation back to H
        embs_approx = LG.expmap(context, self.v_c + embs_approx_T, k=1 / self.curv)

        return embs_approx

    def to_tangent_at(self, embeddings):
        t = torch.sqrt(1 / self.curv + torch.sum(embeddings**2, dim=-1, keepdim=True))
        embeddings_h = torch.cat((t, embeddings), dim=1)
        return LG.logmap(self.context, embeddings_h, k=1 / self.curv)
