import torch

from gde.registry import registry
from gde.factorizer.base import (
    CompositionalFactorizer,
    compute_attr_obj_means,
    compute_group_means,
)


@registry.register("factorizer", "LDE")
class LDE(CompositionalFactorizer):

    def compute_ideal_words(self, embeddings, all_pairs_gt, weights):
        return compute_attr_obj_means(embeddings, all_pairs_gt, weights=weights)

    def combine_ideal_words(self, *ideal_words, context=None):
        if context is None:
            context = self.context
        ideal_words = torch.stack(ideal_words)
        return context + torch.sum(ideal_words, dim=0)

    def get_denoised_pair(self):
        unique_pairs = list(set(self.all_pairs_gt))
        denoised_pair = compute_group_means(
            self.embs_for_IW, self.all_pairs_gt, unique_pairs
        )
        return unique_pairs, denoised_pair

    def to_tangent_at(self, embeddings):
        return embeddings - self.context


@registry.register("factorizer", "hLDE")
class HyperbolicLDE(CompositionalFactorizer):
    curv = 0.1

    def compute_ideal_words(self, embeddings, all_pairs_gt, weights):
        x = embeddings
        t = torch.sqrt(1 / self.curv + torch.sum(x**2, dim=-1, keepdim=True))
        embeddings = torch.cat(
            (t, x), dim=1
        )  # Embeddings (i.e. points on H) have coordinates (t, x)

        return compute_attr_obj_means(embeddings, all_pairs_gt, weights=weights)

    def combine_ideal_words(self, *ideal_words, context=None):
        if context is None:
            context = self.context
        ideal_words = torch.stack(ideal_words)
        return context + torch.sum(ideal_words, dim=0)

    def get_denoised_pair(self):
        unique_pairs = list(set(self.all_pairs_gt))
        denoised_pair = compute_group_means(
            self.embs_for_IW, self.all_pairs_gt, unique_pairs
        )
        return unique_pairs, denoised_pair
