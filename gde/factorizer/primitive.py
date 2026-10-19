import torch

import gde.geometry.sphere as SG
import gde.factorizer.lorentz_geopt as LG

from gde.registry import registry
from gde.factorizer.base import (
    CompositionalFactorizer,
    compute_group_means,
)


@registry.register("factorizer", "prim_e")
class PrimitiveEuclidean(CompositionalFactorizer):

    def compute_ideal_words(self, embeddings, all_pairs_gt, weights):
        attrs, objs = zip(*all_pairs_gt)

        # 1. Per-object arithmetic mean (weighted)
        obj_iw = compute_group_means(
            embeddings, list(objs), self.objs, weights
        )  # [num_objs, D]

        attr_iw = compute_group_means(
            embeddings, list(attrs), self.attrs, weights
        )  # [num_attrs, D]

        return None, attr_iw, obj_iw

    def combine_ideal_words(self, *ideal_words, context=None):
        raise NotImplementedError(
            "It is not possible to combine ideal words for this factorizer"
        )

    def to_tangent_at(self, embeddings):
        raise NotImplementedError(
            "It is not possible to project to a tangent for this factorizer"
        )


@registry.register("factorizer", "prim_s")
class PrimitiveSphere(CompositionalFactorizer):

    def compute_ideal_words(self, embeddings, all_pairs_gt, weights):
        # 1. Per-object intrinsic mean on the sphere (weighted)
        obj_means = []
        for obj in self.objs:
            mask = [i for i, (_, o) in enumerate(all_pairs_gt) if o == obj]
            obj_embs = embeddings[mask]
            obj_weights = None if weights is None else weights[mask]
            obj_mean = SG.calculate_intrinstic_mean(
                obj_embs, obj_weights, init="normalized mean"
            )
            obj_means.append(obj_mean)
        obj_iw = torch.stack(obj_means)  # [num_objs, D]

        # 1. Per-object intrinsic mean on the sphere (weighted)
        attr_means = []
        for attr in self.attrs:
            mask = [i for i, (a, _) in enumerate(all_pairs_gt) if a == attr]
            attr_embs = embeddings[mask]
            attr_weights = None if weights is None else weights[mask]
            attr_mean = SG.calculate_intrinstic_mean(
                attr_embs, attr_weights, init="normalized mean"
            )
            attr_means.append(attr_mean)
        attr_iw = torch.stack(attr_means)  # [num_objs, D]

        return None, attr_iw, obj_iw

    def combine_ideal_words(self, *ideal_words, context=None):
        raise NotImplementedError(
            "It is not possible to combine ideal words for this factorizer"
        )

    def to_tangent_at(self, embeddings):
        raise NotImplementedError(
            "It is not possible to project to a tangent for this factorizer"
        )


@registry.register("factorizer", "prim_hy")
class PrimitiveHyperbolic(CompositionalFactorizer):
    curv = torch.tensor(0.1)

    def compute_ideal_words(self, embeddings, all_pairs_gt, weights):
        k = 1 / self.curv

        # Lift to Lorentz hyperboloid
        t = torch.sqrt(k + torch.sum(embeddings**2, dim=-1, keepdim=True))
        embs_lorentz = torch.cat((t, embeddings), dim=1)  # [N, D+1]

        # 1. Per-object intrinsic mean on the hyperboloid (weighted)
        obj_means = []
        for obj in self.objs:
            mask = [i for i, (_, o) in enumerate(all_pairs_gt) if o == obj]
            obj_embs = embs_lorentz[mask]
            obj_weights = None if weights is None else weights[mask]
            obj_mean = LG.calculate_intrinstic_mean(obj_embs, k=k, weights=obj_weights)
            obj_means.append(obj_mean)
        obj_iw = torch.stack(obj_means)  # [num_objs, D+1]

        # 1. Per-object intrinsic mean on the hyperboloid (weighted)
        attr_means = []
        for attr in self.attrs:
            mask = [i for i, (a, _) in enumerate(all_pairs_gt) if a == attr]
            attr_embs = embs_lorentz[mask]
            attr_weights = None if weights is None else weights[mask]
            attr_mean = LG.calculate_intrinstic_mean(
                attr_embs, k=k, weights=attr_weights
            )
            attr_means.append(attr_mean)
        attr_iw = torch.stack(attr_means)  # [num_objs, D+1]

        return None, attr_iw, obj_iw

    def combine_ideal_words(self, *ideal_words, context=None):
        raise NotImplementedError(
            "It is not possible to combine ideal words for this factorizer"
        )

    def to_tangent_at(self, embeddings):
        raise NotImplementedError(
            "It is not possible to project to a tangent for this factorizer"
        )
