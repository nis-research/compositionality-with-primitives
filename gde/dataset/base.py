from typing import Literal
from itertools import product

import torch

from PIL import Image
from torch.utils.data import Dataset

from gde.config import DATASET_DIR, EMBEDDING_DIR, logger

PHASES = Literal["train", "val", "test", "all"]
DATASETS = Literal[
    "celebA", "cgqa", "clothing16k", "mit-states", "ut-zap50k", "vaw-czsl", "waterbirds"
]


class ImageLoader:
    def __init__(self, root, preprocess):
        self.img_dir = root
        self.preprocess = preprocess

    def __call__(self, img):
        pil_image = Image.open(self.img_dir / img).convert("RGB")
        if self.preprocess is not None:
            return self.preprocess(pil_image)
        else:
            return pil_image


def _load_pairs(filepath):
    pairs = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            delimiter = "+" if "+" in line else " "
            attr, obj = line.split(delimiter, 1)
            pairs.append((attr, obj))
    logger.debug("Loaded {} pairs from {}", len(pairs), filepath)
    return pairs


class CompositionDataset(Dataset):
    METADATA_FILENAME = "metadata_{}.t7"

    def __init__(
        self,
        dataset: DATASETS,
        open_world=False,
        phase: PHASES = "all",
        split="compositional-split-natural",
        transform=None,
    ):
        self.root = DATASET_DIR / dataset

        self.phase = phase
        self.split = split
        self.transform = transform
        self.open_world = open_world
        self.loader = ImageLoader(self.root / "images", transform)

        logger.info(
            "Dataset: {} | phase: {} | open_world: {}", dataset, phase, open_world
        )
        logger.debug("Dataset root: {}", self.root)

        (
            self.attrs,
            self.objs,
            self.pairs,
            self.train_pairs,
            self.val_pairs,
            self.test_pairs,
        ) = self._parse_split()

        self.full_pairs = list(product(self.attrs, self.objs))

        if self.open_world:
            self.pairs = self.full_pairs

        logger.debug(
            "Pairs -> train: {}, val: {}, test: {} | attrs: {}, objs: {}",
            len(self.train_pairs),
            len(self.val_pairs),
            len(self.test_pairs),
            len(self.attrs),
            len(self.objs),
        )

        self.train_data, self.val_data, self.test_data = self._get_split_info()

        if phase == "train":
            self.data = self.train_data
        elif phase == "val":
            self.data = self.val_data
        elif phase == "test":
            self.data = self.test_data
        elif phase == "all":
            self.data = self.train_data + self.val_data + self.test_data
        else:
            raise ValueError(
                f"Invalid phase: {phase}. Expected one of 'train', 'val', 'test', 'all'."
            )

        _, self.all_attrs, self.all_objs = zip(*self.data)
        self.all_pairs = list(zip(self.all_attrs, self.all_objs))

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.idx2obj = {idx: obj for idx, obj in self.obj2idx.items()}

        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.idx2attr = {idx: attr for idx, attr in self.attr2idx.items()}

        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.train_pair_to_idx = {
            pair: idx for idx, pair in enumerate(self.train_pairs)
        }

        seen_pairs = set(self.train_pairs)
        self.seen_mask = torch.BoolTensor([pair in seen_pairs for pair in self.pairs])

        self.objs_by_attr = {k: [] for k in self.attrs}
        self.attrs_by_obj = {k: [] for k in self.objs}

        for a, o in self.all_pairs:
            self.objs_by_attr[a].append(o)
            self.attrs_by_obj[o].append(a)

        logger.info("\n{}", str(self))

    # -----------------------------------------------------------------
    # Split parsing
    # -----------------------------------------------------------------
    def _parse_split(self):
        split_dir = self.root / self.split

        tr_pairs = _load_pairs(split_dir / "train_pairs.txt")
        vl_pairs = _load_pairs(split_dir / "val_pairs.txt")
        ts_pairs = _load_pairs(split_dir / "test_pairs.txt")

        tr_attrs, tr_objs = zip(*tr_pairs)
        vl_attrs, vl_objs = zip(*vl_pairs)
        ts_attrs, ts_objs = zip(*ts_pairs)

        all_attrs = sorted(set(tr_attrs + vl_attrs + ts_attrs))
        all_objs = sorted(set(tr_objs + vl_objs + ts_objs))
        all_pairs = sorted(set(tr_pairs + vl_pairs + ts_pairs))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    # -----------------------------------------------------------------
    # Metadata loading
    # -----------------------------------------------------------------
    def _get_split_info(self):
        metadata_path = self.root / self.METADATA_FILENAME.format(self.split)
        logger.debug("Loading metadata from {}", metadata_path)

        metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)

        pairs = set(self.pairs)
        train_data, val_data, test_data = [], [], []

        for instance in metadata:
            image, attr, obj, settype = (
                instance["image"],
                instance["attr"],
                instance["obj"],
                instance["set"],
            )
            if attr == "NA" or (attr, obj) not in pairs or settype == "NA":
                continue
            item = [image, attr, obj]
            if settype == "train":
                train_data.append(item)
            elif settype == "val":
                val_data.append(item)
            else:
                test_data.append(item)

        logger.debug(
            "Split info -> train: {}, val: {}, test: {}",
            len(train_data),
            len(val_data),
            len(test_data),
        )
        return train_data, val_data, test_data

    # -----------------------------------------------------------------
    # Dataset protocol
    # -----------------------------------------------------------------
    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        if self.phase == "train":
            return [
                img,
                self.attr2idx[attr],
                self.obj2idx[obj],
                self.train_pair_to_idx[(attr, obj)],
            ]
        return [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        seen_pairs = set(self.train_pairs)

        n_seen_val = len(seen_pairs & set(self.val_pairs))
        n_seen_test = len(seen_pairs & set(self.test_pairs))

        _, attr_val, obj_val = zip(*self.val_data)
        n_seen_img_val = sum(p in seen_pairs for p in zip(attr_val, obj_val))

        _, attr_test, obj_test = zip(*self.test_data)
        n_seen_img_test = sum(p in seen_pairs for p in zip(attr_test, obj_test))

        return "\n".join(
            [
                " # train pairs : {:<7} | # val pairs : {:<7} ({:^5} seen) | # test pairs : {:<7} ({:^5} seen)".format(
                    len(self.train_pairs),
                    len(self.val_pairs),
                    n_seen_val,
                    len(self.test_pairs),
                    n_seen_test,
                ),
                " # train images: {:<7} | # val images: {:<7} ({:^5} seen) | # test images: {:<7} ({:^5} seen)".format(
                    len(self.train_data),
                    len(self.val_data),
                    n_seen_img_val,
                    len(self.test_data),
                    n_seen_img_test,
                ),
                " # attrs : {:<10} # objs : {:<10} # full pairs {:<10}".format(
                    len(self.attrs),
                    len(self.objs),
                    len(self.full_pairs),
                ),
            ]
        )


class CompositionDatasetEmbeddings(CompositionDataset):
    """
    Extends CompositionDataset with precomputed image and text embedding loading.

    Image embeddings: EMBEDDING_DIR / dataset / model_name / image_embeddings.pt
    Text embeddings:  EMBEDDING_DIR / dataset / model_name / text_embeddings.pt
    Primitive embeddings:  EMBEDDING_DIR / dataset / model_name / primitive_embeddings.pt
    """

    IMAGE_EMBEDDINGS_FILENAME = "image_embeddings.pt"
    TEXT_EMBEDDINGS_FILENAME = "text_embeddings.pt"
    PRIMITIVE_EMBEDDINGS_FILENAME = "primitive_embeddings.pt"

    def __init__(
        self,
        dataset,
        phase,
        model_name,
        split="compositional-split-natural",
        open_world=False,
    ):
        super().__init__(dataset, open_world, phase, split)

        self.embedding_dir = EMBEDDING_DIR / dataset / model_name

        self.text_embs_path = self.embedding_dir / self.TEXT_EMBEDDINGS_FILENAME
        self.image_embs_path = self.embedding_dir / self.IMAGE_EMBEDDINGS_FILENAME
        self.primitive_embs_path = (
            self.embedding_dir / self.PRIMITIVE_EMBEDDINGS_FILENAME
        )

        logger.debug("Embedding directory: {}", self.embedding_dir)

    # -----------------------------------------------------------------
    # Embedding loading
    # -----------------------------------------------------------------
    def load_text_embs(self, pairs, train=True, normalize=False):
        """
        Load text embeddings for the given (attr, obj) pairs.

        Returns the embedding tensor when there is a one-to-one correspondence
        with the requested pairs, otherwise returns (embeddings, resolved_pairs).
        """
        logger.debug("Loading text embeddings from {}", self.text_embs_path)
        data = torch.load(self.text_embs_path, map_location="cpu", weights_only=False)

        pair2indices = {}
        for i, pair in enumerate(data["pairs"]):
            pair2indices[tuple(pair)] = i
        indices = [pair2indices[pair] for pair in pairs]

        text_embs = data["embeddings"][indices]
        if normalize:
            text_embs = torch.nn.functional.normalize(text_embs, eps=1e-6, dim=-1, p=2)

        logger.info(
            "Loaded {} text embeddings for {} pairs", len(text_embs), len(pairs)
        )

        if not train:
            return text_embs
        return text_embs, [tuple(data["pairs"][i]) for i in indices]

    def load_text_primitive_embs(self, normalize=False):
        """
        Load primitive embeddings for the given (attr, obj) pairs.

        Returns
        -------
        attr_embeddings : torch.Tensor  shape [N_a, D]
        obj_embeddings : torch.Tensor  shape [N_o, D]
        """
        logger.debug("Loading primitive embeddings from {}", self.primitive_embs_path)
        data = torch.load(
            self.primitive_embs_path, map_location="cpu", weights_only=False
        )

        attr_embs = data["attr_embs"]
        if normalize:
            attr_embs = torch.nn.functional.normalize(attr_embs, eps=1e-6, dim=-1, p=2)
        attrs = data["attrs"]

        obj_embs = data["obj_embs"]
        if normalize:
            obj_embs = torch.nn.functional.normalize(obj_embs, eps=1e-6, dim=-1, p=2)
        objs = data["objs"]

        logger.info(
            "Loaded primitive embeddings -> attr_embs: {} | obj_embs: {}",
            tuple(attr_embs.shape),
            tuple(obj_embs.shape),
        )

        return (attr_embs, attrs), (obj_embs, objs)

    def load_all_image_embs(self, normalize=False):
        """
        Load all image embeddings for the active phase, aligned to self.data order.

        Returns
        -------
        embeddings : torch.Tensor  shape [N, D]
        pairs : list[tuple[str, str]]
        """
        logger.debug("Loading image embeddings from {}", self.image_embs_path)
        data = torch.load(self.image_embs_path, map_location="cpu", weights_only=False)

        image_id2idx = {id_: i for i, id_ in enumerate(data["image_ids"])}
        all_image_ids, all_attrs, all_objs = zip(*self.data)
        indices = [image_id2idx[id_] for id_ in all_image_ids]

        image_embs = data["embeddings"][indices]
        if normalize:
            image_embs = torch.nn.functional.normalize(
                image_embs, eps=1e-6, dim=-1, p=2
            )

        logger.info(
            "Loaded {} image embeddings (shape {}) for phase '{}'",
            len(image_embs),
            tuple(image_embs.shape),
            self.phase,
        )
        return image_embs, list(zip(all_attrs, all_objs))

    def load_all_text_embs(self, open_world, normalize=False):
        """
        Load all text embeddings for the active phase, aligned to self.data order.

        Returns
        -------
        embeddings : torch.Tensor  shape [N, D]
        pairs : list[tuple[str, str]]
        """
        logger.debug("Loading text embeddings from {}", self.text_embs_path)
        data = torch.load(self.text_embs_path, map_location="cpu", weights_only=False)

        pair2idx = {tuple(pair): i for i, pair in enumerate(data["pairs"])}
        if not open_world:
            all_attrs, all_objs = zip(*sorted(set(self.train_pairs + self.test_pairs)))
        else:
            all_attrs, all_objs = zip(*sorted(set(self.full_pairs)))
        indices = [pair2idx[(a, o)] for a, o in zip(all_attrs, all_objs)]

        text_embs = data["embeddings"][indices]
        if normalize:
            text_embs = torch.nn.functional.normalize(text_embs, eps=1e-6, dim=-1, p=2)

        test_pairs = list(zip(all_attrs, all_objs))

        logger.info(
            "Loaded {} text embeddings (shape {}) for scenario open_world={}",
            len(text_embs),
            tuple(text_embs.shape),
            open_world,
        )
        return text_embs, test_pairs


if __name__ == "__main__":
    _ds = CompositionDataset(dataset="vaw-czsl", open_world=True)
    print(_ds.data[0])
