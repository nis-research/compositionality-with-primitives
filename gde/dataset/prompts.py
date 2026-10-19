"""
celebA  cgqa  clothing16k  mit-states  ut-zap50k  vaw-czsl  waterbirds
"""

from itertools import product


def get_class_prompts_debiasing(dataset_name):
    # Prompts are the same used in https://arxiv.org/abs/2302.00070
    # (see https://github.com/chingyaoc/debias_vl/blob/main/discriminative/main.py)
    """
    Zero-Shot Prompts
    """
    if dataset_name == "waterbirds":
        class_prompt = [
            "This is a picture of a landbird.",
            "This is a picture of a waterbird.",
        ]
        objs = ["landbird", "waterbird"]
    elif dataset_name == "celebA":
        class_prompt = [
            "A photo of a celebrity with blond hair.",
            "A photo of a celebrity with dark hair.",
        ]
        objs = ["blonde", "dark"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return class_prompt, objs


def get_pair_prompts_debiasing(dataset_name):
    # Prompts are the same used in https://arxiv.org/abs/2302.00070 (see https://github.com/chingyaoc/debias_vl/blob/main/discriminative/main.py)
    if dataset_name == "waterbirds":
        class_prompt = [
            "This is a picture of a landbird.",
            "This is a picture of a waterbird.",
        ]
        class_label = ["landbird", "waterbird"]

        spurious_prompt = [
            "This is a land background.",
            "This is a picture of a forest.",
            "This is a picture of a moutain.",
            "This is a picture of a wood.",
            "This is a water background.",
            "This is a picture of an ocean.",
            "This is a picture of a beach.",
            "This is a picture of a port.",
        ]
        spurious_label = ["land"] * 4 + ["water"] * 4

    elif dataset_name == "celebA":
        class_prompt = [
            "A photo of a celebrity with blond hair.",
            "A photo of a celebrity with dark hair.",
        ]
        class_label = ["blonde", "dark"]
        spurious_prompt = [
            "A photo of a female.",
            "A photo of a female celebrity.",
            "A photo of a woman.",
            "A photo of a male.",
            "A photo of a male celebrity.",
            "A photo of a man.",
        ]
        spurious_label = ["female"] * 3 + ["male"] * 3
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    prompt = [s + " " + c for s, c in product(spurious_prompt, class_prompt)]
    pair = list(product(spurious_label, class_label))

    return prompt, pair


def get_text_prompts(dataset, dataset_name):
    if dataset_name in {"waterbirds", "celebA"}:
        class_prompt, objs = get_class_prompts_debiasing(dataset_name)
        pair_prompts, pair_pairs = get_pair_prompts_debiasing(dataset_name)
        prompts = class_prompt + pair_prompts
        pairs = [(None, o) for o in objs] + pair_pairs
        template = "same used in https://arxiv.org/abs/2302.00070"
    else:

        def prompt_template(attr, obj):
            attr = attr.replace(".", " ").lower()
            obj = obj.replace(".", " ").lower()
            return f"an image of a {attr} {obj}"

        pairs = dataset.full_pairs
        prompts = [prompt_template(a, o) for a, o in pairs]
        template = prompt_template("ATTR", "OBJ")

    return prompts, pairs, template
