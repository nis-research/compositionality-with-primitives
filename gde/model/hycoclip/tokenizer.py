# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

# Modified from github.com/facebookresearch/meru
# Modified from github.com/openai/CLIP

import gzip
import html

from pathlib import Path

import regex as re

import torch


class Tokenizer:
    """
    Text tokenizer that converts natural language (strings) to a padded tensor of
    token IDs using the Byte-Pair Encoding (BPE) algorithm.

    __call__ returns a Tensor of shape [B, context_length] (long, zero-padded),
    consistent with open_clip's tokenizer interface so that encode_text receives
    the same input type regardless of which model family is used.
    """

    def __init__(self, context_length: int = 77):
        self.context_length = context_length

        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        self.byte_encoder = {b: chr(b) for b in bs}
        n = 0
        for b in range(2**8):
            if b not in self.byte_encoder:
                self.byte_encoder[b] = chr(2**8 + n)
                n += 1

        bpe_path = Path(__file__).resolve().parent / "bpe_simple_vocab_16e6.txt.gz"
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def __call__(self, text: str | list[str]) -> torch.Tensor:
        import ftfy

        """
        Tokenise one or more strings and return a zero-padded long tensor.

        Args:
            text: A string or list of strings to tokenise.

        Returns:
            Tensor of shape [B, context_length] (dtype=torch.long).
            Sequences longer than context_length are truncated with the EOT
            token preserved at position context_length - 1.
        """
        text_list = [text] if isinstance(text, str) else text

        sot = self.encoder["<|startoftext|>"]
        eot = self.encoder["<|endoftext|>"]

        result = torch.zeros(len(text_list), self.context_length, dtype=torch.long)

        for i, t in enumerate(text_list):
            bpe_tokens = []

            t = ftfy.fix_text(t)
            t = html.unescape(html.unescape(t))
            t = re.sub(r"\s+", " ", t)
            t = t.strip().lower()

            for token in re.findall(self.pat, t):
                token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                bpe_tokens.extend(
                    self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
                )

            tokens = [sot, *bpe_tokens, eot]

            if len(tokens) > self.context_length:
                tokens = tokens[: self.context_length]
                tokens[-1] = eot  # Always end with EOT

            result[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        return result

    @staticmethod
    def get_pairs(word: str) -> set[str]:
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = self.get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word
