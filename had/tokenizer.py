"""Deterministic Chinese-character / English-word tokenizer, fitted on train only.

Copied from SR3M vln_benchmark/unified/tokenizer.py; class renamed, behaviour unchanged.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable
import unicodedata


TOKENIZER_VERSION = "han_char_ascii_word_number_v1"
_TOKEN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\U00030000-\U000323af]"
    r"|[a-z]+(?:'[a-z]+)*|[+-]?\d+(?:\.\d+)?"
)


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("instruction must be a string")
    tokens = _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())
    if not tokens:
        raise ValueError("instruction has no Chinese characters, English words or numbers")
    return tokens


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass
class Tokenizer:
    vocab: dict[str, int]
    max_length: int = 128
    train_corpus_sha256: str = ""
    train_instruction_count: int = 0

    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if self.vocab.get("<pad>") != 0 or self.vocab.get("<unk>") != 1:
            raise ValueError("reserved token ids must be <pad>=0, <unk>=1")
        if sorted(self.vocab.values()) != list(range(len(self.vocab))):
            raise ValueError("vocabulary ids must be unique and contiguous")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def vocab_sha256(self) -> str:
        return _digest(self.vocab)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def fit(cls, train_instructions: Iterable[str], max_length: int = 128,
            *, split: str = "train", min_frequency: int = 1) -> "Tokenizer":
        if split != "train":
            raise ValueError("vocabulary fitting is permitted only on train")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        sequences = [tokenize(text) for text in train_instructions]
        if not sequences:
            raise ValueError("cannot fit vocabulary on an empty training corpus")
        counts = Counter(token for tokens in sequences for token in tokens)
        vocab = {"<pad>": 0, "<unk>": 1}
        vocab.update({token: i + 2 for i, token in enumerate(
            sorted(token for token, count in counts.items() if count >= min_frequency))})
        # Ordering the same training routes differently must not change identity.
        corpus_sha = _digest(sorted(sequences))
        return cls(vocab, max_length, corpus_sha, len(sequences))

    def encode(self, instruction: str) -> dict:
        tokens = tokenize(instruction)
        retained = tokens[:self.max_length]
        ids = [self.vocab.get(token, 1) for token in retained]
        return {
            "input_ids": ids + [0] * (self.max_length - len(ids)),
            "length": len(ids),
            "token_count": len(tokens),
            "unknown_count": sum(token not in self.vocab for token in retained),
            "unknown_count_before_truncation": sum(token not in self.vocab for token in tokens),
            "truncated_count": max(0, len(tokens) - self.max_length),
        }

    def describe(self, instructions: Iterable[str]) -> dict:
        rows = [self.encode(text) for text in instructions]
        total = sum(row["length"] for row in rows)
        unknown = sum(row["unknown_count"] for row in rows)
        truncated = sum(row["truncated_count"] for row in rows)
        return {"instructions": len(rows), "retained_tokens": total,
                "unknown_tokens": unknown, "unknown_rate": unknown / total if total else 0.0,
                "truncated_tokens": truncated,
                "truncated_instructions": sum(row["truncated_count"] > 0 for row in rows)}

    def to_dict(self) -> dict:
        return {"version": TOKENIZER_VERSION, "fit_split": "train", "vocab": self.vocab,
                "max_length": self.max_length, "vocab_sha256": self.vocab_sha256,
                "train_corpus_sha256": self.train_corpus_sha256,
                "train_instruction_count": self.train_instruction_count}

    @classmethod
    def from_dict(cls, value: dict) -> "Tokenizer":
        if value.get("version") != TOKENIZER_VERSION or value.get("fit_split") != "train":
            raise ValueError("unsupported tokenizer version or fitting split")
        tokenizer = cls(dict(value["vocab"]), int(value["max_length"]),
                        value["train_corpus_sha256"], int(value["train_instruction_count"]))
        if value.get("vocab_sha256") != tokenizer.vocab_sha256:
            raise ValueError("vocabulary hash mismatch")
        return tokenizer

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
