import os
import json
import ftfy
import regex as re
import torch
from functools import lru_cache
from typing import Dict, List, Union, Optional

from diffsynth_engine.tokenizers.base import BaseTokenizer, TOKENIZER_CONFIG_FILE


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

CLIP_DEFAULT_MAX_LENGTH = 77


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# Modified from transformers.models.clip.tokenization_clip.CLIPTokenizer and open_clip.tokenizer.SimpleTokenizer
class CLIPTokenizer(BaseTokenizer):
    """
    Construct a CLIP tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        bos_token: Optional[str] = "<|startoftext|>",
        eos_token: Optional[str] = "<|endoftext|>",
        unk_token: Optional[str] = "<|endoftext|>",
        pad_token: Optional[str] = "<|endoftext|>",  # hack to enable padding
        **kwargs,
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}

        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

        self.model_max_length = self.model_max_length if self.model_max_length else CLIP_DEFAULT_MAX_LENGTH

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Union[str, os.PathLike], **kwargs):
        tokenizer_config_file = os.path.join(pretrained_model_path, TOKENIZER_CONFIG_FILE)
        with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
            init_kwargs = json.load(tokenizer_config_handle)
            init_kwargs.update(**kwargs)
        vocab_file = os.path.join(pretrained_model_path, cls.vocab_files_names["vocab_file"])
        merges_file = os.path.join(pretrained_model_path, cls.vocab_files_names["merges_file"])
        return cls(vocab_file=vocab_file, merges_file=merges_file, **init_kwargs)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return self.encoder

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

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
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

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
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def tokenize(self, texts: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        """Convert string to tokens."""
        if isinstance(texts, str):
            return self._tokenize(texts)

        return [self._tokenize(text) for text in texts]

    def _tokenize(self, text: str) -> List[str]:
        bpe_tokens = []
        text = whitespace_clean(ftfy.fix_text(text)).lower()

        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def encode(self, texts: str) -> List[int]:
        tokens = self.tokenize(texts)
        return self.convert_tokens_to_ids(tokens)

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def decode(
        self, ids: List[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = None
    ) -> str:
        tokens = self.convert_ids_to_tokens(ids, skip_special_tokens)
        text = self.convert_tokens_to_string(tokens)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)
        return text

    def batch_decode(
        self, ids: List[List[int]], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = None
    ) -> List[str]:
        return [self.decode(index, skip_special_tokens, clean_up_tokenization_spaces) for index in ids]

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self.encoder.get(tokens, self.encoder.get(self.unk_token))

        return [self.encoder.get(token, self.encoder.get(self.unk_token)) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: bool = False) -> List[str]:
        if isinstance(ids, int):
            return self.decoder.get(ids)

        tokens = []
        for index in ids:
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self.decoder.get(index))
        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        text = "".join(tokens)
        byte_array = bytearray([self.byte_decoder[c] for c in text])
        text = byte_array.decode("utf-8", errors="replace").replace("</w>", " ").strip()
        return text

    def __call__(
        self,
        texts: Union[str, List[str]],
        max_length: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, "torch.Tensor"]:
        """
        Tokenize text and prepare for model inputs.

        Args:
            texts (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded.

            max_length (`int`, *optional*):
                Each encoded sequence will be truncated or padded to max_length.

        Returns:
            `Dict[str, "torch.Tensor"]`: tensor dict compatible with model_input_names.
        """

        if isinstance(texts, str):
            texts = [texts]

        max_length = max_length if max_length else self.model_max_length

        encoded = torch.zeros(len(texts), max_length, dtype=torch.long)
        encoded.fill_(self.pad_token_id)
        attention_mask = torch.zeros(len(texts), max_length, dtype=torch.long)

        for i, text in enumerate(texts):
            tokens = self.tokenize(text)
            ids = [self.bos_token_id] + self.convert_tokens_to_ids(tokens) + [self.eos_token_id]
            if len(ids) > max_length:
                ids = ids[:max_length]
                ids[-1] = self.eos_token_id
            encoded[i, : len(ids)] = torch.tensor(ids)
            attention_mask[i, : len(ids)] = torch.ones((1, len(ids)))

        return {"input_ids": encoded, "attention_mask": attention_mask}
