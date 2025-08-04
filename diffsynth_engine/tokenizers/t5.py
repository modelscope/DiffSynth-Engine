import os
import json
import torch
from typing import Dict, List, Union, Optional
from tokenizers import Tokenizer as TokenizerFast

from diffsynth_engine.tokenizers.base import BaseTokenizer, TOKENIZER_CONFIG_FILE


VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

T5_DEFAULT_MAX_LENGTH = 512


class T5TokenizerFast(BaseTokenizer):
    """
    Construct a "fast" T5 tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`):
            Precompiled file for initializing a fast tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        **kwargs,
    ):
        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

        fast_tokenizer = TokenizerFast.from_file(tokenizer_file)
        self._tokenizer = fast_tokenizer
        # disable truncation and padding
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

        self.model_max_length = self.model_max_length if self.model_max_length else T5_DEFAULT_MAX_LENGTH

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Union[str, os.PathLike], **kwargs):
        tokenizer_config_file = os.path.join(pretrained_model_path, TOKENIZER_CONFIG_FILE)
        with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
            init_kwargs = json.load(tokenizer_config_handle)
            init_kwargs.update(**kwargs)
        vocab_file = os.path.join(pretrained_model_path, cls.vocab_files_names["vocab_file"])
        tokenizer_file = os.path.join(pretrained_model_path, cls.vocab_files_names["tokenizer_file"])
        return cls(vocab_file=vocab_file, tokenizer_file=tokenizer_file, **init_kwargs)

    @property
    def vocab_size(self):
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def get_vocab(self):
        return self._tokenizer.get_vocab(with_added_tokens=True)

    def tokenize(self, texts: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        if isinstance(texts, str):
            encoding = self._tokenizer.encode(texts)
            return encoding.tokens

        encodings = self._tokenizer.encode_batch(texts)
        return [encoding.tokens for encoding in encodings]

    def encode(self, texts: str) -> List[int]:
        encoding = self._tokenizer.encode(texts, add_special_tokens=True)
        return encoding.ids

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=True)
        return [encoding.ids for encoding in encodings]

    def decode(
        self, ids: List[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = None
    ) -> str:
        text = self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

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
        texts = self._tokenizer.decode_batch(ids, skip_special_tokens=skip_special_tokens)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            texts = [self.clean_up_tokenization(text) for text in texts]
        return texts

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            index = self._tokenizer.token_to_id(tokens)
            return index if index is not None else self._tokenizer.token_to_id(self.unk_token)

        ids = [self._tokenizer.token_to_id(token) for token in tokens]
        return [index if index is not None else self._tokenizer.token_to_id(self.unk_token) for index in ids]

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)

        tokens = []
        for index in ids:
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._tokenizer.id_to_token(index))
        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self._tokenizer.decode(tokens)

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

        batch_ids = self.batch_encode(texts)
        for i, ids in enumerate(batch_ids):
            if len(ids) > max_length:
                ids = ids[:max_length]
                ids[-1] = self.eos_token_id
            encoded[i, : len(ids)] = torch.tensor(ids)
            attention_mask[i, : len(ids)] = torch.ones((1, len(ids)))

        return {"input_ids": encoded, "attention_mask": attention_mask}
