import os
import json
import torch
from typing import Dict, List, Union, Optional
from tokenizers import Tokenizer as TokenizerFast, AddedToken

from diffsynth_engine.tokenizers.base import BaseTokenizer, PaddingStrategy, TOKENIZER_CONFIG_FILE


VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

QWEN2_DEFAULT_MAX_LENGTH = 32768


class Qwen2TokenizerFast(BaseTokenizer):
    """
    Construct a "fast" Qwen2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    Same with GPT2Tokenizer, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> import Qwen2TokenizerFast

    >>> tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")
    >>> tokenizer("Hello world")["input_ids"]
    [9707, 1879]

    >>> tokenizer(" Hello world")["input_ids"]
    [21927, 1879]
    ```
    This is expected.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. Not applicable to this tokenizer.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        unk_token: str = "<|endoftext|>",
        bos_token: Optional[str] = None,
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        **kwargs,
    ):
        # We need to at least pass vocab_file and merges_file to base class;
        # other can be configured through files.
        # following GPT2TokenizerFast, also adding unk_token, bos_token, and eos_token
        super().__init__(unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, **kwargs)
        self._tokenizer = TokenizerFast.from_file(tokenizer_file)
        self._tokenizer.no_truncation()
        self._tokenizer.encode_special_tokens = False

        self.padding_side = kwargs.pop("padding_side", "right")
        self.model_max_length = kwargs.pop("model_max_length", QWEN2_DEFAULT_MAX_LENGTH)
        self.chat_template = kwargs.pop("chat_template", None)
        added_tokens_decoder = kwargs.pop("added_tokens_decoder", None)
        if added_tokens_decoder is not None and isinstance(added_tokens_decoder, Dict):
            tokens = []
            for idx, token in added_tokens_decoder.items():
                if isinstance(token, Dict):
                    token = AddedToken(**token)
                if isinstance(token, AddedToken):
                    tokens.append(token)
                else:
                    raise ValueError(
                        f"Found a {token.__class__} in the saved `added_tokens_decoder`, should be a dictionary or an AddedToken instance"
                    )
            if len(tokens) > 0:
                self._tokenizer.add_tokens(tokens)

    def tokenize(self, texts: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        if isinstance(texts, str):
            encoding = self._tokenizer.encode(texts)
            return encoding.tokens

        encodings = self._tokenizer.encode_batch(texts)
        return [encoding.tokens for encoding in encodings]

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Union[str, os.PathLike], **kwargs):
        tokenizer_config_file = os.path.join(pretrained_model_path, TOKENIZER_CONFIG_FILE)
        with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
            init_kwargs = json.load(tokenizer_config_handle)
            init_kwargs.update(**kwargs)
        vocab_file = os.path.join(pretrained_model_path, VOCAB_FILES_NAMES["vocab_file"])
        merges_fle = os.path.join(pretrained_model_path, VOCAB_FILES_NAMES["merges_file"])
        tokenizer_file = os.path.join(pretrained_model_path, VOCAB_FILES_NAMES["tokenizer_file"])
        return cls(vocab_file=vocab_file, merges_file=merges_fle, tokenizer_file=tokenizer_file, **init_kwargs)

    def encode(self, texts: str) -> List[int]:
        encoding = self._tokenizer.encode(texts, add_special_tokens=True)
        return encoding.ids

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=True)
        return [encoding.ids for encoding in encodings]

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            index = self._tokenizer.token_to_id(tokens)
            return index if index is not None else self._tokenizer.token_to_id(self.unk_token)

        ids = [self._tokenizer.token_to_id(token) for token in tokens]
        return [index if index is not None else self._tokenizer.token_to_id(self.unk_token) for index in ids]

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
        self,
        ids: List[List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
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

    def __call__(
        self,
        texts: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding_side: Optional[str] = None,
        padding_strategy: Union[PaddingStrategy, str] = "longest",
        **kwargs,
    ) -> Dict[str, "torch.Tensor"]:
        """
        Tokenize text and prepare for model inputs.

        Args:
            texts (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded.

            max_length (`int`, *optional*):
                Maximum length of the encoded sequences.

            padding_side (`str`, *optional*):
                The side on which the padding should be applied. Should be selected between `"right"` and `"left"`.
                Defaults to `"right"`.

            padding_strategy (`PaddingStrategy`, `str`, *optional*):
                If `"longest"`, will pad the sequences to the longest sequence in the batch.
                If `"max_length"`, will pad the sequences to the `max_length` argument.
                Defaults to `"longest"`.

        Returns:
            `Dict[str, "torch.Tensor"]`: tensor dict compatible with model_input_names.
        """

        if isinstance(texts, str):
            texts = [texts]

        batch_ids = self.batch_encode(texts)
        ids_lens = [len(ids_) for ids_ in batch_ids]
        max_length = max_length if max_length is not None else self.model_max_length
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = min(max(ids_lens), max_length)
        padding_side = padding_side if padding_side is not None else self.padding_side

        encoded = torch.zeros(len(texts), max_length, dtype=torch.long)
        encoded.fill_(self.pad_token_id)
        attention_mask = torch.zeros(len(texts), max_length, dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            if len(ids) > max_length:
                ids = ids[:max_length]
                ids[-1] = self.eos_token_id
            if padding_side == "right":
                encoded[i, : len(ids)] = torch.tensor(ids)
                attention_mask[i, : len(ids)] = torch.ones((1, len(ids)))
            elif padding_side == "left":
                encoded[i, -len(ids) :] = torch.tensor(ids)
                attention_mask[i, -len(ids) :] = torch.ones((1, len(ids)))

        return {"input_ids": encoded, "attention_mask": attention_mask}
