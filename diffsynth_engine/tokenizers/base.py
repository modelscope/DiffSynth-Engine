# Modified from transformers.tokenization_utils_base
from typing import Dict, List, Union, overload
from enum import Enum


TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


class PaddingStrategy(str, Enum):
    LONGEST = "longest"
    MAX_LENGTH = "max_length"


class BaseTokenizer:
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
    ]

    def __init__(self, **kwargs):
        self.bos_token = None
        self.eos_token = None
        self.unk_token = None
        self.pad_token = None

        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if isinstance(value, str):
                    setattr(self, key, value)
                else:
                    raise TypeError(f"Special token {key} has to be str but got: {type(value)}")

        self.model_max_length = kwargs.pop("model_max_length", None)

        self.clean_up_tokenization_spaces = kwargs.pop("clean_up_tokenization_spaces", False)

    @property
    def bos_token_id(self) -> int:
        if self.bos_token is None:
            raise ValueError("Special token bos_token is not defined")
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        if self.eos_token is None:
            raise ValueError("Special token eos_token is not defined")
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self) -> int:
        if self.unk_token is None:
            raise ValueError("Special token unk_token is not defined")
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def pad_token_id(self) -> int:
        if self.pad_token is None:
            raise ValueError("Special token pad_token is not defined")
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def special_tokens_map(self) -> Dict[str, str]:
        """
        `Dict[str, str]`: A dictionary mapping special token class attributes (`bos_token`, `unk_token`, etc.)
        to their values (`'<bos>'`, `'<unk>'`, etc.).
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: A list of the unique special tokens (`'<bos>'`, `'<unk>'`, ..., etc.).
        """
        return list(self.special_tokens_map.values())

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<bos>'`, `'<unk>'`, etc.) mapped to class attributes.
        """
        return self.convert_tokens_to_ids(self.all_special_tokens)

    @overload
    def tokenize(self, texts: str) -> List[str]: ...

    @overload
    def tokenize(self, texts: List[str]) -> List[List[str]]: ...

    def tokenize(self, texts: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        raise NotImplementedError()

    def encode(self, texts: str) -> List[int]:
        raise NotImplementedError()

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        raise NotImplementedError()

    def decode(
        self, ids: List[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = None
    ) -> str:
        raise NotImplementedError()

    def batch_decode(
        self, ids: List[List[int]], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = None
    ) -> List[str]:
        raise NotImplementedError()

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]: ...

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        raise NotImplementedError()

    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = False) -> str: ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: bool = False) -> List[str]: ...

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        raise NotImplementedError()

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        raise NotImplementedError()

    @staticmethod
    def clean_up_tokenization(text: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            text (`str`): The text to clean up.

        Returns:
            `str`: The cleaned-up string.
        """
        text = (
            text.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return text
