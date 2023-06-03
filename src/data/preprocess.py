# MIT License
# Copyright (c) 2022 Rohit Singh and Yevhenii Maslov
# link: https://github.com/rohitsingh02/kaggle-feedback-english-language-learning-1st-place-solution
# Modified by Majidov Ikhtiyor
from typing import Tuple
import codecs
import re
from text_unidecode import unidecode
from tqdm import tqdm
\
def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text

def preprocess_text(text):
    text = resolve_encodings_and_normalize(text)
    return text

def get_max_len_from_df(df, tokenizer, n_special_tokens=3):
    lengths = []
    tk0 = tqdm(df['text'].fillna("").values, total=len(df))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    max_length = max(lengths) + n_special_tokens
    return max_length
