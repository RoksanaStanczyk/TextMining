from cleaning import cleaning
from cleaning import prepare_data
from cleaning import stopword_rem


def text_tokenizer(text: str) -> list:
    text_tokenizer_list = []
    clean = cleaning(text)
    stemm = prepare_data(clean)
    s_w = stopword_rem(stemm)
    for i in s_w:
        if len(i) > 3:
            text_tokenizer_list.append(i)
    return text_tokenizer_list
