import re

import contractions
import nltk
import torch

TAG_SCHEMAS = {
    "OT": [],
    "BIO": ["O", "EQ", "B-POS", "I-POS", "B-NEG", "I-NEG", "B-NEU", "I-NEU"],
    "BIEOS": [
        "O",
        "EQ",
        "B-POS",
        "I-POS",
        "E-POS",
        "S-POS",
        "B-NEG",
        "I-NEG",
        "E-NEG",
        "S-NEG",
        "B-NEU",
        "I-NEU",
        "E-NEU",
        "S-NEU",
    ],
}


def processing_text(text: str, lower_case: bool = True) -> str:
    """
    Xử lý văn bản: viết thường, mở rộng từ viết tắt, và loại bỏ ký tự không hợp lệ.
    """
    text = text.lower() if lower_case else text
    text = contractions.fix(text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens)


def processing_and_tokenize(
    text: str,
    tokenizer,
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
    sequence_a_segment_id: int = 0,
    cls_token_segment_id: int = 1,
    mask_padding_with_zero: bool = True,
    cls_token_at_end: bool = False,
) -> dict:
    """
    Xử lý và mã hóa văn bản thành các token kèm thông tin định danh.
    """
    processed_text = processing_text(text)
    words = processed_text.split()
    tokens, token_orders = [], []
    order = 0

    for word in words:
        subwords = tokenizer.tokenize(word)
        tokens.extend(subwords)
        token_orders.extend([order] * len(subwords))
        order += 1

    tokens.append(sep_token)
    token_orders.append(None)
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens.append(cls_token)
        segment_ids.append(cls_token_segment_id)
        token_orders.append(None)
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids
        token_orders = [None] + token_orders

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    return {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "token_orders": token_orders,
    }


def convert_ids_to_text(
    sample: dict, predicted_labels: list, token_orders: list, tokenizer
) -> tuple:
    """
    Chuyển đổi token IDs và nhãn dự đoán thành các từ và nhãn tương ứng.
    """
    converted_tokens, aligned_labels = [], []
    input_ids = sample["input_ids"]

    cur_order, cur_ids, cur_labels = 0, [], []

    for order, ids, label in zip(token_orders, input_ids, predicted_labels):
        if order is None:
            continue
        if order == cur_order:
            cur_ids.append(ids)
            cur_labels.append(label)
        else:
            converted_tokens.append(
                tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(cur_ids)
                )
            )
            aligned_labels.append(cur_labels[0])
            cur_ids, cur_labels = [ids], [label]
            cur_order = order

    if cur_ids:
        converted_tokens.append(
            tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(cur_ids))
        )
        aligned_labels.append(cur_labels[0])

    return converted_tokens, aligned_labels


def predict(
    text: str, tokenizer, model, device, label_list: list = TAG_SCHEMAS["BIEOS"]
) -> list:
    """
    Thực hiện suy luận trên văn bản đầu vào để gán nhãn.
    """
    sample = processing_and_tokenize(text, tokenizer)
    label_list_without_eq = [label for label in label_list if label != "EQ"]
    label_map = {i: label for i, label in enumerate(label_list_without_eq)}
    token_orders = sample["token_orders"]

    inputs = {
        "input_ids": torch.tensor(sample["input_ids"]).unsqueeze(0).to(device),
        "attention_mask": torch.tensor(sample["input_mask"]).unsqueeze(0).to(device),
        "token_type_ids": torch.tensor(sample["segment_ids"]).unsqueeze(0).to(device),
    }

    logits = model(**inputs)[0]
    predicted_labels = torch.argmax(logits, dim=2).squeeze(0).tolist()
    predicted_labels = [label_map[label] for label in predicted_labels]

    converted_tokens, aligned_labels = convert_ids_to_text(
        sample, predicted_labels, token_orders, tokenizer
    )

    return list(zip(converted_tokens, aligned_labels))
