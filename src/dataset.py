import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import BertTokenizer
import pandas as pd

import random


def get_max_input_len(arguments, tokenizer):
    max_len = 0
    for arg in arguments:
        input_ids = tokenizer.encode(arg, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    return max_len


def load_data(tokenizer, subset, data, path, load_args=False):
    df = pd.read_csv(path)

    if load_args:
        arguments = list(df["argument"])
        labels = list(df['argument_quality_ibm'])
    else:
        df = df[df["set"] == subset]
        arguments = list(df["argument"])
        labels = list(df[data])

    max_len = get_max_input_len(arguments, tokenizer)

    input_ids = []
    attention_masks = []
    for arg, topic in zip(arguments, list(df["topic"])):
        encoded_dict = tokenizer.encode_plus(
                            arg, topic, add_special_tokens=True,
                            max_length=max_len + 10, padding="max_length",
                            truncation=True, return_attention_mask=True,
                            return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    dataset = TensorDataset(input_ids, attention_masks, torch.tensor(labels))

    return dataset, arguments


def get_dataloader(data, path, batch_size, num_subset=0):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_data, _ = load_data(tokenizer, "train", data, path)
    val_data, _ = load_data(tokenizer, "dev", data, path)
    test_data, test_args = load_data(tokenizer, "test", data, path)

    if num_subset > 0:
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        train_data_subset = Subset(train_data, indices[:num_subset])
        train_dl = DataLoader(train_data_subset, shuffle=True, batch_size=batch_size)
    else:
        train_dl = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    val_dl = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    test_dl = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    return tokenizer, train_dl, val_dl, test_dl, test_args
