from collections import Counter
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset


with open("E:/Projects/Seq2Seq/data/corpus/MergeRules.txt", "r") as f:
    merge_rules = {tuple(line.split()): idx for idx, line in enumerate(f)}


vocab = []
with open("E:/Projects/Seq2Seq/data/corpus/Vocab.txt", "r", encoding="utf-8") as f:
    for tok in f:
        vocab.append(tok.strip())

token2id = {v: k for k, v in enumerate(vocab)}
id2token = {k: v for k, v in enumerate(vocab)}


def tok_id(pieces):

    ids = []
    for p in pieces:

        if p in token2id:
            ids.append(token2id[p])
        else:
            ids.append(token2id["unk"])
    return ids

def bpe_encode(word: str, merge_rules: dict):

    seq = list(word) + ["</w>"]

    while True:

        pairs = {(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}

        required = min(pairs, key=lambda p: merge_rules.get(p, float("inf")))
        if required not in merge_rules:
            break

        first, second = required
        new_seq, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == first and seq[i + 1] == second:
                new_seq.append(first + second)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        seq = new_seq
    return seq


# bpe_decode = bpe_encode


def detokenise(ids):
    pieces = [id2token[i] for i in ids]
    words, buff = [], ""
    for p in pieces:
        if p == "</w>":
            if buff:
                words.append(buff)
                buff = ""
        elif p in ("<bos>", "<eos>"):
            continue
        elif p == "unk":
            buff += "[UNK]"
        else:
            buff += p
    if buff:
        words.append(buff)
    return " ".join(words)


train_ids_stream = []
with open("E:/Projects/Seq2Seq/data/corpus/Final_Train_Text.txt", "r") as f:
    for line in f:
        train_ids_stream.append(token2id["<bos>"])
        for word in line.split():
            pieces = bpe_encode(word.lower(), merge_rules)
            train_ids_stream.extend(tok_id(pieces))
        train_ids_stream.append(token2id["<eos>"])

train_ids_tensor = torch.tensor(train_ids_stream, dtype=torch.long)
torch.save(train_ids_tensor, "E:/Projects/Seq2Seq/data/train_texttokens.pt")
# print('train tokens saved')


val_ids_stream = []
with open("E:/Projects/Seq2Seq/data/corpus/Final_Val_Text.txt", "r") as f:
    for line in f:
        val_ids_stream.append(token2id["<bos>"])
        for word in line.split():
            pieces = bpe_encode(word.lower(), merge_rules)
            val_ids_stream.extend(tok_id(pieces))
        val_ids_stream.append(token2id["<eos>"])

val_ids_tensor = torch.tensor(val_ids_stream, dtype=torch.long)
torch.save(val_ids_tensor, "E:/Projects/Seq2Seq/data/val_texttokens.pt")
# print('val tokens saved')

# # Train tokens
# train_ids = torch.load("E:/Projects/Seq2Seq/data/train_texttokens.pt", map_location="cpu")
# train_ids_stream       = train_ids.tolist() 


# # validation set tokens 
# val_ids = torch.load("E:/Projects/Seq2Seq/data/val_texttokens.pt", map_location="cpu")
# val_ids_stream       = val_ids.tolist() 



train_inputs, train_targets = [], []


for i in range(0, len(train_ids_stream) - 257, 256):
    window = train_ids_stream[i : i + 257]
    if window.count(token2id["<pad>"]) / 257 > 0.2:
        continue
    train_inputs.append(window[:-1])
    train_targets.append(window[1:])

val_inputs, val_targets = [], []

for i in range(0, len(val_ids_stream) - 257, 256):
    window = val_ids_stream[i : i + 257]
    if window.count(token2id["<pad>"]) / 257 > 0.2:
        continue
    val_inputs.append(window[:-1])
    val_targets.append(window[1:])

inputs_tensor_train  = torch.tensor(train_inputs,  dtype=torch.long)
targets_tensor_train = torch.tensor(train_targets, dtype=torch.long)

inputs_tensor_val  = torch.tensor(val_inputs,  dtype=torch.long)
targets_tensor_val = torch.tensor(val_targets, dtype=torch.long)

train_data   = TensorDataset(inputs_tensor_train, targets_tensor_train)
val_data = TensorDataset(inputs_tensor_val, targets_tensor_val)


train_dataset = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=4,pin_memory=True, persistent_workers=True)
val_dataset = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=4,pin_memory=True, persistent_workers=True)


def encode_prompt(prompt: str):
    ids = [token2id["<bos>"]]
    for word in prompt.split():
        ids.extend(tok_id(bpe_encode(word.lower(), merge_rules)))
    return torch.tensor(ids, dtype=torch.long)
