from importlib.metadata import version

print("파이토치 버전:", version("torch"))
print("tiktoken 버전:", version("tiktoken"))

import os
import urllib.request

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# print("총 문자 개수:", len(raw_text))
# print(raw_text[:99])

# import re

# preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(preprocessed[:30])

# print(len(preprocessed))

# all_words = sorted(set(preprocessed))
# vocab_size = len(all_words)

# print(vocab_size)

# vocab = {token:integer for integer,token in enumerate(all_words)}

# # for i, item in enumerate(vocab.items()):
# #     print(item)
# #     if i >= 50:
# #         break

# # class SimpleTokenizerV1:
# #     def __init__(self, vocab):
# #         self.str_to_int = vocab
# #         self.int_to_str = {i:s for s,i in vocab.items()}

# #     def encode(self, text):
# #         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text) # 'hello,. world'

# #         preprocessed = [
# #             item.strip() for item in preprocessed if item.strip()
# #         ]
# #         ids = [self.str_to_int[s] for s in preprocessed]
# #         return ids

# #     def decode(self, ids):
# #         text = " ".join([self.int_to_str[i] for i in ids])
# #         # 구둣점 문자 앞의 공백을 삭제합니다.
# #         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
# #         return text

# # tokenizer = SimpleTokenizerV1(vocab)

# # text = """"It's the last he painted, you know,"
# #            Mrs. Gisburn said with pardonable pride."""
# # ids = tokenizer.encode(text)
# # print(ids)

# # print(tokenizer.decode(ids))

# # all_tokens = sorted(list(set(preprocessed)))
# # all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# # vocab = {token:integer for integer,token in enumerate(all_tokens)}

# # print(len(vocab.items()))

# # class SimpleTokenizerV2:
# #     def __init__(self, vocab):
# #         self.str_to_int = vocab
# #         self.int_to_str = { i:s for s,i in vocab.items()}

# #     def encode(self, text):
# #         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# #         preprocessed = [item.strip() for item in preprocessed if item.strip()]
# #         preprocessed = [
# #             item if item in self.str_to_int
# #             else "<|unk|>" for item in preprocessed
# #         ]

# #         ids = [self.str_to_int[s] for s in preprocessed]
# #         return ids

# #     def decode(self, ids):
# #         text = " ".join([self.int_to_str[i] for i in ids])
# #         # 구둣점 문자 앞의 공백을 삭제합니다.
# #         text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
# #         return text
    

# # tokenizer = SimpleTokenizerV2(vocab)
# # text1 = "Hello, do you like tea?"
# # text2 = "In the sunlit terraces of the palace."

# # text = " <|endoftext|> ".join((text1, text2))

# # print(text)

# # print(tokenizer.encode(text))

# # print(tokenizer.decode(tokenizer.encode(text))) 

import tiktoken

# tokenizer = tiktoken.get_encoding("gpt2")

# text = (
#     "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
#      "of someunknownPlace."
# )

# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# print(integers)

# string = tokenizer.decode(integers)
# print(string)

# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))

# enc_sample = enc_text[50:]

# context_size = 4

# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]

# print(f"x: {x}")
# print(f"y:      {y}")

# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]

#     print(context, "---->", desired)

# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]

#     print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))



import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 전체 텍스트를 토큰화합니다.
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "토큰화된 입력의 개수는 적어도 max_length+1과 같아야 합니다."

        # 슬라이딩 윈도를 사용해 책을 max_length 길이의 중첩된 시퀀스로 나눕니다.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # 토크나이저를 초기화합니다.
    tokenizer = tiktoken.get_encoding("gpt2")

    # 데이터셋을 만듭니다.
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 데이터 로더를 만듭니다.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# dataloader = create_dataloader_v1(
#     raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
# )

# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)

# second_batch = next(data_iter)
# print(second_batch)



# dataloader = create_dataloader_v1(
#     raw_text, batch_size=4, max_length=8, stride=1, shuffle=False
# )

# data_iter = iter(dataloader)
# inputs, targets = next(data_iter)
# print("inputs:", inputs)
# print("targets:", targets)

# import torch
# input_ids = torch.tensor([2, 3, 5, 1])
# print(input_ids.shape)
# vocab_size = 6
# output_dim = 3

# torch.manual_seed(123)
# embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# print(embedding_layer.weight)

# print(embedding_layer(input_ids))

import torch
from torch.utils.data import Dataset, DataLoader
vocab_size = 50257
output_dim = 256

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # 토크나이저를 초기화합니다.
    tokenizer = tiktoken.get_encoding("gpt2")

    # 데이터셋을 만듭니다.
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 데이터 로더를 만듭니다.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("토큰 ID:\n", inputs)
print("\n입력 크기:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# 임베딩 벡터의 값을 확인합니다.
print(token_embeddings)