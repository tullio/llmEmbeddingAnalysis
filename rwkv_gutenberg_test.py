import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from scipy import spatial
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity as cos_sim

from transformers import PreTrainedTokenizerFast

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

model = RWKV(model='RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth', strategy='cpu fp32')
#print(model)

tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')
pipeline = PIPELINE(model, "20B_tokenizer.json") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV


def encode(text):
    chars = sorted(list(set(text)))
    char_size = len(chars)
    print("学習データで使っている文字数　：　", char_size)

    char2int = { ch : i for i, ch in enumerate(chars) }
    int2char = { i : ch for i, ch in enumerate(chars) }
    #encode = lambda a: [char2int[b] for b in a ]
    encode = lambda a: [char2int[b] for b in a ]
#    print("chars:", chars[100])
#    print("enumerate(chars):", list(enumerate(chars))[:100])


#    print("encode(chars):", encode(chars)[:100])
#    print("encode(text):", encode(text)[:100])
    print("encode(chars):", encode(chars)[:10])
    print("encode(text):", encode(text)[:10])
    return torch.tensor(encode(text), dtype=torch.long)

def rwkv_embeddings(text):
    encoded_words = encode(text)
    out, state = model.forward(encoded_words, None)
    hidden_states =out.detach().cpu()
    #print(out.detach().cpu().numpy())                   # get logits
    #print(out.detach().cpu().numpy().shape)                   # get logits
    print("hidden shape=", hidden_states.shape)
    return hidden_states

data_top_dir = "/home/tetsu.sato/RWKV-LM/data"
data_subdirs = ["carroll", "einstein", "lovecraft"]

for subdir in data_subdirs:
    path = os.path.join(data_top_dir, subdir)
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        with open(file_path, "r") as f:
            text = f.read()
            embeddign = rwkv_embeddings(text)
            print(file)
            print(embedding[:30])
            
