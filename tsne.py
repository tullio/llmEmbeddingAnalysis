from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import torch
from source_file_iterator import SourceFileIterator
from cache import Cache
import logging
from logging import config
from embeddings_base import embeddings_base
import sys
import time
import pickle
import os
from tqdm import tqdm
from rwkv_runner import rwkv
config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)
model_name = 'RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth'
tokenizer_name = '20B_tokenizer.json'

class tsne(embeddings_base):


    # 使わない引数も互換性のために残す．使わないのはNoneで呼んでね
    def __init__(self, model_filename, tokenizer_filename, model_load = True):
        super().__init__(model_filename, tokenizer_filename, model_load)

    def getTsneEmbeddings(self, file, numTokens):
        """
        file: io.TextIOWrapper
        """
        start = time.time()
        logger.debug(f"file={file}")
        text = file.read()
        file.seek(0)
        logger.debug(f"raw text={text[0:200]}")
        text = self.removeGutenbergComments(text)
        logger.debug(f"ETLed text={text[0:200]}")
        enc = self.encoding(text)
        tokens = enc.ids[:numTokens]
        logger.info(f"total tokens = {len(enc.ids)}")
        logger.info(f"splitted tokens = {len(tokens)}")        
        logger.debug(f"tokens[0:30]={tokens[0:30]}")
        tsne = TSNE(n_components=2, random_state = 0, perplexity = 1, n_iter = 1000)
        embeddings = tsne.fit_transform(np.array(tokens).reshape(1, -1))
        val = embeddings
        logger.debug(f"tsne embedding shape={val.shape}")
        end = time.time()
        logger.info(f"elapsed = {end - start}")
        return val


if __name__ == "__main__":
    args = sys.argv
    t = tsne(None, tokenizer_name, model_load = False)
    r = rwkv(model_name, tokenizer_name, model_load = False)
    iter = SourceFileIterator(t.data_top_dir, t.data_subdirs, t.numTokensList)
    #filename="rwkvEmbs.pickle"
    rwkv_list = {}
    #pool = multiprocessing.Pool(8)
    args_list = []
    cache_rebuild = False
    #embFunc = r.getRwkvEmbeddings
    embFunc = r.getHeadPersistenceDiagramEmbeddings
    filename=f"f{embFunc.__name__}-embs.pickle"
    if not os.path.exists(filename):
        for out in tqdm(iter):
            print(out)
            indexed_file = out[0]
            file_index = indexed_file[0]
            file = indexed_file[1]            
            indexed_numTokens = out[1]
            #print("indexed_numTokens=", indexed_numTokens)
            numTokens_index = indexed_numTokens[0]
            numTokens = indexed_numTokens[1]
            with open(file, "r", encoding="utf-8") as f:
                #print("file=", file)
                #embeddings = r.getRwkvEmbeddings(f, numTokens)
                #embeddings = r.getHeadPersistenceDiagramEmbeddings(f, numTokens)
                embeddings = embFunc(f, numTokens)
                #args_list.append((f, numTokens, cache_rebuild))
                embeddings = torch.tensor(embeddings) # (topN*2)
                print("embed shape=", embeddings.shape)
                key = f"{numTokens}"
                if key not in rwkv_list:
                    rwkv_list[key] = []
                rwkv_list[key].append((file, embeddings))
            #args_list.append((out, cache_rebuild))
        with open(filename, mode="wb") as f:
            pickle.dump(rwkv_list, f)

    with open(filename, mode="rb") as f:
        rwkv_list = pickle.load(f)

    for numTokens in rwkv_list.keys():
        file_list = [item[0] for item in rwkv_list[numTokens]]
        print(file_list)
        emb_list = [item[1] for item in rwkv_list[numTokens]]
        emb = torch.stack(emb_list)
        logger.info(f"{numTokens}:{emb}:{emb.shape}(raw)")
        tsne = TSNE(n_components=2, random_state = 0, perplexity = 10, n_iter = 1000)
        #embeddings = tsne.fit_transform(np.array(emb.cpu().reshape(-1, 2)))
        embeddings = tsne.fit_transform(np.array(emb.cpu()))
        logger.info(f"{numTokens}:{embeddings}:{embeddings.shape}(tsne)")
        tsne_filename = f"tsne-{embFunc.__name__}-{numTokens}.pickle"
        with open(tsne_filename, mode="wb") as f:
            pickle.dump(embeddings, f)
        names_filename = f"tsne-{numTokens}-names.pickle"
        with open(names_filename, mode="wb") as f:
            pickle.dump(file_list, f)
