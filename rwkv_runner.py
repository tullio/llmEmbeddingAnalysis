import time
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
#import multiprocessing
#import multiprocess
import torch.multiprocessing as multiprocessing
import pickle

from concurrent.futures import ProcessPoolExecutor

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries


from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from scipy import spatial
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity as cos_sim
import textwrap
import math
from params import params
import hashlib
import re

import sys
#sys.path.append("../ChatRWKV")
from rwkv_tokenizer import rwkv_tokenizer

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

from sklearn.manifold import TSNE
from scipy import spatial

import homcloud.interface as hc 

# data wrangling
import numpy as np
import pandas as pd
from pathlib import Path
#from IPython.display import YouTubeVideo
#from fastprogress import progress_bar

# hepml
#from hepml.core import make_gravitational_waves, download_dataset

# tda magic
from gtda.homology import VietorisRipsPersistence, CubicalPersistence
from gtda.diagrams import PersistenceEntropy, Scaler
from gtda.plotting import plot_heatmap, plot_point_cloud, plot_diagram
from gtda.pipeline import Pipeline
from gtda.time_series import TakensEmbedding

# ml tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
#from sklearn.metrics import plot_roc_curve, roc_auc_score, accuracy_score

# dataviz
import matplotlib.pyplot as plt
#import seaborn as sns

#sns.set(color_codes=True)
#sns.set_palette(sns.color_palette("muted"))

from embeddings_base import embeddings_base

import logging
from logging import config

from source_file_iterator import SourceFileIterator

config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)

#logger.basicConfig(encoding='utf-8', level=logging.DEBUG)
#logging.basicConfig(filename = "hoge.log", encoding='utf-8',level = logging.INFO,
#logger.basicConfig(encoding='utf-8',level = logging.INFO,
#                       format='[%(asctime)s] %(module)s.%(funcName)s %(levelname)s -> %(message)s')
#logger.setLevel(logging.INFO)
#formatter = logging.Formatter('[%(asctime)s] %(module)s.%(funcName)s %(levelname)s -> %(message)s')
#ch = logging.StreamHandler()
#ch.setFormatter(formatter)
#logger.addHandler(ch)


#import shelve
import diskcache


CharRWKV_HOME = "/research/ChatRWKV"
model_name = 'RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth'
tokenizer_name = '20B_tokenizer.json'
multiprocessing.set_start_method("spawn", force=True)

def getEmbFuncForMP(r, getEmbFuncName, list1, list2):
    #print("getEmbFUnc=", getEmbFuncName)
    #print("list1=", list1)
    #print("list2=", list2)
    logger.debug(f"embedding={getEmbFuncName}")
    if getEmbFuncName == "BottleneckSim":
        return r.BottleneckSim(list1, list2)


class rwkv(embeddings_base):
    
    def __init__(self, model_filename, tokenizer_filename, model_load = True):
        logger.info(f"initializing rwkv")
        super().__init__(model_filename, tokenizer_filename, model_load)
        
        self.model_filename = model_filename
        if model_load:
            self.model = RWKV(model=model_filename, strategy='cuda fp16i8')
            self.pipeline = PIPELINE(self.model, tokenizer_filename) # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV

        #self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_filename)
        #self.tokenizer = Tokenizer.from_file(tokenizer_filename)
        self.tokenizer = Tokenizer.from_file(tokenizer_filename)
        print(self.tokenizer.__class__)
        #self.tokenizer = rwkv_tokenizer("rwkv_vocab_v20230424.txt")
        self.AVOID_REPEAT_TOKENS = []
        self.start = time.time()


        print("tokenizer class=", self.tokenizer.__class__)
        if self.tokenizer.__class__ == "tokenizers.Tokenizer":
            self.AVOID_REPEAT = '，。：？！'
            for i in self.AVOID_REPEAT:
                dd = self.tokenizer.encode(i).ids
                assert len(dd) == 1
                self.AVOID_REPEAT_TOKENS += dd
        elif self.tokenizer.__class__ == 'rwkv_tokenizer.rwkv_tokenizer':
            None
            
        self.CHUNK_LEN: int = 8192*4
        """Batch size for prompt processing."""

        #self.key_prefix = self.__class__.__name__ + '_' + str(id(self)) + '_' + model_filename
        self.key_prefix = self.__class__.__name__ + \
            ':' + model_filename + \
            ':' + self.tokenizer.__class__.__name__
        print("key_prefix=", self.key_prefix)
        #filename = "rwkv_gutenberg.db"
        filename = params.cache_filename
        logger.info(f"cache_filename={params.cache_filename}")
        #self.db = shelve.open(filename)
        self.db = diskcache.Cache(filename)

        # Dataset作成用パラメータ
        self.datasetParameters = [
            [self.getRwkvEmbeddings, self.CosSim],
            [self.getRwkvEmbeddings, self.JFIP],
            [self.getHeadPersistenceDiagramEmbeddings,
             self.CosSim],
            [self.getHeadPersistenceDiagramEmbeddings,
             self.JFIP],
            [self.getHeadPersistenceDiagramEmbeddings,
             self.BottleneckSim]
            ]


    def getCacheKey(self, keyName,
                  file = None,
                  embFunc = None,
                  simFunc = None,
                  numTokens = None,
                  postfunc = None,
                  list1 = None,
                  list2 = None,
                  comment = None):
                    
        """
        keyName: 識別するための名前．基本はrequired
        targetFile: ドキュメントのファイル名．simFuncがNoneのときはrequired
        numTokens: ドキュメント先頭からのトークン数．postfuncがNoneならrequired
        getEmbFunc: 埋め込みベクトル計算関数．postfuncがNoneならrequired
        simFunc: 類似度計算関数

        postfunc: simFuncのための前処理関数
        hash1: simFuncに渡す引数のハッシュ --> hashはkey作成のためにしか使われていないからlist1でいい
        hash2: simFuncに渡す引数のハッシュ

        """
        key = ""
        if keyName == "emb" and embFunc == self.getHeadPersistenceDiagramEmbeddings:
            raise ValueError("use getPersistenceDiagramEmbeddings")

        if keyName == "dis" and simFunc == self.BottleneckSim:
            raise ValueError("use Bottleneck")

        if keyName == "emb":
            key += f"{keyName}"
            key += f":file={file.name}"
            key += f":embFunc={embFunc.__name__}"
            key += f":tokens={numTokens}"
        elif keyName == "embmat":
            key += f"{keyName}"            
            key += f":embFunc={embFunc.__name__}"
            key += f":simFunc={simFunc.__name__}"
            key += f":tokens={numTokens}"
        elif keyName == "simmat":
            key += f"{keyName}"
            key += f":embFunc={embFunc.__name__}"
            key += f":simFunc={simFunc.__name__}"
            key += f":tokens={numTokens}"
        elif keyName == "dis":
            key += f"{keyName}"
            key += f":postfunc={postfunc.__name__}"
            key += f":simFunc={simFunc.__name__}"
            hash1 = self.hash_algorithm(list1.tobytes()).hexdigest()
            key += f":hash1={hash1}"
            hash2 = self.hash_algorithm(list2.tobytes()).hexdigest()
            key += f":hash2={hash2}"
        if comment is not None:
            key += f":comment={comment}"
        return key
    
    def setDb(self, key, val):
        keyval = f"{self.key_prefix}:{key}"
        logger.info(f"key={keyval}")
        self.db[keyval] = val
        #self.db.sync()
    def getDb(self, key):
        keyval = f"{self.key_prefix}:{key}"
        return self.db.get(keyval)
    def run_rnn(self, tokens, newline_adj = 0):
        start = time.time()
        #global model_tokens, model_state
        model_tokens = []
        model_state = None
        states = []
        #logger.debug(f"input tokens={tokens}")
        tokens = [int(x) for x in tokens]
        #logger.debug(f"tokens={tokens}")
        model_tokens += tokens

        while len(tokens) > 0:
            out, model_state = self.model.forward(
                tokens[: self.CHUNK_LEN], model_state
            )
            tokens = tokens[self.CHUNK_LEN :]
            states.append(model_state)
        END_OF_LINE = 187
        out[END_OF_LINE] += newline_adj  # adjust \n probability
        
        # print(f'### model ###\n{tokens}\n[{tokenizer.decode(model_tokens)}]')

        #out[0] = -999999999  # disable <|endoftext|>
        #out[187] += newline_adj # adjust \n probability
        # if newline_adj > 0:
        #     out[15] += newline_adj / 2 # '.'
        if model_tokens[-1] in self.AVOID_REPEAT_TOKENS:
            out[model_tokens[-1]] = -999999999
        end = time.time()
        logger.info(f"run_rnn elapsed = {end - start}")
        #return out
        logger.debug(f"out={out}")
        #logger.debug(f"states={states}")
        #return states
        #logger.debug(f"len(model_states)={len(model_state)}")
        #for i, state in enumerate(model_state):
        #    logger.debug(f"model_states[{i}]={model_state[i]}")
        #    logger.debug(f"model_states[{i}].shape={model_state[i].shape}")
        #model_state = torch.cat(model_state)
        #logger.debug(f"model_states={model_state}")
        #return model_state
        final_layer_state = torch.cat([model_state[-5], model_state[-4],
                                      model_state[-3], model_state[-2],
                                       model_state[-1]])
        np_state=[i.detach().cpu().numpy() for i in model_state]
        np_state=np.concatenate(np_state, axis=0)
        logger.debug(f"final_layer_states={final_layer_state}")
        logger.debug(f"final_layer_states.shape={final_layer_state.shape}")
        logger.debug(f"np_state={np_state}")
        logger.debug(f"np_state.shape={np_state.shape}")
        return final_layer_state
        #return np_state
        
    def encoding(self, text):
        enc = self.tokenizer.encode(text)
        tokenIds = enc.ids
        tokens = enc.tokens
        logger.debug(f"token Ids[0:30]={tokenIds[0:30]}")
        logger.debug(f"tokens[0:30]={tokens[0:30]}")
        #return tokenIds
        return enc

    def removeGutenbergComments(self, text):
        start_line = "START OF THE PROJECT GUTENBERG EBOOK"
        end_line = "END OF THE PROJECT GUTENBERG EBOOK"
        lines = []

        # 特定の文字列を含む行の範囲を抽出
        should_extract = False
        for line in text.split('\n'):
            if should_extract:
                if end_line in line:
                    break
                lines.append(line.strip())
            elif start_line in line:
                should_extract = True

        if should_extract is False:
            lines = text.split('\n')
            
        return '\n'.join(lines)

    
    def getRwkvEmbeddings(self, file, numTokens):
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
        #embeddign = rwkv_embeddings(text)
        enc = self.encoding(text)
        tokens = enc.ids[:numTokens]
        logger.info(f"total tokens = {len(enc.ids)}")
        logger.info(f"splitted tokens = {len(tokens)}")        
        #logger.info(f"all tokens={enc.tokens[:numTokens]}")
        #logger.info(f"splitted all tokens={enc.tokens[:numTokens]}")
        logger.debug(f"tokens[0:30]={tokens[0:30]}")
        ## key = f"{file.name}:tokens={numTokens}:rwkvemb" # old
        #key = self.getCacheKey("rwkvemb", None, file, numTokens, None, None)
        #key = self.getCacheKey("rwkvemb", self.tokenizer, file, numTokens,
        #                       self.getRwkvEmbeddings, None)
        key = self.getCacheKey("emb", file = file,
                               numTokens = numTokens,
                               embFunc = self.getRwkvEmbeddings
                               )
        val = self.getDb(key)
        if self.enable_rwkvemb_cache:
            if val is None:
                logger.debug(f"getDB({key}) is None. Rebuild Cache")
                if hasattr(self, "model"):
                    embeddings = self.run_rnn(tokens)
                else:
                    logger.error("LLM model was not loaded. Cannot continue")
                    raise NameError("LLM model was not loaded. Cannot continue")
                logger.debug(f"writing cache... key={key}")
                self.setDb(key, embeddings)
                val = embeddings
                #print(embeddings[:30])
            else:
                logger.debug(f"getDB({key}) found the cache value")
        else:
            logger.debug(f"Cache access is disabled")
            embeddings = self.run_rnn(tokens)
            val = embeddings
        logger.debug(f"rwkv embedding shape={val.shape}")
        end = time.time()
        logger.info(f"elapsed = {end - start}")
        return val

    def listCache(self, keyName,
                  file = None, # file descriptor
                  embFunc = None, # function object
                  simFunc = None, # function object
                  numTokens = None, # int
                  postfunc = None, # function object
                  list1 = None, # numpy array
                  list2 = None, # numpy array
                  comment = None):
        """
        keyName: 識別するための名前．基本はrequired
            - "emb"
              emb:file={}:embFunc={}:tokens={}[:comment={}]
            - "embmat"
              embmat:embFunc={}:simFunc={}:tokens={}[:comment={}]
            - "simmat"
              simmat:embFunc={}:simFunc={}:tokens={}[:comment={}]
            - "dis"
              dis:postfunc={}:hash1={}:hash2={}[:comment={}]

        # tokenizer: 使ったトークナイザ
        # targetFile: ドキュメントのファイル名

        # おもに埋め込みベクトル，類似度行列で使う

        # 埋め込みベクトルではこう呼び出されている
        # key = self.getCacheKey("swemb", None, file, numTokens,
        #                       self.getSlidingWindowEmbeddings, None)

        # simMatのための埋め込みベクトルリストのキャッシュを新設（2023/08/18 12:44）

        # simMatではこう呼び出されている
        # key = self.getCacheKey("rwkvemb", None, None, numTokens, getEmbFunc, simFunc)
        numTokens: ドキュメント先頭からのトークン数
        getEmbFunc: 埋め込みベクトル計算関数
        simFunc: 類似度計算関数

        # Bottleneckではこう呼び出されている
        # key = f"postfunc={self.pdemb_postfunc.__name__}:simFunc={}:{hash1}:{hash2}"
        # おもにsimFunc/距離関数で使う
        postfunc: 距離関数のための前処理関数
        simFunc: 距離関数: 類似度関数と距離関数が同時に使われることはないので
        hash1: 距離関数に渡す引数のハッシュ --> hashはkey作成のためにしか使われていないからlist1でいい
        hash2: 距離関数に渡す引数のハッシュ

        # ----- 仕様
        Noneは.*に変換される
        """

        iter = self.db.iterkeys(reverse=False)
        substring = ""
        if keyName == "emb":
            substring += f"{keyName}"
            if file is not None:
                substring += f":file={file.name}"
            else:
                substring += f":[^:]*"
            if embFunc is not None:
                substring += f":embFunc={embFunc.__name__}"
            else:
                substring += f":[^:]*"
            if numTokens is not None:
                substring += f":tokens={numTokens}"
            else:
                substring += f":[^:]*"
        elif keyName == "embmat":
            substring += f"{keyName}"            
            if embFunc is not None:
                substring += f":embFunc={embFunc.__name__}"
            else:
                substring += f":[^:]*"
            if simFunc is not None:
                substring += f":simFunc={simFunc.__name__}"
            else:
                substring += f":[^:]*"
            if numTokens is not None:
                substring += f":tokens={numTokens}"
            else:
                substring += f":[^:]*"
        elif keyName == "simmat":
            substring += f"{keyName}"            
            if embFunc is not None:
                substring += f":embFunc={embFunc.__name__}"
            else:
                substring += f":[^:]*"
            if simFunc is not None:
                substring += f":simFunc={simFunc.__name__}"
            else:
                substring += f":[^:]*"
            if numTokens is not None:
                substring += f":tokens={numTokens}"
            else:
                substring += f":[^:]*"
        elif keyName == "dis":
            substring += f"{keyName}"            
            if postfunc is not None:
                substring += f":postfunc={postfunc.__name__}"
            else:
                substring += f":[^:]*"
            if simFunc is not None:
                substring += f":simFunc={simFunc.__name__}"
            else:
                substring += f":[^:]*"
                
            if list1 is not None:
                hash1 = self.hash_algorithm(list1.tobytes()).hexdigest()
                substring += f":hash1={hash2}"
            else:
                substring += f":[^:]*"
            if list2 is not None:
                hash2 = self.hash_algorithm(list2.tobytes()).hexdigest()
                substring += f":hash2={hash2}"
            else:
                substring += f":[^:]*"
        else:
            raise ValueError(f"Invalid tag:{keyName}")
        
        hit_keys = []
        for key in iter:
            #logger.debug(f"target key={substring}")
            #logger.debug(f"target record={key}")
            #print(re.search(substring, key))
            if re.search(substring, key):
                hit_keys.append(key)
        return hit_keys

    def deleteCache(self, keyName,
                  file = None, # file descriptor
                  embFunc = None, # function object
                  simFunc = None, # function object
                  numTokens = None, # int
                  postfunc = None, # function object
                  list1 = None, # numpy array
                  list2 = None, # numpy array
                  comment = None):
        """
        Noneは.*に変換される
        return: Counter indicating the number of deleted items
        """
        if keyName == "emb" and embFunc == self.getHeadPersistenceDiagramEmbeddings:
            raise ValueError("use getPersistenceDiagramEmbeddings")
        if keyName == "dis" and simFunc == self.BottleneckSim:
            raise ValueError("use Bottleneck")
        
        keys = self.listCache(keyName,
                              file = file,
                              embFunc = embFunc,
                              simFunc = simFunc,
                              numTokens = numTokens,
                              postfunc = postfunc,
                              list1 = list1,
                              list2 = list2,
                              comment = comment)
        count = 0
        logger.info(f"delete target keys[0:5]={keys[0:5]}")
        #input("OK?")
        for key in keys:
            logger.debug(f"deleting candidate key={key}")
            #input("OK?")
            if self.db.delete(key):
                count += 1
        logger.info(f"deleted {count} records")
        return count
    def getSlidingWindowEmbeddings(self, file, numTokens):
        start = time.time()
        # key = f"{file.name}:tokens={numTokens}:swemb" # old
        #key = self.getCacheKey("swemb", None, file, numTokens,
        #                       self.getSlidingWindowEmbeddings, None)
        key = self.getCacheKey("emb", file = file,
                               numTokens = numTokens,
                               embFunc = self.getSlidingWindowEmbeddings
                               )
        val = self.getDb(key)
        #logging.info(f"sliding window cache={val}")
        if self.enable_swemb_cache:
            if val is None:
                logger.debug(f"getDB({key}) is None. Rebuild Cache")
                rwkv_emb = self.getRwkvEmbeddings(file, numTokens)
                logger.debug(f"input rwkv emg shape={rwkv_emb.shape}")
                sw_embeddings = self.sw_embedder.fit_transform(rwkv_emb.reshape(1, -1).cpu())
                sw_embeddings = sw_embeddings[0, :, :]
                logger.debug(f"set key={key}")
                self.setDb(key, sw_embeddings)
                val = sw_embeddings
            else:
                logger.debug(f"SW embedding cache found")
                logger.info(f"getDB({key}) found the cache value")
        else:
            logger.info(f"Cache access is disabled")
            rwkv_emb = self.getRwkvEmbeddings(file, numTokens)
            logger.debug(f"input rwkv emg shape={rwkv_emb.shape}")
            if hasattr(self, "model"):
                if type(rwkv_emb).__name__ == "Tensor":
                    sw_embeddings = self.sw_embedder.fit_transform(rwkv_emb.reshape(1, -1).cpu())
                else:
                    sw_embeddings = self.sw_embedder.fit_transform(rwkv_emb.reshape(1, -1))
                sw_embeddings = sw_embeddings[0, :, :]
            else:
                logger.error("LLM model was not loaded. Cannot continue")
                raise NameError("LLM model was not loaded. Cannot continue")
            logger.debug(f"set key={key}")
            self.setDb(key, sw_embeddings)
            val = sw_embeddings

            
        logger.debug(f"sliding window embedding shape={val.shape}")
        end = time.time()
        logger.info(f"elapsed = {end - start}")
        return val
    def getHeadPersistenceDiagramEmbeddings(self, file, numTokens):
        """
        getPersistenceDiagramEmbeddingsで得た埋め込みの，
        """
        emb = self.getPersistenceDiagramEmbeddings(file, numTokens) # [2, n]
        lifeTime = emb[1, :] - emb[0, :]
        #print("emb=", emb)
        #print("lifeTime=", lifeTime)
        #print("zip=",list(zip(lifeTime, emb)))
        ##sorted_pairs = sorted(zip(lifeTime, emb), key=lambda pair: pair[0], reverse=True)
        ##topN_lifeTime, topN_emb = zip(*sorted_pairs[:self.topN])
        # Aの要素を降順にソートして上位5件のインデックスを取得
        indices = np.argsort(lifeTime)[::-1][:self.topN]

        # ソートされたAの上位5件の要素と対応するBの要素を取得
        sorted_lifeTime = lifeTime[indices]
        #print("sorted lifeTime=", sorted_lifeTime)
        sorted_emb = emb[:, indices]
        logger.debug(f"sorted embeddings={sorted_emb}")
        # x軸でソートしなおしてベクトルの順序（？）をもとに戻す
        indices = np.argsort(sorted_emb[0, :])
        sorted_emb = sorted_emb[:, indices]
        #sorted_emb = sorted_emb[:, indices]/1000
        #sorted_emb = self.normalize(sorted_emb)
        #sorted_emb = self.regularize(sorted_emb)
        #sorted_emb = self.pdemb_postfunc(sorted_emb)
        logger.debug(f"re-sorted embeddings={sorted_emb}")        
        # 最後に，[2, n]のベクトル群を1次元に変換する
        sorted_emb = sorted_emb.reshape(sorted_emb.shape[0]*sorted_emb.shape[1])
        #print("reshaped embeddings=", sorted_emb)
        #print("re-reshaped embeddings=", sorted_emb.reshape(2, -1))
        return sorted_emb
        

    # 1次元のPDの2次元ベクトル列をnumpyで返す
    def getPersistenceDiagramEmbeddings(self, file, numTokens):
        """
        PDの2次元ベクトル列をnumpyで返す
        """
        start = time.time()
        # key = f"{file.name}:tokens={numTokens}:pdemb" # old
        #key = self.getCacheKey("pdemb", None, file, numTokens, None, None)
        #key = self.getCacheKey("pdemb", self.tokenizer, file, numTokens,
        #                       self.getPersistenceDiagramEmbeddings, None)
        key = self.getCacheKey("emb", file = file,
                               numTokens = numTokens,
                               embFunc = self.getPersistenceDiagramEmbeddings
                               )
        val = self.getDb(key)
        #logger.info(f"sliding window cache={val}")

        ns = time.time_ns()
        id = ns - int(ns / 1000)*1000
        filename = f"pointcloud-{id}.pdgm"
        if self.enable_pdemb_cache:
            if val is None:
                logger.info(f"getDB({key}) is None. Rebuild Cache")
                sw_emb = self.getSlidingWindowEmbeddings(file, numTokens)
                logger.debug(f"input sw emg shape={sw_emb.shape}")
                pdlist = hc.PDList.from_alpha_filtration(sw_emb, 
                                                save_to=filename,
                                    save_boundary_map=True)
                #pdlist = hc.PDList(filename)
                if os.path.exists(filename):
                    os.remove(filename)

                pd1 = pdlist.dth_diagram(1)
                pd_embeddings = np.array(pd1.birth_death_times())

                logger.debug(f"set key={key}")
                self.setDb(key, pd_embeddings)

                val = pd_embeddings
            else:
                logger.info(f"PD embedding cache({key}) found")
        else:
            logger.info(f"Cache access is disabled")
            sw_emb = self.getSlidingWindowEmbeddings(file, numTokens)
            hc.PDList.from_alpha_filtration(sw_emb, 
                                            save_to=filename,
                                    save_boundary_map=True)
            pdlist = hc.PDList(filename)
            os.remove(filename)

            pd1 = pdlist.dth_diagram(1)
            pd_embeddings = np.array(pd1.birth_death_times())
            val = pd_embeddings            
        logger.debug(f"Persistence Diagram embedding shape={val.shape}")
        logger.debug(f"Persistence Diagram embedding[0:30]={val[0:30]}")
        end = time.time()
        logger.info(f"elapsed = {end - start}")
        return val

    def getPersistenceDiagramEmbeddings1d(self, file, numTokens):
        emb = self.getPersistenceDiagramEmbeddings(file, numTokens)
        logger.debug(f"emb shape={emb.shape}")
        emb1d = emb.reshape(1, -1).flatten()
        logger.debug(f"emb1d shape={emb1d.shape}")
        return emb1d

    def getEmbeddingsFromOut(self, out, cache_rebuild = False):
        indexed_file = out[0]
        file_index = indexed_file[0]
        file = indexed_file[1]            
        indexed_numTokens = out[1]
        #print("indexed_numTokens=", indexed_numTokens)
        numTokens_index = indexed_numTokens[0]
        numTokens = indexed_numTokens[1]
        with open(file, "r", encoding="utf-8") as f:
            self.getEmbeddings(f, numTokens, cache_rebuild)

    # キャッシュは全部ここで管理したい
    # 言語モデルのベクトル
    # Sliding Window埋め込み
    # パーシステンスホモロジーの2次元ベクトル
    # ここは，cache rebuildについては，フラグからclearに変える
    def getEmbeddings(self, file, numTokens, cache_rebuild = False):
        start = time.time()
        logger.info(f"Start getEmbeddings: file={file}, numTokens={numTokens}")

        self.enable_rwkvemb_cache = not cache_rebuild

        ### 言語モデルのベクトル
        rwkv_emb = self.getRwkvEmbeddings(file, numTokens)

        # 一度作り直したら，キャッシュを使う
        self.enable_rwkvemb_cache = True
        # こっちはこれから作り直す
        self.enable_swemb_cache = not cache_rebuild
        ### Sliding Window
        sw_emb = self.getSlidingWindowEmbeddings(file, numTokens)
        
        # 一度作り直したら，キャッシュを使う
        self.enable_swemb_cache = True
        # こっちはこれから作り直す
        self.enable_pdemb_cache = not cache_rebuild

        ### Persistence Diagram
        pd_emb = self.getPersistenceDiagramEmbeddings(file, numTokens)

        #self.db.sync()
        end = time.time()
        logger.info(f"End: file={file}, numTokens={numTokens},elapsed = {end - start}")
        return pd_emb

    def describeEmbeddings(self, emb):
        """
        埋め込みベクトルの統計量を出力する
        """
        print(f"shape={emb.shape}")
        print(f"head[0:10]={emb[0:10]}")
        desc = pd.DataFrame(emb.cpu().numpy()).describe()
        print(f"describe={desc}")
        print(f"mean={desc.loc['mean'].iloc[0]}")
            
    def CosSim(self, list1, list2):
        #print(type(list1).__name__)
        logger.debug(f"list1.shape={list1.shape}")
        logger.debug(f"list1={list1}, list2={list2}")        
        if type(list1).__name__ == "Tensor":
            sim = cos_sim(list1.reshape(1, -1), list2.reshape(1, -1)).item()
        elif type(list1).__name__ == "list" or type(list1).__name__ == "ndarray":
            sim = 1 - spatial.distance.cosine(list1, list2)
        return sim
    def FIP(self, f, g):
        if type(f).__name__ == "Tensor":
            fip = torch.dot(f.view(-1), g.view(-1))
        elif type(f).__name__ == "list" or type(f).__name__ == "ndarray":
            fip = np.dot(f.ravel(), g.ravel())
        return fip
    def JFIP(self, f, g):
        #print(type(f).__name__)
        logger.debug(f"list1.shape={f.shape}")
        logger.debug(f"list1={f}, list2={g}")        
        if type(f).__name__ == "Tensor":
            f = torch.cov(f)
            g = torch.cov(g)
            jfip = 2*self.FIP(f, g)/(self.FIP(f, f)+self.FIP(g, g))

        elif type(f).__name__ == "list" or type(f).__name__ == "ndarray":

            f = np.cov(f)
            g = np.cov(g)
            jfip = 2*self.FIP(f, g)/(self.FIP(f, f)+self.FIP(g, g))
        return jfip

    # getPersistenceDiagramEmbeddingsがbirth_death_times()で取った2次元ベクトル列を
    # 返しちゃうので，そこからPDを再現したい
    # from_birth_deathでいいのか
    # getHeadPersistenceDiagramEmbeddingsは1次元に直しちゃう
    # これは統一するか→getEmbeddingsシリーズのIF仕様なので，戻り地は1次元で
    def Bottleneck(self, list1, list2):
        logger.debug(f"list1.shape={list1.shape}")
        logger.debug(f"list1={list1}, list2={list2}")        
        start = time.time()
        if type(list1).__name__ == "Tensor":
            logger.info("##############################################################")
            sim = cos_sim(list1.reshape(1, -1), list2.reshape(1, -1)).item()
        elif type(list1).__name__ == "list" or type(list1).__name__ == "ndarray":
            logger.info(f"type={type(list1).__name__}")
            logger.debug(f"list1={list1}")
            hash1 = self.hash_algorithm(list1.tobytes()).hexdigest()
            logger.debug(f"list1 hash={hash1}")
            logger.debug(f"list2={list2}")
            hash2 = self.hash_algorithm(list2.tobytes()).hexdigest()
            logger.debug(f"list2 hash={hash2}")
            #key = f"postfunc={self.pdemb_postfunc.__name__}:bottleneck:{hash1}:{hash2}"
            key = self.getCacheKey("dis",
                                   postfunc = self.pdemb_postfunc,
                                   simFunc = self.Bottleneck,
                                   list1 = list1,
                                   list2 = list2
                                   )

            val = self.getDb(key)
            if self.enable_bottleneck_cache:
                if val is None:
                    logger.debug(f"getDB({key}) is None. Rebuild Cache")
                    list1 = list1.reshape(2, -1)
                    list2 = list2.reshape(2, -1)
                    #list1 = self.normalize(list1)
                    #list2 = self.normalize(list2)
                    list1 = self.pdemb_postfunc(list1)
                    list2 = self.pdemb_postfunc(list2)
                    pd1 = hc.PD.from_birth_death(1, list1[0, :], list1[1, :])
                    pd2 = hc.PD.from_birth_death(1, list2[0, :], list2[1, :])
                    dis = hc.distance.bottleneck(pd1, pd2)
                    self.setDb(key, dis)
                    val = dis
                else:
                    logger.debug(f"getDB({key}) found the cache value")
            else:
                logger.debug(f"Cache access is disabled")
                list1 = list1.reshape(2, -1)
                list2 = list2.reshape(2, -1)
                #list1 = self.normalize(list1)
                #list2 = self.normalize(list2)
                list1 = self.pdemb_postfunc(list1)
                list2 = self.pdemb_postfunc(list2)
                pd1 = hc.PD.from_birth_death(1, list1[0, :], list1[1, :])
                pd2 = hc.PD.from_birth_death(1, list2[0, :], list2[1, :])
                dis = hc.distance.bottleneck(pd1, pd2)
                self.setDb(key, dis)
                val = dis
        end = time.time()
        logger.info(f"elapsed = {end - start}")
        return val

    def BottleneckSim(self, list1, list2):
        dis = self.Bottleneck(list1, list2)
        logger.info(f"dis={dis}")
        sim = 0.0

        logger.info(f"sigma={self.sigma}")
        kernel_func = lambda x: np.exp(-x**2 / (2 * self.sigma**2))
        sim = kernel_func(dis)
        #sim = 1.0 - dis

        logger.info(f"sim={sim}")
        return sim        


    def Wasserstein(self, list1, list2):
        start = time.time()
        if type(list1).__name__ == "Tensor":
            logger.info("##############################################################")
            sim = cos_sim(list1.reshape(1, -1), list2.reshape(1, -1)).item()
        elif type(list1).__name__ == "list" or type(list1).__name__ == "ndarray":
            logger.info(f"type={type(list1).__name__}")
            logger.debug(f"list1={list1}")
            hash1 = self.hash_algorithm(list1.tobytes()).hexdigest()
            logger.debug(f"list1 hash={hash1}")
            logger.debug(f"list2={list2}")
            hash2 = self.hash_algorithm(list2.tobytes()).hexdigest()
            logger.debug(f"list2 hash={hash2}")
            key = f"postfunc={self.pdemb_postfunc.__name__}:wasserstein:{hash1}:{hash2}"
            val = self.getDb(key)
            if self.enable_wasserstein_cache:
                if val is None:
                    logger.debug(f"getDB({key}) is None. Rebuild Cache")
                    list1 = list1.reshape(2, -1)
                    list2 = list2.reshape(2, -1)
                    #list1 = self.normalize(list1)
                    #list2 = self.normalize(list2)
                    list1 = self.pdemb_postfunc(list1)
                    list2 = self.pdemb_postfunc(list2)
                    pd1 = hc.PD.from_birth_death(1, list1[0, :], list1[1, :])
                    pd2 = hc.PD.from_birth_death(1, list2[0, :], list2[1, :])
                    dis = hc.distance.wasserstein(pd1, pd2)
                    self.setDb(key, dis)
                    val = dis
                else:
                    logger.debug(f"getDB({key}) found the cache value")
            else:
                logger.debug(f"Cache access is disabled")
                list1 = list1.reshape(2, -1)
                list2 = list2.reshape(2, -1)
                #list1 = self.normalize(list1)
                #list2 = self.normalize(list2)
                list1 = self.pdemb_postfunc(list1)
                list2 = self.pdemb_postfunc(list2)
                pd1 = hc.PD.from_birth_death(1, list1[0, :], list1[1, :])
                pd2 = hc.PD.from_birth_death(1, list2[0, :], list2[1, :])
                dis = hc.distance.wasserstein(pd1, pd2)
                self.setDb(key, dis)
                val = dis
        end = time.time()
        logger.info(f"elapsed = {end - start}")
        return val

    def WassersteinSim(self, list1, list2):
        dis = self.Wasserstein(list1, list2)
        logger.debug(f"dis={dis}")
        sim = 0.0

        logger.debug(f"sigma={self.sigma}")
        kernel_func = lambda x: np.exp(-x**2 / (2 * self.sigma**2))
        sim = kernel_func(dis)
        #sim = 1.0 - dis

        logger.debug(f"sim={sim}")
        return sim        

    def normalize(self, list):
        """
        input: [[x1, x2...], [y1, y2, ...]]
        """
        min_vals = np.array([np.min(list, axis=1)])
        max_vals = np.max(list, axis=1)
        normalized = (list - min_vals.T) / (max_vals - min_vals).T
        return normalized

    def regularize(self, list):
        """
        input: [[x1, x2...], [y1, y2, ...]]
        """
        mean_vals = np.array([np.mean(list, axis=1)])
        std_vals = np.array([np.std(list, axis=1)])
        regularized = (list - mean_vals.T) / std_vals.T
        return regularized

    def scaling(self, list):
        """
        input: [[x1, x2...], [y1, y2, ...]]
        """
        scaled = list / self.scaling_const
        scaled = scaled.astype(int)
        logger.debug(f"convert {list} to {scaled}")
        return scaled
    
    def identical(self, list):
        return list
    
    def simMatrixPlot(self, fig, ax, matrix):

        ax.invert_yaxis()
        #cax=ax.imshow(matrix, cmap="Paired", origin="lower")
        #cax=ax.imshow(matrix, cmap="viridis", origin="upper", vmin=0, vmax=1)
        cax=ax.imshow(matrix, cmap="viridis", origin="upper")
        cbar = fig.colorbar(cax)


    def all_simMatrixPlot(self):
        fig = plt.figure(figsize=(10, 25))
        fig.subplots_adjust(hspace=1.0, wspace=0.2)
        fig.suptitle(f"topN={self.topN}, postfunc={self.pdemb_postfunc},timedelay={self.embedding_time_delay_periodic}", fontsize=10)
        fig.tight_layout(rect=[0,0,1,0.96])
        #fig.tight_layout(rect=[0, 0, 2,0.96])
        #max_cols = 4 # 論文の図から
        max_cols = 2 # デバグのため大きく表示したい
        max_rows = len(self.data_top_dir) * len(self.numTokensList) // max_cols # 縦は定めなくてどんどん増えてもいいんだけど，subplotの仕様上仕方がない
        embFuncList = [self.getRwkvEmbeddings, self.getHeadPersistenceDiagramEmbeddings]
        simFuncList = [self.CosSim, self.JFIP, self.BottleneckSim]
        # self.numTokensList = [1024, 2048, 4096] # これはinitのを流用する
        # この組み合わせだけど，rwkvにbottleneckとかないので，
        # [rwkv, (cos, JFIP), [1024, 2048, 4096]) = 6,
        # [rwkv->PD, (cos, JFIP, bottleneck), (1024 2048, 4096)] -> 9で15?
        # 原稿を見ると15でビンゴ
        # じゃあそれでいったん実装するか
        seq = 1

        pool = multiprocessing.Pool(8)
        embFunc = self.getRwkvEmbeddings
        for simFunc in [self.CosSim, self.JFIP]:
            for numTokens in self.numTokensList:
                matrix = self.simMat(embFunc, simFunc, numTokens)
                column = seq % max_cols + 1
                row = seq // max_cols + 1
                title = f"raw:{simFunc.__name__}({numTokens})"
                ax = fig.add_subplot(max_rows, max_cols, seq)
                ax.set_title(title, fontsize = 8)
                self.simMatrixPlot(fig, ax, matrix)
                seq += 1

        
        embFunc = self.getHeadPersistenceDiagramEmbeddings
        for simFunc in [self.CosSim, self.JFIP, self.BottleneckSim]:
            for numTokens in self.numTokensList:
                matrix = self.simMat(embFunc, simFunc, numTokens)
                column = seq % max_cols + 1
                row = seq // max_cols + 1
                title = f"TDA:{simFunc.__name__}({numTokens})"
                ax = fig.add_subplot(max_rows, max_cols, seq)
                ax.set_title(title, fontsize = 8)
                self.simMatrixPlot(fig, ax, matrix)                
                seq += 1
        
    def all_simMatrixDescribe(self):

        embFuncList = [self.getRwkvEmbeddings, self.getHeadPersistenceDiagramEmbeddings]
        simFuncList = [self.CosSim, self.JFIP, self.BottleneckSim]
        # self.numTokensList = [1024, 2048, 4096] # これはinitのを流用する
        # この組み合わせだけど，rwkvにbottleneckとかないので，
        # [rwkv, (cos, JFIP), [1024, 2048, 4096]) = 6,
        # [rwkv->PD, (cos, JFIP, bottleneck), (1024 2048, 4096)] -> 9で15?
        # 原稿を見ると15でビンゴ
        # じゃあそれでいったん実装するか
        seq = 1
        embFunc = self.getRwkvEmbeddings
        for simFunc in [self.CosSim, self.JFIP]:
            for numTokens in self.numTokensList:
                matrix = self.simMat(embFunc, simFunc, numTokens)
                column = seq % max_cols + 1
                row = seq // max_cols + 1
                title = f"raw:{simFunc.__name__}({numTokens})"
                ax = fig.add_subplot(max_rows, max_cols, seq)
                ax.set_title(title, fontsize = 8)
                self.simMatrixPlot(fig, ax, matrix)
                seq += 1
        embFunc = self.getHeadPersistenceDiagramEmbeddings
        for simFunc in [self.CosSim, self.JFIP, self.BottleneckSim]:
            for numTokens in self.numTokensList:
                matrix = self.simMat(embFunc, simFunc, numTokens)
                column = seq % max_cols + 1
                row = seq // max_cols + 1
                title = f"TDA:{simFunc.__name__}({numTokens})"
                ax = fig.add_subplot(max_rows, max_cols, seq)
                ax.set_title(title, fontsize = 8)
                self.simMatrixPlot(fig, ax, matrix)                
                seq += 1
        
    def pd_plot(self, file, numTokens):
        fig = None
        ax = None
        ### Visualize
        if self.enable_pdemb_visualize is True:
            pd_emb = self.getPersistenceDiagramEmbeddings(file, numTokens)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(pd_emb[0], pd_emb[1])
            ax.set_title(f"file={file}, numTokens={numTokens}")
        return ax

    def count_source_files(self):
        iter = SourceFileIterator(self.data_top_dir, self.data_subdirs, self.numTokensList)
        out = next(iter, None)
        filename_set = set()
        while out:
            indexed_filename = out[0]
            filename = indexed_filename[1]
            filename_set.add(filename)
            out = next(iter, None)
        count = len(filename_set)
        return count
            
    def all_pd_plot(self):
        iter = SourceFileIterator(self.data_top_dir, self.data_subdirs, self.numTokensList)
        print("data_subdirs=", self.data_subdirs)
        print("numTokensList=", self.numTokensList)
        #max_rows = len(self.data_subdirs) * 2
        max_rows = self.count_source_files()
        max_cols = len(self.numTokensList)
        out = next(iter, None)
        count = 1
        fig = plt.figure(figsize=[10,100])
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle(f"topN={self.topN}, postfunc={self.pdemb_postfunc},timedelay={self.embedding_time_delay_periodic}", fontsize=10)
        while out:
            #logger.info(out)
            indexed_file = out[0]
            file_index = indexed_file[0]
            file = indexed_file[1]            
            indexed_numTokens = out[1]
            #print("indexed_numTokens=", indexed_numTokens)
            numTokens_index = indexed_numTokens[0]
            numTokens = indexed_numTokens[1]
            with open(file, "r", encoding="utf-8") as f:
                #print("file=", file)
                self.pd_subplot(fig, f, numTokens, max_rows, max_cols, count)
            count += 1
            out = next(iter, None)

    def pd_subplot(self, fig, file, numTokens, row, column, seq):
        logger.debug(f"row={row}, column={column}")
        ax = None
        ### Visualize
        if self.enable_pdemb_visualize is True:
            #pd_emb = self.getPersistenceDiagramEmbeddings(file, numTokens)
            pd_emb = self.getHeadPersistenceDiagramEmbeddings(file, numTokens).reshape(2, -1)
            ax = fig.add_subplot(row, column, seq)
            ax.scatter(pd_emb[0], pd_emb[1])
            ax.set_title(textwrap.fill(f"file={file.name}, numTokens={numTokens}", 20), fontsize=8, wrap=True)
        return ax

    # get embeddings from the file descriptor of the output of the SourceFileIterator
    def __getEmbeddingsFromFD(self, fd, getEmbFunc):
        """
        getEmbFunc: getRwkvEmbeddings, getHeadPersistenceDiagramEmbeddings
        """
        logger.debug(f"fd={fd}")
        indexed_file = fd[0]
        file_index = indexed_file[0]
        file = indexed_file[1]            
        indexed_numTokens = fd[1]
        #print("indexed_numTokens=", indexed_numTokens1)
        numTokens_index = indexed_numTokens[0]
        numTokens = indexed_numTokens[1]
        with open(file, "r", encoding="utf-8") as f:
            #rwkv_emb1 = self.getRwkvEmbeddings(f1, numTokens1)
            emb = getEmbFunc(f, numTokens)
        logger.debug(f"emb={emb}")
        return emb

        
    def simMat(self, getEmbFunc, simFunc, numTokens):
        """
            getEmbFunc: getEwkvEmbeddings, getHeadPersistenceDiagramEmbeddings
            simFUnc: getCosineSimilarity, JFIP, Bottleneck
        """        
        logger.info(f"embedding={getEmbFunc.__name__},"
                    f"similarity={simFunc.__name__},"
                    f" tokens={numTokens}")
        # key = f"{getEmbFunc.__name__}:{simFunc.__name__}:tokens={numTokens}:simMat" # old
        key = self.getCacheKey("simmat", embFunc = getEmbFunc,
                               simFunc = simFunc,
                               numTokens = numTokens
                               )
        logger.debug(f"cache key={key}")
        val = self.getDb(key)
        if val is None or self.enable_simmat_cache is False:
            logger.info(f"simMat cache not found")
            simMat = self.getSimMatWithoutCache(getEmbFunc, simFunc, numTokens)
            self.setDb(key, simMat)
            val = simMat
        else:
            logger.info(f"simMat cache found")
        return val
    # 類似度の詳細情報を得る
    def getSimDescWithoutCache(self, getEmbFunc, simFunc, numTokens):
        #iter1 = SourceFileIterator(self.data_top_dir, self.data_subdirs, self.numTokensList)
        iter1 = SourceFileIterator(self.data_top_dir, self.data_subdirs, [numTokens])
        logger.debug(f"subdirs={self.data_subdirs}")
        max_rows = len(self.data_subdirs)
        # max_cols = len(self.numTokensList)
        #output_lens = max_rows*max_cols # 一辺がこのサイズの類似度行列になる想定
        output_lens = len(self.data_subdirs)  # 一辺がこのサイズの類似度行列になる想定
        out1 = next(iter1, None)
        count = 1
        fig = plt.figure()
        while out1:
            #logger.info(out1)
            emb1 = self.__getEmbeddingsFromFD(out1, getEmbFunc)

            #iter2 = SourceFileIterator(self.data_top_dir, self.data_subdirs, self.numTokensList)
            iter2 = SourceFileIterator(self.data_top_dir, self.data_subdirs, [numTokens])
            out2 = next(iter2, None)
            while out2:
                #logger.info(out2)
                    #rwkv_emb2 = self.getRwkvEmbeddings(f2, numTokens2)
                emb2 = self.__getEmbeddingsFromFD(out2, getEmbFunc)
                #sim = self.getCosineSimilarity(rwkv_emb1, rwkv_emb2)
                sim = simFunc(emb1, emb2)
                logger.info(f"{out1},{out2}, sim={sim}")
                out2 = next(iter2, None)
                count += 1
                logger.debug(f"count={count}")
            out1 = next(iter1, None)


    def getSimilarityFromOut(self, count, embFunc, simFunc, out1, out2):
        emb1 = self.__getEmbeddingsFromFD(out1, embFunc)
        emb2 = self.__getEmbeddingsFromFD(out2, embFunc)
        sim = simFunc(emb1, emb2)
        logger.info(f"count={count}, emb1={emb1}(shape={emb1.shape}), emb2={emb2}(shape={emb1.shape}), sim={sim}")
        return (count, sim)
    
    def getSimMatWithoutCache(self, getEmbFunc, simFunc, numTokens):
        """
        numTokensに対し，self.data_sub_dir, self.data_subdirsの
        全ファイルを対象に類似度行列を計算する
        WithoutCacheは，キャッシュrを利用しないという意味で考えていたが，
        キャッシュを参照しないという意味にしてみたい
        なので，キャッシュフラグやキャッシュ有無はチェックせず，結果を
        無理やり（？）書き込む
        """
        ### ここでscaing_constを変えればnumTokensを反映できる
        #n = math.log2(numTokens)
        #r.scaling_const = n * r.scaling_const0
        #iter1 = SourceFileIterator(self.data_top_dir, self.data_subdirs, self.numTokensList)
        iter1 = SourceFileIterator(self.data_top_dir, self.data_subdirs, [numTokens])
        logger.debug(f"subdirs={self.data_subdirs}")
        max_rows = len(self.data_subdirs)
        # max_cols = len(self.numTokensList)
        #output_lens = max_rows*max_cols # 一辺がこのサイズの類似度行列になる想定
        output_lens = len(self.data_subdirs)  # 一辺がこのサイズの類似度行列になる想定
        #out1 = next(iter1, None)
        count = 1
        fig = plt.figure()
        simMatList = []
        pool = multiprocessing.Pool(8)
        #pool = multiprocess.Pool(8)
        emb_pair_list = []
        out_pair_list = []
        emb1_list = []
        emb2_list = []
        self_list = []
        name_list = []
        for out1 in iter1:
            logger.debug(f"target1={out1}")
            #emb1 = self.__getEmbeddingsFromFD(out1, getEmbFunc)

            #iter2 = SourceFileIterator(self.data_top_dir, self.data_subdirs, self.numTokensList)
            iter2 = SourceFileIterator(self.data_top_dir, self.data_subdirs, [numTokens])
            #out2 = next(iter2, None)

            #while out2:
            for out2 in iter2:
                logger.debug(f"targeget2={out2}")            
                ##rwkv_emb2 = self.getRwkvEmbeddings(f2, numTokens2)
                #emb2 = self.__getEmbeddingsFromFD(out2, getEmbFunc)

                #emb_pair_list.append((emb1, emb2))
                out_pair_list.append((count, getEmbFunc, simFunc, out1, out2))
                # forMPを使うとき
                #emb_pair_list.append((self, simFunc.__name__, emb1, emb2))
                #emb1_list.append(emb1)
                #emb2_list.append(emb2)
                ##sim = self.getCosineSimilarity(rwkv_emb1, rwkv_emb2)
                #sim = simFunc(rwkv_emb1, rwkv_emb2)
                
                
                #logger.debug(f"{out1}({rwkv_emb1};{rwkv_emb1.shape}),{out2}({rwkv_emb2};{rwkv_emb2.shape}), sim={sim}")
                #logger.debug(f"{out1}({emb1};{emb1.shape}),{out2}({emb2};{emb2.shape})")
                #out2 = next(iter2, None)
                #simMatList.append(sim)
                count += 1
                logger.debug(f"count={count}")
            #out1 = next(iter1, None)

        logger.info(f"total count={count}")
        logger.debug(f"go to pool with {out_pair_list}")
        #results = pool.starmap(simFunc, emb_pair_list)
        #print("args org=", emb_pair_list)
        #print("args=", (simFunc,emb_pair_list))
        # ノートブックを考えなかったら，ForMPなしでもいいんじゃないか
        #results = pool.starmap(getEmbFuncForMP, emb_pair_list)

        """
        # 最大値を求める
        item = emb1_list[0]
        elem_max = 0.0
        if type(item).__name__ == "Tensor":
            elem_max = torch.max(torch.cat(emb1_list))
        elif type(item).__name__ == "list":
            elem_max = np.max(emb1_list)

        #logger.debug(f"max value in {emb1_list} = {elem_max}")

        self.scaling_const = elem_max / 1000
        """
        #results = pool.starmap(simFunc, emb_pair_list)
        results = pool.starmap(self.getSimilarityFromOut, out_pair_list)
        #with ProcessPoolExecutor(max_workers = 8) as executor:
        #    results = executor.map(getEmbFuncForMP, self_list, name_list, emb1_list, emb2_list)
        #results = map(simFunc, emb1_list, emb2_list)

        logger.debug(f"results = {results}")
        sorted_results = sorted(results, key=lambda x: x[0])
        simMatList = [item[1] for item in sorted_results]
        item = simMatList[0]
        print("item=", type(item).__name__)
        if type(item).__name__ == "Tensor":
            simMat = torch.tensor(simMatList).reshape(int(math.sqrt(count)), int(math.sqrt(count)))
        elif type(item).__name__ == "list" or\
             type(item).__name__ == "float"\
             or type(item).__name__ == "int"\
             or type(item).__name__ == "float64":
            simMat = np.array(simMatList).reshape(int(math.sqrt(count)), int(math.sqrt(count)))
        logger.debug(f"simMat={simMat}")
        key = self.getCacheKey("simmat", embFunc = getEmbFunc,
                               simFunc = simFunc,
                               numTokens = numTokens
                               )

        logger.debug(f"cache key={key}")
        self.setDb(key, simMat)
        return simMat

    def get_simMatrix(self, simFunc, targetVecList):
        logger.info(f"creating simMatrix...func={simFunc}")
        simList = []
        for i in targetVecList:
            for j in targetVecList:
                sim = simFunc(i, j)
                simList.append(sim)
        simMatrix = np.array(simList).reshape(len(targetVecList), len(targetVecList))
        logger.info(simMatrix)
        return simMatrix
                
                
    """
    def getEmbeddingDataset(self):
        for subdir in self.data_subdirs:
            logger.info(f"target subdir={subdir}")
            path = os.path.join(self.data_top_dir, subdir)
            files = os.listdir(path)
            indexed_files = enumerate(files)
            #for file in files:
            for indexed_file in indexed_files:
                logger.info(f"indexed_file={indexed_file}")
                file_index = indexed_file[0]
                file = indexed_file[1]
                logger.info(f"target file={file}")
                file_path = os.path.join(path, file)
                numTokensList = [1024, 2048, 4096, 8192]
                indexed_numTokensList = enumerate(numTokensList)
                for indexed_numTokens in indexed_numTokensList:
                    numTokens_index = indexed_numTokens[0]
                    numTokens = indexed_numTokens[1]
                    #print("file_path=", file_path)
                    with open(file_path, "r", encoding="utf-8") as f:
                        embeddings = self.getEmbeddings(f, numTokens)
                        #print("embed shape=", embeddings.shape)
    """
    def getEmbeddingDataset(self, cache_rebuild = False):
        iter = SourceFileIterator(self.data_top_dir, self.data_subdirs, self.numTokensList)
        #out = next(iter, None)
        #while out:
        pool = multiprocessing.Pool(8)
        args_list = []
        for out in tqdm(iter):
            print(out)
            indexed_file = out[0]
            file_index = indexed_file[0]
            file = indexed_file[1]            
            indexed_numTokens = out[1]
            #print("indexed_numTokens=", indexed_numTokens)
            numTokens_index = indexed_numTokens[0]
            numTokens = indexed_numTokens[1]
            #with open(file, "r", encoding="utf-8") as f:
                #print("file=", file)
                #embeddings = self.getEmbeddings(f, numTokens, cache_rebuild)
                #args_list.append((f, numTokens, cache_rebuild))
                #print("embed shape=", embeddings.shape)
            #out = next(iter, None)
            args_list.append((out, cache_rebuild))
        logger.debug(f"go to pool with {args_list}")
        #results = pool.starmap(self.getEmbeddings, args_list)
        results = pool.starmap(self.getEmbeddingsFromOut, args_list)

        logger.debug(f"results = {results}")

    def getSimilarityMatrixDataset(self, cache = True):
        if cache is False:
            self.enable_simmat_cache = False
        logger.info("Construct similarity matrix dataset and cache them")
        for embedding, similarity in self.datasetParameters:
            for numTokens in self.numTokensList:
                sim = self.simMat(embedding, similarity, numTokens)

    def getAllScores(self, cache = True):
        scoreArray = []
        if cache is False:
            self.enable_simmat_cache = False
        logger.info("Construct similarity matrix dataset and cache them")
        for embedding, similarity in self.datasetParameters:
            for numTokens in self.numTokensList:
                sim = self.simMat(embedding, similarity, numTokens)
                score = self.getScore(sim)
                scoreArray.append(score)
        return scoreArray

    def hit(self, i, j, simMat):
        hit = 0
        k = self.cl[i]
        l = self.cl[j]
        if k == l and simMat[i, j] >= 0.5:
            hit = 1
        elif k == l and simMat[i, j] < 0.5:
            hit = 0
        elif k != l and simMat[i, j] < 0.5:
            hit = 1
        elif k != l and simMat[i, j] >= 0.5:
            hit = 0
        return hit
    
    def getScore(self, simMat):
        n = simMat.shape[0]
        score = 0
        for i in range(n):
            for j in range(n):
                score += self.hit(i, j, simMat)
        return score/(n*n)
    
    def close(self):
        self.db.close()

if __name__ == "__main__":
    args = sys.argv
    if args[1] == "--rebuild-cache":
        if args[2] == "bottleneck":
            logger.info(f"rebuild bottleneck cache")
            start = time.time()
            r = rwkv(model_name, tokenizer_name, model_load = False)
            #r.enable_bottleneck_cache = False
            c = r.deleteCache("dis",
                              simFunc = r.Bottleneck
                              )
            #r = rwkv(model_name, tokenizer_name, model_load = True)
            #r.pdemb_postfunc = r.normalize
            #r.pdemb_postfunc = r.regularize
            #r.pdemb_postfunc = r.scaling
            # テストで一部だけ実行するとき
            embFunc = r.getHeadPersistenceDiagramEmbeddings
            simFunc = r.BottleneckSim
            #for numTokens in r.numTokensList:
            #for embFunc, simFunc in [[embFunc, simFunc]]:
            #for numTokens in [1024]:
            #for embFunc, simFunc in r.datasetParameters:
            for numTokens in r.numTokensList:
                r.getSimMatWithoutCache(embFunc, simFunc, numTokens)
            end = time.time()
            logger.info(f"finished rebuilding cache. elapsed = {end - start}")
        if args[2] == "wasserstein":
            logger.info(f"rebuild wasserstein cache")
            start = time.time()
            r = rwkv(model_name, tokenizer_name, model_load = False)
            r.enable_wasserstein_cache = False
            #r = rwkv(model_name, tokenizer_name, model_load = True)
            #r.pdemb_postfunc = r.normalize
            #r.pdemb_postfunc = r.regularize
            #r.pdemb_postfunc = r.scaling
            # テストで一部だけ実行するとき
            embFunc = r.getHeadPersistenceDiagramEmbeddings
            simFunc = r.WassersteinSim
            #for numTokens in r.numTokensList:
            #for embFunc, simFunc in [[embFunc, simFunc]]:
            #for numTokens in [1024]:
            #for embFunc, simFunc in r.datasetParameters:
            for numTokens in r.numTokensList:
                r.getSimMatWithoutCache(embFunc, simFunc, numTokens)
            end = time.time()
            logger.info(f"finished rebuilding cache. elapsed = {end - start}")
        if args[2] == "sw":
            logger.info(f"rebuild sw cache (and the follow steps)")
            start = time.time()
            r = rwkv(model_name, tokenizer_name, model_load = False)
            #c = r.deleteCache("swemb", r.getSlidingWindowEmbeddings, None, None)
            #c = r.deleteCache("pdemb", r.getPersistenceDiagramEmbeddings, None, None)
            c = r.deleteCache("emb",
                              embFunc = r.getSlidingWindowEmbeddings
                              )
            c = r.deleteCache("emb",
                              embFunc = r.getPersistenceDiagramEmbeddings
                              )
            c = r.deleteCache("embmat",
                              embFunc = r.getHeadPersistenceDiagramEmbeddings
                              )
            c = r.deleteCache("simmat",
                              embFunc = r.getHeadPersistenceDiagramEmbeddings
                              )
            c = r.deleteCache("dis",
                              simFunc = r.Bottleneck
                              )
            #r.enable_sw_cache = False # これは不必要になったはず
            #r = rwkv(model_name, tokenizer_name, model_load = True)
            #r.pdemb_postfunc = r.normalize
            #r.pdemb_postfunc = r.regularize
            #r.pdemb_postfunc = r.scaling
            # テストで一部だけ実行するとき
            #embFunc = r.getHeadPersistenceDiagramEmbeddings
            #simFunc = r.WassersteinSim
            #for numTokens in r.numTokensList:
            #for embFunc, simFunc in [[embFunc, simFunc]]:
            #for numTokens in [1024]:
            #for embFunc, simFunc in r.datasetParameters:
            r.getEmbeddingDataset(cache_rebuild = False)
            end = time.time()
            logger.info(f"finished rebuilding cache. elapsed = {end - start}")
        if args[2] == "pd":
            logger.info(f"rebuild PD cache (and the follow steps)")
            start = time.time()
            r = rwkv(model_name, tokenizer_name, model_load = False)
            #                  embFunc = r.getSlidingWindowEmbeddings
            #                  )
            c = r.deleteCache("emb",
                              embFunc = r.getPersistenceDiagramEmbeddings
                              )
            c = r.deleteCache("embmat",
                              embFunc = r.getHeadPersistenceDiagramEmbeddings
                              )
            c = r.deleteCache("simmat",
                              embFunc = r.getHeadPersistenceDiagramEmbeddings
                              )
            c = r.deleteCache("dis",
                              simFunc = r.Bottleneck
                              )
            r.getEmbeddingDataset(cache_rebuild = False)
            end = time.time()
            logger.info(f"finished rebuilding cache. elapsed = {end - start}")
        if args[2] == "rwkv":
            logger.info(f"rebuild rwkv embedding cache")
            start = time.time()
            r = rwkv(model_name, tokenizer_name, model_load = True)
            #c = r.deleteCache("emb", embFunc = r.getRwkvEmbeddings)
            #c = r.deleteCache("emb", embFunc = r.getSlidingWindowEmbeddings)
            #c = r.deleteCache("emb", embFunc = r.getPersistenceDiagramEmbeddings)
            r.data_subdirs = ["einstein"]
            iter = SourceFileIterator(r.data_top_dir, r.data_subdirs, r.numTokensList)
            pool = multiprocessing.Pool(8)
            args_list = []
            for out in tqdm(iter):
                print(out)
                args_list.append((out, True))
                #r.getEmbeddingsFromOut(out, cache_rebuild = True) # rwkv/sw/pd全部やる
                indexed_file = out[0]
                file_index = indexed_file[0]
                file = indexed_file[1]            
                indexed_numTokens = out[1]
                #print("indexed_numTokens=", indexed_numTokens)
                numTokens_index = indexed_numTokens[0]
                numTokens = indexed_numTokens[1]
                with open(file, "r", encoding="utf-8") as f:
                    r.getRwkvEmbeddings(f, numTokens)
            # RuntimeError: Tried to serialize object __torch__.rwkv.model.RWKV which does not have a __getstate__ method defined!   
            #pool.starmap(r.getEmbeddingsFromOut, args_list)
            end = time.time()
            logger.info(f"finished rebuilding cache. elapsed = {end - start}")
        if args[2] == "sim-all": # BottleneckSimも含むはず
            logger.info(f"rebuild similarity matrix cache")
            start = time.time()
            r = rwkv(model_name, tokenizer_name, model_load = False)
            c = r.deleteCache("simmat")
            r.all_simMatrixPlot()
            end = time.time()
            logger.info(f"finished rebuilding cache. elapsed = {end - start}")
        if args[2] == "sim":
            start = time.time()
            r = rwkv(model_name, tokenizer_name, model_load = False)
            #simFunc = r.CosSim
            simFunc = r.BottleneckSim
            logger.info(f"rebuild similarity matrix cache for {simFunc}")

            embFunc = r.getHeadPersistenceDiagramEmbeddings
            numTokens = 1024
            c = r.deleteCache("simmat", embFunc = embFunc,
                              simFunc = simFunc,
                              numTokens = numTokens)
            r.simMat(embFunc, simFunc, numTokens)
            end = time.time()
            logger.info(f"finished rebuilding cache. elapsed = {end - start}")
    if args[1] == "--list-cache-all":
        r = rwkv(model_name, tokenizer_name, model_load = False)
        keys = r.listCache("")
        print(keys)
    if args[1] == "--cache-volume":
        r = rwkv(model_name, tokenizer_name, model_load = False)
        size = r.db.volume()
        print(size)
    if args[1] == "--list-cache":
        if args[2] == "rwkv":
            logger.info(f"list rwkv embedding cache")
            start = time.time()
            r = rwkv(model_name, tokenizer_name, model_load = False)
            keys = r.listCache("emb", embFunc = r.getRwkvEmbeddings)
            logger.info(f"target keys[0:3]={keys[0:3]}")
            for key in keys:
                print(key)
        if args[2] == "sw":
            logger.info(f"list sw embedding cache")
            start = time.time()
            r = rwkv(model_name, tokenizer_name, model_load = False)
            keys = r.listCache("emb", embFunc = r.getSlidingWindowEmbeddings)
            logger.info(f"target keys[0:3]={keys[0:3]}")
            for key in keys:
                print(key)
                
        if args[2] == "bottleneck":
            logger.info(f"list bottleneck cache")
            start = time.time()
            r = rwkv(model_name, tokenizer_name, model_load = False)
            #r.enable_bottleneck_cache = False
            keys = r.geteCacheKeys("dis",
                              simFunc = r.Bottleneck
                              )
    if args[1] == "--export-cache":
        if args[2] == "rwkv":
            logger.info(f"export rwkv embedding cache")
            start = time.time()
            r = rwkv(model_name, tokenizer_name, model_load = False)
            kv = {}
            keys = r.listCache("emb", embFunc = r.getRwkvEmbeddings)
            logger.info(f"target keys[0:3]={keys[0:3]}")
            for key in keys:
                print(key)
                val = r.db.get(key)
                print(val)
                kv[key] = val
            filename = "rwkv.pickle"
            with open(filename, "wb") as f:
                pickle.dump(kv, f)
    if args[1] == "--import-cache":
        if args[2] == "rwkv":
            logger.info(f"import rwkv embedding cache")
            start = time.time()
            r = rwkv(model_name, tokenizer_name, model_load = False)
            kv = {}
            keys = r.listCache("emb", embFunc = r.getRwkvEmbeddings)
            logger.info(f"target keys[0:3]={keys[0:3]}")
            for key in keys:
                print(key)
                val = r.db.get(key)
                print(val)
                kv[key] = val
            filename = "rwkv.pickle"
            with open(filename, "rb") as f:
                kv = pickle.load(f)
            
            for k in kv:
                r.db.set(k, kv[k])
