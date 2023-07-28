import time
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
import textwrap
import math

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


import shelve

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

CharRWKV_HOME = "/research/ChatRWKV"

class rwkv():
    
    def __init__(self, model_filename, tokenizer_filename, model_load = True):
        logger.info(f"initializing rwkv")
        self.model_filename = model_filename
        if model_load:
            self.model = RWKV(model=model_filename, strategy='cuda fp16i8')
            self.pipeline = PIPELINE(self.model, tokenizer_filename) # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV

        #self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_filename)
        #self.tokenizer = Tokenizer.from_file(tokenizer_filename)
        self.tokenizer = Tokenizer.from_file(tokenizer_filename)
        print(self.tokenizer.__class__)
        self.tokenizer = rwkv_tokenizer("rwkv_vocab_v20230424.txt")
        self.AVOID_REPEAT_TOKENS = []
        self.start = time.time()

        if self.tokenizer.__class__ == "tokenizers.Tokenizer":
            self.AVOID_REPEAT = '，。：？！'
            for i in self.AVOID_REPEAT:
                dd = self.tokenizer.encode(i).ids
                assert len(dd) == 1
                self.AVOID_REPEAT_TOKENS += dd
        self.CHUNK_LEN: int = 8192*4
        """Batch size for prompt processing."""

        #self.key_prefix = self.__class__.__name__ + '_' + str(id(self)) + '_' + model_filename
        self.key_prefix = self.__class__.__name__ + ':' + model_filename
        print("key_prefix=", self.key_prefix)
        filename = "rwkv_gutenberg.db"
        self.db = shelve.open(filename)

        embedding_dimension_periodic = 3
        embedding_time_delay_periodic = 1
        stride = 1
        self.sw_embedder = TakensEmbedding(
            #parameters_type="fixed",
            #n_jobs=2,
            time_delay=embedding_time_delay_periodic,
            dimension=embedding_dimension_periodic,
            stride=stride,
         )
        self.enable_rwkvemb_cache = True
        self.enable_swemb_cache = True
        self.enable_pdemb_cache = True

        self.enable_simmat_cache = True
        self.enable_pdemb_visualize = True
        self.data_top_dir = "./data"
        self.data_subdirs = ["carroll", "einstein", "lovecraft"]
        self.numTokensList = [1024, 2048, 4096]
        self.topN = 10 # PDから取るベクトルの数

        # ディレクトリ情報から自動生成したい
        self.cl = [0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2]  # インデックスが属するクラスタID情報


        # Dataset作成用パラメータ
        self.datasetParameters = [
            [self.getRwkvEmbeddings, self.CosSim],
            [self.getRwkvEmbeddings, self.JFIP],
            [self.getHeadPersistenceDiagramEmbeddings,
                              self.CosSim],
            [self.getHeadPersistenceDiagramEmbeddings,
                              self.JFIP],
            [self.getHeadPersistenceDiagramEmbeddings,
                              self.Bottleneck]
            ]
    def setDb(self, key, val):
        keyval = f"{self.key_prefix}:{key}"
        
        self.db[keyval] = val
        self.db.sync()
    def getDb(self, key):
        keyval = f"{self.key_prefix}:{key}"
        return self.db.get(keyval)
    def run_rnn(self, tokens, newline_adj = 0):
        start = time.time()
        #global model_tokens, model_state
        model_tokens = []
        model_state = None
        #logger.debug(f"input tokens={tokens}")
        tokens = [int(x) for x in tokens]
        #logger.debug(f"tokens={tokens}")
        model_tokens += tokens

        while len(tokens) > 0:
            out, model_state = self.model.forward(
                tokens[: self.CHUNK_LEN], model_state
            )
            tokens = tokens[self.CHUNK_LEN :]
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
        return out
    def encoding(self, text):
        enc = self.tokenizer.encode(text)
        tokenIds = enc.ids
        tokens = enc.tokens
        print("tokens=", tokens)
        return tokenIds

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

        return '\n'.join(lines)

    
    def getRwkvEmbeddings(self, file, numTokens):
        logger.debug(f"file={file}")
        text = file.read()
        file.seek(0)
        #logger.debug(f"raw text={text[0:200]}")
        text = self.removeGutenbergComments(text)
        #print("ETLed text=", text[0:200])
        #embeddign = rwkv_embeddings(text)
        tokens = self.encoding(text)[:numTokens]
        #logger.debug(f"tokens={tokens}")
        key = f"{file.name}:tokens={numTokens}:rwkvemb"
        val = self.getDb(key)
        if val == None or self.enable_rwkvemb_cache is False:
            logger.debug(f"getDB({key}) is None. Rebuild Cache")
            embeddings = self.run_rnn(tokens)
            self.setDb(key, embeddings)
            val = embeddings
            #print(embeddings[:30])
        else:
            logger.debug(f"RWKV embedding cache found")
        logger.debug(f"rwkv embedding shape={val.shape}")
        return val

    def getSlidingWindowEmbeddings(self, file, numTokens):
        key = f"{file.name}:tokens={numTokens}:swemb"
        val = self.getDb(key)
        #logging.info(f"sliding window cache={val}")
        if val is None or self.enable_swemb_cache is False:
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
        logger.debug(f"sliding window embedding shape={val.shape}")
        return val
    def getHeadPersistenceDiagramEmbeddings(self, file, numTokens):
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
        #print("sorted embeddings=", sorted_emb)
        # 最後に，[2, n]のベクトル群を1次元に変換する
        sorted_emb = sorted_emb.reshape(sorted_emb.shape[0]*sorted_emb.shape[1])
        #print("reshaped embeddings=", sorted_emb)
        #print("re-reshaped embeddings=", sorted_emb.reshape(2, -1))
        return sorted_emb
        

    # 1次元のPDの2次元ベクトル列をnumpyで返す
    def getPersistenceDiagramEmbeddings(self, file, numTokens):
        key = f"{file.name}:tokens={numTokens}:pdemb"
        val = self.getDb(key)
        #logger.info(f"sliding window cache={val}")
        if val is None or self.enable_pdemb_cache is False:
            logger.debug(f"getDB({key}) is None. Rebuild Cache")
            sw_emb = self.getSlidingWindowEmbeddings(file, numTokens)
            logger.debug(f"input sw emg shape={sw_emb.shape}")
            hc.PDList.from_alpha_filtration(sw_emb, 
                                save_to="pointcloud.pdgm",
                                save_boundary_map=True)
            pdlist = hc.PDList("pointcloud.pdgm")
            pd1 = pdlist.dth_diagram(1)
            pd_embeddings = np.array(pd1.birth_death_times())

            logger.debug(f"set key={key}")
            self.setDb(key, pd_embeddings)
            
            val = pd_embeddings
        else:
            logger.debug(f"PD embedding cache found")
        logger.debug(f"Persistence Diagram embedding shape={val.shape}")
        return val

    # キャッシュは全部ここで管理したい
    # 言語モデルのベクトル
    # Sliding Window埋め込み
    # パーシステンスホモロジーの2次元ベクトル
    def getEmbeddings(self, file, numTokens, cache_rebuild = False):
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

        self.db.sync()
        return pd_emb

            
    def CosSim(self, list1, list2):
        #print(type(list1).__name__)
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
        print(type(f).__name__)
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
        sigma = 3.0
        kernel_func = lambda x: np.exp(-x**2 / (2 * sigma**2))
        if type(list1).__name__ == "Tensor":
            sim = cos_sim(list1.reshape(1, -1), list2.reshape(1, -1)).item()
        elif type(list1).__name__ == "list" or type(list1).__name__ == "ndarray":
            list1 = list1.reshape(2, -1)
            list2 = list2.reshape(2, -1)
            pd1 = hc.PD.from_birth_death(1, list1[0, :], list1[1, :])
            pd2 = hc.PD.from_birth_death(1, list2[0, :], list2[1, :])
            dis = hc.distance.bottleneck(pd1, pd2)
        sim = kernel_func(dis)
        return sim

    def simMatrixPlot(self, fig, ax, matrix):

        ax.invert_yaxis()
        #cax=ax.imshow(matrix, cmap="Paired", origin="lower")
        cax=ax.imshow(matrix, cmap="viridis", origin="upper")
        cbar = fig.colorbar(cax)


    def all_simMatrixPlot(self):
        fig = plt.figure(figsize=(10, 6))
        fig.subplots_adjust(hspace=0.9, wspace=0.2)
        fig.tight_layout(rect=[0,0,1,0.96])
        max_cols = 4 # 論文の図から
        max_rows = 4 # 縦は定めなくてどんどん増えてもいいんだけど，subplotの仕様上仕方がない
        embFuncList = [self.getRwkvEmbeddings, self.getHeadPersistenceDiagramEmbeddings]
        simFuncList = [self.CosSim, self.JFIP, self.Bottleneck]
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
        for simFunc in [self.CosSim, self.JFIP, self.Bottleneck]:
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
        simFuncList = [self.CosSim, self.JFIP, self.Bottleneck]
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
        for simFunc in [self.CosSim, self.JFIP, self.Bottleneck]:
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
    def pd_subplot(self, fig, file, numTokens, row, column, seq):
        ax = None
        ### Visualize
        if self.enable_pdemb_visualize is True:
            pd_emb = self.getPersistenceDiagramEmbeddings(file, numTokens)
            ax = fig.add_subplot(row, column, seq)
            ax.scatter(pd_emb[0], pd_emb[1])
            ax.set_title(textwrap.fill(f"file={file.name}, numTokens={numTokens}", 20), fontsize=8, wrap=True)
        return ax

    def all_pd_plot(self):
        iter = SourceFileIterator(self.data_top_dir, self.data_subdirs, self.numTokensList)
        max_rows = len(self.data_subdirs)
        max_cols = len(self.numTokensList)
        out = next(iter, None)
        count = 1
        fig = plt.figure()
        while out:
            logger.info(out)
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

    # get embeddings from the file descriptor of the output of the SourceFileIterator
    def __getEmbeddingsFromFD(self, fd, getEmbFunc):
        print("fd=", fd)
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
        return emb

    """
        getEmbFunc: getEwkvEmbeddings, getHeadPersistenceDiagramEmbeddings
        simFUnc: getCosineSimilarity, JFIP, Bottleneck
    """        
    def simMat(self, getEmbFunc, simFunc, numTokens):
        logger.info(f"embedding={getEmbFunc.__name__},"
                    f"similarity={simFunc.__name__},"
                    f" tokens={numTokens}")
        key = f"{getEmbFunc.__name__}:{simFunc.__name__}:tokens={numTokens}:simMat"
        print("key=", key)
        val = self.getDb(key)
        if val is None or self.enable_simmat_cache is False:
            logger.debug(f"simMat cache not found")
            simMat = self.getSimMatWithoutCache(getEmbFunc, simFunc, numTokens)
            self.setDb(key, simMat)
            val = simMat
        else:
            logger.debug(f"simMat cache found")
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
            rwkv_emb1 = self.__getEmbeddingsFromFD(out1, getEmbFunc)

            #iter2 = SourceFileIterator(self.data_top_dir, self.data_subdirs, self.numTokensList)
            iter2 = SourceFileIterator(self.data_top_dir, self.data_subdirs, [numTokens])
            out2 = next(iter2, None)
            while out2:
                #logger.info(out2)
                    #rwkv_emb2 = self.getRwkvEmbeddings(f2, numTokens2)
                rwkv_emb2 = self.__getEmbeddingsFromFD(out2, getEmbFunc)
                #sim = self.getCosineSimilarity(rwkv_emb1, rwkv_emb2)
                sim = simFunc(rwkv_emb1, rwkv_emb2)
                logger.info(f"{out1},{out2}, sim={sim}")
                out2 = next(iter2, None)
                count += 1
                logger.debug(f"count={count}")
            out1 = next(iter1, None)


    
    def getSimMatWithoutCache(self, getEmbFunc, simFunc, numTokens):
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
        simMatList = []
        while out1:
            logger.info(out1)
            rwkv_emb1 = self.__getEmbeddingsFromFD(out1, getEmbFunc)

            #iter2 = SourceFileIterator(self.data_top_dir, self.data_subdirs, self.numTokensList)
            iter2 = SourceFileIterator(self.data_top_dir, self.data_subdirs, [numTokens])
            out2 = next(iter2, None)
            while out2:
                logger.info(out2)
                    #rwkv_emb2 = self.getRwkvEmbeddings(f2, numTokens2)
                rwkv_emb2 = self.__getEmbeddingsFromFD(out2, getEmbFunc)
                #sim = self.getCosineSimilarity(rwkv_emb1, rwkv_emb2)
                sim = simFunc(rwkv_emb1, rwkv_emb2)
                logger.info(f"{out1},{out2}, sim={sim}")
                out2 = next(iter2, None)
                simMatList.append(sim)
                count += 1
                logger.debug(f"count={count}")
            out1 = next(iter1, None)
        item = simMatList[0]
        print("item=", type(item).__name__)
        if type(item).__name__ == "Tensor":
            simMat = torch.tensor(simMatList).reshape(int(math.sqrt(count)), int(math.sqrt(count)))
        elif type(item).__name__ == "list"\
          or type(item).__name__ == "float"\
          or type(item).__name__ == "int"\
          or type(item).__name__ == "float64":
            simMat = np.array(simMatList).reshape(int(math.sqrt(count)), int(math.sqrt(count)))
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
        out = next(iter, None)
        while out:
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
                embeddings = self.getEmbeddings(f, numTokens, cache_rebuild)
                #print("embed shape=", embeddings.shape)
            out = next(iter, None)
            

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
