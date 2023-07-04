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

class rwkv():
    
    def __init__(self, model_filename, tokenizer_filename):
        logger.info(f"initializing rwkv")
        self.model_filename = model_filename
        self.model = model = RWKV(model=model_filename, strategy='cuda fp16i8')
        #self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_filename)
        self.tokenizer = Tokenizer.from_file(tokenizer_filename)
        self.pipeline = PIPELINE(self.model, tokenizer_filename) # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
        self.AVOID_REPEAT_TOKENS = []
        self.start = time.time()


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
        self.enable_pdemb_visualize = True
        self.data_top_dir = "./data"
        self.data_subdirs = ["carroll", "einstein", "lovecraft"]
        self.numTokensList = [1024, 2048, 4096]

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

        tokens = [int(x) for x in tokens]
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
        tokens = self.tokenizer.encode(text).ids
        #print("tokens=", tokens)
        return tokens
    def getRwkvEmbeddings(self, file, numTokens):
        text = file.read()
        #print("text=", text)
        #embeddign = rwkv_embeddings(text)
        tokens = self.encoding(text)[:numTokens]
       
        key = f"{file.name}:tokens={numTokens}:rwkvemb"
        val = self.getDb(key)
        if val == None:
            logger.debug(f"getDB({key}) is None. Rebuild Cache")
            embeddings = self.run_rnn(tokens)
            logger.debug("set key=", key)
            self.setDb(key, embeddings)
            val = embeddings
            #print(embeddings[:30])
            logger.debug("getDB after setDb key = ", key, "val =", self.getDb(key))
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
    def getEmbeddings(self, file, numTokens):
        logger.info(f"Start getEmbeddings: file={file}, numTokens={numTokens}")
        ### 言語モデルのベクトル
        rwkv_emb = self.getRwkvEmbeddings(file, numTokens)
        
        ### Sliding Window
        sw_emb = self.getSlidingWindowEmbeddings(file, numTokens)
        

        ### Persistence Diagram
        pd_emb = self.getPersistenceDiagramEmbeddings(file, numTokens)

        self.db.sync()
        return pd_emb

            
    def getCosineSimilarity(list1, list2):
        sim = 1 - spatial.distance.cosine(list1, list2)
        return sim

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

    #def all_pd_plot(self):

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
    def getEmbeddingDataset(self):
        iter = SourceFileIterator(self.data_top_dir, self.data_subdirs, self.numTokensList)
        out = next(iter, None)
        while out:
            print(out)
            indexed_file = out[0]
            file_index = indexed_file[0]
            file = indexed_file[1]            
            indexed_numTokens = out[1]
            print("indexed_numTokens=", indexed_numTokens)
            numTokens_index = indexed_numTokens[0]
            numTokens = indexed_numTokens[1]
            with open(file, "r", encoding="utf-8") as f:
                print("file=", file)
                embeddings = self.getEmbeddings(f, numTokens)
                #print("embed shape=", embeddings.shape)
            out = next(iter, None)
            
    #def getCosineSimilarityMatrixFromRwkv(self, file, numTokens):
        
        
        
    def close(self):
        self.db.close()
