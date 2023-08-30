#from absl import logging
import logging
#import tensorflow as tf
from tqdm import tqdm
import torch.multiprocessing as multiprocessing
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import re
import seaborn as sns
#import torch.multiprocessing as multiprocessing
from tokenizers import Tokenizer
import time
from params import Params
from cache import Cache
import homcloud.interface as hc 

from rwkv.utils import PIPELINE, PIPELINE_ARGS

from logging import config

from source_file_iterator import SourceFileIterator
import textwrap

config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)
print("logger=", logger)
import diskcache

from embeddings_base import embeddings_base

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]

tokenizer_name = '20B_tokenizer.json'
#multiprocessing.set_start_method("spawn", force=True)

#from tensorflow.experimental.numpy import enable_numpy_behavior
#enable_numpy_behavior()
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class use_wrapper(embeddings_base):
    def __init__(self, model_filename, tokenizer_filename, model_load = True):
        logger.info(f"initializing USE")
        super().__init__(model_filename, tokenizer_filename, model_load)
        self.model_filename = model_filename
        if model_load:
            self.model = hub.load(module_url)
            # self.model = RWKV(model=model_filename, strategy='cuda fp16i8')
            self.pipeline = PIPELINE(self.model, tokenizer_filename) # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
            print ("module %s loaded" % module_url)


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
        params = Params("config-use.toml")
        filename = params.cache_filename
        logger.info(f"cache_filename={params.cache_filename}")
        #self.db = diskcache.Cache(filename)
        self.db = Cache(self, filename, model_filename, self.tokenizer)
    
    def embedding(self, input):
        #logger.debug(f"input text={input}")
        #print(f"input text={input}")
        #print(f"and embed={self.model([input])}")
        return self.model([input])

    def getUseEmbeddings(self, file, numTokens):
        start = time.time()
        text = file.read()
        file.seek(0)
        logger.debug(f"raw text={text[0:200]}")
        text = self.removeGutenbergComments(text)
        logger.debug(f"ETLed text={text[0:200]}")
        #embeddign = rwkv_embeddings(text)
        enc = self.encoding(text)
        tokens = enc.ids[:numTokens]
        #logger.debug(f"token ids={tokens}")
        #print(f"token ids={tokens}")
        key = self.db.getCacheKey("emb", file = file,
                                  numTokens = numTokens,
                                  embFunc = self.getUseEmbeddings
                                  )
        val = self.db.getDb(key)
        if self.enable_useemb_cache:
            if val is None:
                logger.debug(f"getDB({key}) is None. Rebuild Cache")
                if hasattr(self, "model"):
                    embeddings = self.embedding(self.decoding(tokens))
                else:
                    logger.error("LLM model was not loaded. Cannot continue")
                    raise NameError("LLM model was not loaded. Cannot continue")
                logger.debug(f"writing cache... key={key}")
                self.db.setDb(key, embeddings)
                val = embeddings
                #print(embeddings[:30])
            else:
                logger.debug(f"getDB({key}) found the cache value")
        else:
            logger.debug(f"Cache access is disabled")
            embeddings = self.embedding(self.decoding(tokens))
            val = embeddings
        logger.debug(f"rwkv embedding shape={val.shape}")
        end = time.time()
        logger.info(f"elapsed = {end - start}")

        return val


    def getSlidingWindowEmbeddings(self, file, numTokens):
        start = time.time()
        # key = f"{file.name}:tokens={numTokens}:swemb" # old
        #key = self.getCacheKey("swemb", None, file, numTokens,
        #                       self.getSlidingWindowEmbeddings, None)
        key = self.db.getCacheKey("emb", file = file,
                               numTokens = numTokens,
                               embFunc = self.getSlidingWindowEmbeddings
                               )
        val = self.db.getDb(key)
        #logging.info(f"sliding window cache={val}")
        if self.enable_swemb_cache:
            if val is None:
                logger.debug(f"getDB({key}) is None. Rebuild Cache")
                use_emb = self.getUseEmbeddings(file, numTokens)
                logger.debug(f"input rwkv emg shape={use_emb.shape}")
                sw_embeddings = self.sw_embedder.fit_transform(use_emb.reshape(1, -1).cpu())
                sw_embeddings = sw_embeddings[0, :, :]
                logger.debug(f"set key={key}")
                self.db.setDb(key, sw_embeddings)
                val = sw_embeddings
            else:
                logger.debug(f"SW embedding cache found")
                logger.info(f"getDB({key}) found the cache value")
        else:
            logger.info(f"Cache access is disabled")
            use_emb = self.getUseEmbeddings(file, numTokens)
            logger.debug(f"input use emb shape={use_emb.shape}")
            if hasattr(self, "model"):
                if type(use_emb).__name__ == "Tensor":
                    sw_embeddings = self.sw_embedder.fit_transform(use_emb.reshape(1, -1).cpu())
                else:
                    sw_embeddings = self.sw_embedder.fit_transform(use_emb.reshape(1, -1))
                sw_embeddings = sw_embeddings[0, :, :]
            else:
                logger.error("LLM model was not loaded. Cannot continue")
                raise NameError("LLM model was not loaded. Cannot continue")
            logger.debug(f"set key={key}")
            self.db.setDb(key, sw_embeddings)
            val = sw_embeddings

            
        logger.debug(f"sliding window embedding shape={val.shape}")
        end = time.time()
        logger.info(f"elapsed = {end - start}")
        return val

    # 1次元のPDの2次元ベクトル列をnumpyで返す
    def getPersistenceDiagramEmbeddings(self, file, numTokens):
        """
        PDの2次元ベクトル列をnumpyで返す
        """
        start = time.time()
        key = self.db.getCacheKey("emb", file = file,
                               numTokens = numTokens,
                               embFunc = self.getPersistenceDiagramEmbeddings
                               )
        val = self.db.getDb(key)
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
                self.db.setDb(key, pd_embeddings)

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
    def getHeadPersistenceDiagramEmbeddings(self, file, numTokens):
        """
        getPersistenceDiagramEmbeddingsで得た埋め込みの，
        """
        emb = self.getPersistenceDiagramEmbeddings(file, numTokens) # [2, n]
        lifeTime = emb[1, :] - emb[0, :]
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

if __name__ == "__main__":
    logger.info(f"■■■■■■■■■■■■■■■■■■■■■■■ main")
    args = sys.argv
    model_filename = "universal-sentence-encoder-multilingual-large_3"
    logger.info(f"model_filename={model_filename}")
    u = use_wrapper(model_filename, tokenizer_name, model_load = True)
    logger.info(f"use_wrapper: {u}")
    if args[1] == "--rebuild-cache":
        if args[2] == "use":
            logger.info(f"rebuild use embedding cache")
            start = time.time()
            u = use_wrapper(model_filename, tokenizer_name, model_load = True)
            #c = r.deleteCache("emb", embFunc = r.getRwkvEmbeddings)
            #c = r.deleteCache("emb", embFunc = r.getSlidingWindowEmbeddings)
            #c = r.deleteCache("emb", embFunc = r.getPersistenceDiagramEmbeddings)
            #u.data_subdirs = ["einstein"]
            iter = SourceFileIterator(u.data_top_dir, u.data_subdirs, u.numTokensList)
            #pool = multiprocessing.Pool(8)
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
                    u.getUseEmbeddings(f, numTokens)
            # RuntimeError: Tried to serialize object __torch__.rwkv.model.RWKV which does not have a __getstate__ method defined!   
            #pool.starmap(r.getEmbeddingsFromOut, args_list)
            end = time.time()
            logger.info(f"finished rebuilding cache. elapsed = {end - start}")
        if args[2] == "sw":
            logger.info(f"rebuild sw cache (and the follow steps)")
            start = time.time()
            u = use_wrapper(model_filename, tokenizer_name, model_load = False)
            #c = r.deleteCache("swemb", r.getSlidingWindowEmbeddings, None, None)
            #c = r.deleteCache("pdemb", r.getPersistenceDiagramEmbeddings, None, None)
            c = u.db.deleteCache("emb",
                              embFunc = u.getSlidingWindowEmbeddings
                              )
            """
            c = u.db.deleteCache("emb",
                              embFunc = r.getPersistenceDiagramEmbeddings
                              )
            c = u.db.deleteCache("embmat",
                              embFunc = r.getHeadPersistenceDiagramEmbeddings
                              )
            c = u.db.deleteCache("simmat",
                              embFunc = r.getHeadPersistenceDiagramEmbeddings
                              )
            c = u.db.deleteCache("dis",
                              simFunc = r.Bottleneck
                              )
            """
            iter = SourceFileIterator(u.data_top_dir, u.data_subdirs, u.numTokensList)
            for out in tqdm(iter):
                indexed_file = out[0]
                file_index = indexed_file[0]
                file = indexed_file[1]            
                indexed_numTokens = out[1]
                #print("indexed_numTokens=", indexed_numTokens)
                numTokens_index = indexed_numTokens[0]
                numTokens = indexed_numTokens[1]
                with open(file, "r", encoding="utf-8") as f:
                    u.getSlidingWindowEmbeddings(f, numTokens)
            #r.getEmbeddingDataset(cache_rebuild = False)
            end = time.time()
            logger.info(f"finished rebuilding cache. elapsed = {end - start}")
        if args[2] == "pd":
            logger.info(f"rebuild PD cache (and the follow steps)")
            start = time.time()
            u = use_wrapper(model_filename, tokenizer_name, model_load = False)
            c = u.db.deleteCache("emb",
                              embFunc = u.getPersistenceDiagramEmbeddings
                              )
            """
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
            """
            iter = SourceFileIterator(u.data_top_dir, u.data_subdirs, u.numTokensList)
            for out in tqdm(iter):
                indexed_file = out[0]
                file_index = indexed_file[0]
                file = indexed_file[1]            
                indexed_numTokens = out[1]
                #print("indexed_numTokens=", indexed_numTokens)
                numTokens_index = indexed_numTokens[0]
                numTokens = indexed_numTokens[1]
                with open(file, "r", encoding="utf-8") as f:
                    u.getPersistenceDiagramEmbeddings(f, numTokens)
            
            end = time.time()
            logger.info(f"finished rebuilding cache. elapsed = {end - start}")
        if args[2] == "bottleneck":
            logger.info(f"rebuild bottleneck cache")
            start = time.time()
            u = use_wrapper(model_filename, tokenizer_name, model_load = False)
            #r.enable_bottleneck_cache = False
            c = u.db.deleteCache("dis",
                              simFunc = u.Bottleneck
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
