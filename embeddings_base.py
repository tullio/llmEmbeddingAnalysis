from gtda.time_series import TakensEmbedding
import hashlib
import logging
from tokenizers import Tokenizer
import time
import homcloud.interface as hc 
import numpy as np
from torch.nn.functional import cosine_similarity as cos_sim
from logging import config
config.fileConfig("logging.conf")

logger = logging.getLogger(__name__)

print("embeddings_base_logger=", logger)


class embeddings_base:
    def __init__(self, model_filename, tokenizer_filename, model_load = True):
        embedding_dimension_periodic = 3
        #embedding_time_delay_periodic = 10
        self.embedding_time_delay_periodic = 16
        #self.embedding_time_delay_periodic = 20
        stride = 1
        self.sw_embedder = TakensEmbedding(
            #parameters_type="fixed",
            #n_jobs=2,
            time_delay=self.embedding_time_delay_periodic,
            dimension=embedding_dimension_periodic,
            stride=stride,
         )
        #self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_filename)
        #self.tokenizer = Tokenizer.from_file(tokenizer_filename)
        self.tokenizer = Tokenizer.from_file(tokenizer_filename)
        print(self.tokenizer.__class__)
        #self.tokenizer = rwkv_tokenizer("rwkv_vocab_v20230424.txt")

        # cache disableはキャッシュを使わない，にしてたけど，
        # キャッシュから読み出さない，にしたほうが良さそうだ
        self.enable_rwkvemb_cache = True
        self.enable_useemb_cache = True

        self.enable_swemb_cache = True
        self.enable_pdemb_cache = True

        self.enable_bottleneck_cache = True
        self.enable_wasserstein_cache = True
        self.enable_simmat_cache = True
        self.enable_pdemb_visualize = True

        self.sigma = 1000

        self.pdemb_postfunc = self.identical
        #self.pdemb_postfunc = self.normalize
        #self.pdemb_postfunc = self.scaling
        self.scaling_const = 100
        #self.scaling_const0 = 20
        self.data_top_dir = "./data"
        self.data_subdirs = ["carroll", "einstein", "lovecraft"]
        #self.numTokensList = [1024, 2048, 4096, 8192, 16384]
        # スナーク狩りが7992トークンしかない
        self.numTokensList = [1024, 2048, 4096]
        #self.numTokensList = [1024]
        #self.topN = 1000 # PDから取るベクトルの数
        self.topN = 10 # PDから取るベクトルの数

        # ディレクトリ情報から自動生成したい
        self.cl = [0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2]  # インデックスが属するクラスタID情報


        # キャッシュのキーに使うハッシュ関数
        self.hash_algorithm = hashlib.sha256
        self.db = None

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

    def encoding(self, text):
        enc = self.tokenizer.encode(text)
        tokenIds = enc.ids
        tokens = enc.tokens
        logger.debug(f"token Ids[0:30]={tokenIds[0:30]}")
        logger.debug(f"tokens[0:30]={tokens[0:30]}")
        #return tokenIds
        return enc

    def decoding(self, tokens):
        dec = self.tokenizer.decode(tokens)
        logger.debug(f"input tokens[0:30]={tokens[0:30]}")
        logger.debug(f"output string[0:30]={dec[0:30]}")
        return dec

    # get embeddings from the file descriptor of the output of the SourceFileIterator
    def getEmbeddingsFromFD(self, fd, getEmbFunc):
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
            if self.enable_bottleneck_cache and self.db is not None:
                key = self.db.getCacheKey("dis",
                                       postfunc = self.pdemb_postfunc,
                                       simFunc = self.Bottleneck,
                                       list1 = list1,
                                       list2 = list2
                                       )

                val = self.db.getDb(key)
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
                    self.db.setDb(key, dis)
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
                if self.db is not None:
                    self.db.setDb(key, dis)
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
