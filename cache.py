import diskcache
import logging
from logging import config
import re
import sys
import time
from source_file_iterator import SourceFileIterator
import torch.multiprocessing as multiprocessing
from tqdm import tqdm
from params import Params

config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)


class Cache():

    def __init__(self, source_class, cache_filename,
                 model_filename, tokenizer, hash_algorithm):
        self.key_prefix = source_class.__class__.__name__ + \
            ':' + model_filename + \
            ':' + tokenizer.__class__.__name__
        print("key_prefix=", self.key_prefix)
        #filename = "rwkv_gutenberg.db"
        #self.db = shelve.open(filename)
        self.db = diskcache.Cache(cache_filename)
        self.hash_algorithm = hash_algorithm

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
        #if keyName == "emb" and embFunc == self.getHeadPersistenceDiagramEmbeddings:
        #    raise ValueError("use getPersistenceDiagramEmbeddings")

        #if keyName == "dis" and simFunc == self.BottleneckSim:
        #    raise ValueError("use Bottleneck")

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
                substring += f":hash1={hash1}"
            else:
                substring += f":[^:]*"
            if list2 is not None:
                hash2 = self.hash_algorithm(list2.tobytes()).hexdigest()
                substring += f":hash2={hash2}"
            else:
                substring += f":[^:]*"
        elif keyName == "all":
            substring += f".*"
        else:
            raise ValueError(f"Invalid tag:{keyName}")
        
        hit_keys = []
        #logger.debug(f"target key={substring}")
        count = 0
        for key in iter:
            #logger.debug(f"target record={key}")
            #print(re.search(substring, key))
            if re.search(substring, key):
                hit_keys.append(key)
            count += 1
        logger.debug(f"total number of items = {count}")
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
        #if keyName == "emb" and embFunc == self.getHeadPersistenceDiagramEmbeddings:
        #    raise ValueError("use getPersistenceDiagramEmbeddings")
        #if keyName == "dis" and simFunc == self.BottleneckSim:
        #    raise ValueError("use Bottleneck")
        
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

if __name__ == "__main__":
    logger.info(f"■■■■■■■■■■■■■■■■■■■■■■■ main")
    args = sys.argv
    print(args)
    if args[1] == "--list-cache":
        if args[2] == "rwkv":
            logger.info(f"list rwkv embedding cache")
            start = time.time()
            #filename = "rwkv_gutenberg"
            params = Params("config.toml")
            filename = params.cache_filename
            model_name = 'RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth'
            tokenizer_name = '20B_tokenizer.json'
            from rwkv_runner import rwkv
            r = rwkv(model_name, tokenizer_name, model_load = False)
            c = Cache(Cache, filename, model_name, r.tokenizer)
            keys = c.listCache("emb", embFunc = r.getRwkvEmbeddings)

            #filename = "cache_test"
            #r = rwkv(model_name, tokenizer_name, model_load = False)
            #keys = c.listCache("all")
            logger.info(f"target keys[0:3]={keys[0:3]}")
            for key in keys:
                logger.info(f"hit: {key}")
        elif args[2] == "use":
            logger.info(f"list use embedding cache")
            start = time.time()
            #from rwkv_runner import rwkv
            #r = rwkv(model_name, tokenizer_name, model_load = False)
            #c = Cache(Cache, filename, model_name, r.tokenizer)
            #keys = c.listCache("emb", embFunc = r.getRwkvEmbeddings)

            #filename = "cache_test"
            #filename = "use_gutenberg"
            params = Params("config-use.toml")
            filename = params.cache_filename

            model_name = 'RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth'
            tokenizer_name = '20B_tokenizer.json'
            #r = rwkv(model_name, tokenizer_name, model_load = False)
            from use_wrapper import use_wrapper
            u = use_wrapper(model_name, tokenizer_name, model_load = False)
            c = Cache(Cache, filename, model_name, u.tokenizer)
            keys = c.listCache("emb", embFunc = u.getUseEmbeddings)
            #keys = c.listCache("all")
            logger.info(f"target keys[0:3]={keys[0:3]}")
            for key in keys:
                logger.info(f"hit: {key}")

        elif args[2] == "pd":
            logger.info(f"list PD embedding cache")
            start = time.time()
            #from rwkv_runner import rwkv
            #r = rwkv(model_name, tokenizer_name, model_load = False)
            #c = Cache(Cache, filename, model_name, r.tokenizer)
            #keys = c.listCache("emb", embFunc = r.getRwkvEmbeddings)

            #filename = "cache_test"
            #filename = "use_gutenberg"
            params = Params("config-use.toml")
            filename = params.cache_filename

            model_name = 'RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth'
            tokenizer_name = '20B_tokenizer.json'
            #r = rwkv(model_name, tokenizer_name, model_load = False)
            from use_wrapper import use_wrapper
            u = use_wrapper(model_name, tokenizer_name, model_load = False)
            c = Cache(Cache, filename, model_name, u.tokenizer)
            keys = c.listCache("emb", embFunc = u.getPersistenceDiagramEmbeddings)
            #keys = c.listCache("all")
            logger.info(f"target keys[0:3]={keys[0:3]}")
            for key in keys:
                logger.info(f"hit: {key}")
