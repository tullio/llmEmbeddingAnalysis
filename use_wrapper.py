from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

from rwkv.utils import PIPELINE, PIPELINE_ARGS

from logging import config

from source_file_iterator import SourceFileIterator

config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)
import diskcache

import embeddings_base

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]

tokenizer_name = '20B_tokenizer.json'
multiprocessing.set_start_method("spawn", force=True)

class use_wrapper(embeddings_base):
    def __init__(self, model_filename, tokenizer_filename, model_load = True):
        super.__init__(self, model_filename, tokenizer_filename, model_load)
        logger.info(f"initializing USE")
        self.model_filename = model_filename
        if model_load:
            self.model = hub.load(module_url)
            # self.model = RWKV(model=model_filename, strategy='cuda fp16i8')
            self.pipeline = PIPELINE(self.model, tokenizer_filename) # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
            print ("module %s loaded" % module_url)


        #self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_filename)
        #self.tokenizer = Tokenizer.from_file(tokenizer_filename)
        self.tokenizer = Tokenizer.from_file(tokenizer_filename)
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
    
    def embed(input):
        return model(input)

if __name__ == "__main__":
    args = sys.argv
    model_filename = "universal-sentence-encoder-multilingual-large_3"
    u = use_wrapper(model_name, tokenizer_name, model_load = False)
    print(u)
