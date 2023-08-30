import unittest
from use_wrapper import use_wrapper
import torch
import logging
from logging import config
config.fileConfig("logging.conf")

logger = logging.getLogger(__name__)

model_filename = "universal-sentence-encoder-multilingual-large_3"
tokenizer_name = '20B_tokenizer.json'
u = use_wrapper(model_filename, tokenizer_name, model_load = True)

text = "Hello, World"
tmpfilename = "tmp.txt"
with open(tmpfilename, "w") as f:
    f.write(text)

class TestUseWrapper(unittest.TestCase):
    def test_init(self):
        logger.info(f"use instance={u}")
        self.assertNotEqual(u, None)

    def test_getUseEmbeddings(self):
        numTokens = 3
        u.enable_rwkvemb_cache = False
        with open(tmpfilename, "r") as f:
            vec = u.getUseEmbeddings(f, numTokens)
            logger.info(f"vec[0:5]={vec[:, 0:5]}")
            #self.assertEqual(vec.shape, torch.Size([50277]))
            #self.assertEqual(vec.shape, torch.Size([12800]))
            #self.assertEqual(vec.shape, torch.Size([12800]))
            self.assertEqual(vec.shape, torch.Size([1, 512]))

if __name__ == "__main__":
    unittest.main()

