import unittest
from rwkv_runner import rwkv
import torch
import logging
from logging import config
from custom_formatter import CustomFormatter

config.fileConfig("logging.conf")

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


model_name = 'RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth'
tokenizer_name = '20B_tokenizer.json'


logger.info(f"unittest initialize")
r = rwkv(model_name, tokenizer_name)
text = "Hello, World"
tmpfilename = "tmp.txt"
with open(tmpfilename, "w") as f:
    f.write(text)

class TestRwkvRunner(unittest.TestCase):
    
    def test_init(self):
        logger.info(f"rwkv instance={r}")
        self.assertNotEqual(r, None)
    def test_getRwkvEmbeddings(self):
        numTokens = 3
        with open(tmpfilename, "r") as f:
            vec = r.getRwkvEmbeddings(f, numTokens)
            logger.info(f"vec={vec}")
            self.assertEqual(vec.shape, torch.Size([50277]))
    def test_getSlidingWindowEmbeddings(self):
        numTokens = 3
        with open(tmpfilename, "r") as f:
            vec = r.getSlidingWindowEmbeddings(f, numTokens)
            logger.info(f"vec={vec}")
            self.assertEqual(vec.shape, torch.Size([50275, 3]))
    def test_getPersistenceDiagramEmbeddings(self):
        numTokens = 3
        with open(tmpfilename, "r") as f:
            vec = r.getPersistenceDiagramEmbeddings(f, numTokens)
            logger.info(f"vec={vec}")
            self.assertEqual(vec.shape, torch.Size([2, 147349]))
    def test_getEmbeddingDataset(self):
        r.getEmbeddingDataset()
        
if __name__ == "__main__":
    unittest.main()

