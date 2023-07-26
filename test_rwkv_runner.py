import unittest
from rwkv_runner import rwkv
import torch
import logging
from logging import config
from custom_formatter import CustomFormatter
import numpy as np

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
r = rwkv(model_name, tokenizer_name, model_load = False)
#r = rwkv(model_name, tokenizer_name, model_load = True)
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
    def test_getHeadPersistenceDiagramEmbeddings(self):
        np.set_printoptions(suppress=True)
        numTokens = 3
        with open(tmpfilename, "r") as f:
            vec = r.getHeadPersistenceDiagramEmbeddings(f, numTokens)
            logger.info(f"vec={vec}, shape={vec.shape}")
            self.assertEqual(vec.shape, torch.Size([2*r.topN]))
    def test_getEmbeddingDataset(self):
        # もとのテキストデータにETL加えて全部作り直すようなとき
        r.getEmbeddingDataset(cache_rebuild = False)
    def test_getCosineSimilarity(self):
        sim = r.CosSim([1, 2], [2, 1])
        self.assertEqual(sim, 0.8)
        sim = r.CosSim(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0]))
        np.testing.assert_almost_equal(sim, 0.8, decimal=4)
    def test_FIP(self):
        sim = r.FIP(np.array([1, 2]), np.array([2, 1]))
        print(sim)
        self.assertEqual(sim, 4)
    def test_JFIP(self):
        sim = r.JFIP(np.array([1, 2]), np.array([2, 1]))
        print(sim)
        self.assertEqual(sim, 1)
        sim = r.JFIP(np.array([1, 2]), np.array([-2, 1]))
        print(sim)
        np.testing.assert_almost_equal(sim, 0.2195, decimal=4)

    def test_Bottleneck(self):
        sim1 = r.Bottleneck(np.array([[1, 2], [1, 3]]), np.array([[1, 2], [1, 3]]))
        print("most high", sim1)
        self.assertEqual(sim1, 1)
        sim2 = r.Bottleneck(np.array([[1, 2], [1, 3]]), np.array([[1, 2], [1, 4]]))
        print("middle", sim2)
        sim3 = r.Bottleneck(np.array([[1, 2], [1, 3]]), np.array([[1, 2], [1, 5]]))
        print("most low", sim3)
        #np.testing.assert_almost_equal(sim, 0.3246, decimal=4)
        #np.testing.assert_almost_equal(sim2 - sim1, 0.3246, decimal=4)
        self.assertGreater(sim1, sim2)
        self.assertGreater(sim2, sim3)


        
    def test_get_simMatrix(self):
        simMatrix = r.get_simMatrix(r.CosSim, [[1, 2], [1, 2], [2, 1]])
        np.testing.assert_array_equal(simMatrix, np.array([[1.0, 1.0, 0.8], [1.0, 1.0, 0.8], [0.8, 0.8, 1.0]]))
        print("simMatrix=", simMatrix)
        r.simMatrixPlot(simMatrix)

    def test_simMatByCos(self):
        sim = r.simMat(r.getRwkvEmbeddings, r.CosSim, 2048)
        print(sim)

    def test_plotSimMatByCos(self):
        mat = r.simMat(r.getRwkvEmbeddings, r.CosSim, 2048)
        r.simMatrixPlot(mat)

    def test_simMatByJFIP(self):
        sim = r.simMat(r.getRwkvEmbeddings, r.JFIP, 2048)
        print(sim)

    def test_PDsimMatByJFIP(self):
        sim = r.simMat(r.getHeadPersistenceDiagramEmbeddings, r.JFIP, 2048)
        print(sim)

    def test_PDsimMatByCos(self):
        sim = r.simMat(r.getHeadPersistenceDiagramEmbeddings, r.CosSim, 2048)
        print(sim)

    def test_PDsimMatByBottleneck(self):
        sim = r.simMat(r.getHeadPersistenceDiagramEmbeddings, r.Bottleneck, 2048)
        print(sim)
    def test_getSimilarityMatrixDataset(self):
        #r.getSimilarityMatrixDataset(cache = False)
        r.getSimilarityMatrixDataset()

    def test_getSimDescWithoutCache(self):
        r.getSimDescWithoutCache(r.getRwkvEmbeddings, r.CosSim, 1024)

    def test_getScore(self):
        simMatrix = r.get_simMatrix(r.CosSim, [[1, 2], [1, 2], [2, 1]])
        np.testing.assert_array_equal(simMatrix, np.array([[1.0, 1.0, 0.8], [1.0, 1.0, 0.8], [0.8, 0.8, 1.0]]))
        print("simMatrix=", simMatrix)
        r.cl = [0, 0, 1]
        score = r.getScore(simMatrix)
        print(score)

        simMatrix = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        score = r.getScore(simMatrix)
        self.assertEqual(score, 1.0)

        simMatrix = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        score = r.getScore(simMatrix)
        self.assertEqual(score, 0.0)

        # 1つだめ
        simMatrix = np.array([[1.0, 1.0, 0.6], [1.0, 1.0, 0.0], [0.6, 0.0, 1.0]])
        score = r.getScore(simMatrix)
        print(score)
        np.testing.assert_almost_equal(score, 0.778, decimal=3)

        # 2つだめ
        simMatrix = np.array([[1.0, 4.0, 0.6], [0.4, 1.0, 0.0], [0.6, 0.0, 1.0]])
        score = r.getScore(simMatrix)
        print(score)
        np.testing.assert_almost_equal(score, 0.667, decimal=3)

        # 3つだめ
        # 対角は固定とすると，これが最低スコア
        simMatrix = np.array([[1.0, 4.0, 0.6], [0.4, 1.0, 0.6], [0.6, 0.6, 1.0]])
        score = r.getScore(simMatrix)
        print(score)
        np.testing.assert_almost_equal(score, 0.444, decimal=3)



if __name__ == "__main__":
    unittest.main()

