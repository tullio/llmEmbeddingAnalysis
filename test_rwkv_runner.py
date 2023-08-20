import unittest
from rwkv_runner import rwkv
import torch
import logging
from logging import config
from custom_formatter import CustomFormatter
import numpy as np
from tokenizers import Tokenizer
from rwkv_tokenizer import rwkv_tokenizer
import sys
import os
from source_file_iterator import SourceFileIterator
import matplotlib.pyplot as plt
import hashlib

np.set_printoptions(suppress=True)

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
        r.enable_rwkvemb_cache = False
        with open(tmpfilename, "r") as f:
            vec = r.getRwkvEmbeddings(f, numTokens)
            logger.info(f"vec={vec}")
            #self.assertEqual(vec.shape, torch.Size([50277]))
            self.assertEqual(vec.shape, torch.Size([12800]))
        numTokens = 1024
        filename1 = "data/carroll/Alice's Adventures in Wonderland by Lewis Carroll"
        filename2 = 'data/einstein/Relativity: The Special and General Theory by Albert Einstein'
        filename3 = 'data/lovecraft/The Dunwich Horror by H. P. Lovecraft'
        filename4 = 'data/carroll/Through the Looking-Glass by Lewis Carroll'
        with open(filename1, "r", encoding="utf-8") as f:
            emb1 = r.getRwkvEmbeddings(f, numTokens)


    def test_getSlidingWindowEmbeddings(self):
        r.enable_swemb_cache = False
        numTokens = 3
        with open(tmpfilename, "r") as f:
            vec = r.getSlidingWindowEmbeddings(f, numTokens)
            logger.info(f"vec={vec}")
            #self.assertEqual(vec.shape, torch.Size([50275, 3]))
            self.assertEqual(vec.shape, torch.Size([12798, 3]))
            
    def test_getPersistenceDiagramEmbeddings(self):
        numTokens = 3
        with open(tmpfilename, "r") as f:
            vec = r.getPersistenceDiagramEmbeddings(f, numTokens)
            logger.info(f"vec={vec}")
            #self.assertEqual(vec.shape, torch.Size([2, 147349]))
            self.assertEqual(vec.shape, torch.Size([2, 35742]))
    def test_getHeadPersistenceDiagramEmbeddings(self):
        np.set_printoptions(suppress=True)
        numTokens = 10
        with open(tmpfilename, "r") as f:
            vec = r.getHeadPersistenceDiagramEmbeddings(f, numTokens)
            logger.info(f"vec={vec}, shape={vec.shape}")
            self.assertEqual(vec.shape, torch.Size([2*r.topN]))
    def test_getEmbeddingDataset(self):
        # もとのテキストデータにETL加えて全部作り直すようなとき
        r.getEmbeddingDataset(cache_rebuild = True)
    def test_CosSim(self):
        sim = r.CosSim([1, 2], [2, 1])
        self.assertEqual(sim, 0.8)
        sim = r.CosSim(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0]))
        np.testing.assert_almost_equal(sim, 0.8, decimal=4)
        file1 = "./data/carroll/Alice's Adventures in Wonderland by Lewis Carroll"
        with open(file1, "r") as f1:
            vec1 = r.getEmbeddings(f1, 1024)
        
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
        r.enable_bottleneck_cache = True
        sim1 = r.BottleneckSim(np.array([[1, 2], [1, 3]]), np.array([[1, 2], [1, 3]]))
        print("most high", sim1)
        self.assertEqual(sim1, 1)
        sim2 = r.BottleneckSim(np.array([[1, 2], [1, 3]]), np.array([[1, 2], [1, 4]]))
        print("middle", sim2)
        sim3 = r.BottleneckSim(np.array([[1, 2], [1, 3]]), np.array([[1, 2], [1, 5]]))
        print("most low", sim3)
        #np.testing.assert_almost_equal(sim, 0.3246, decimal=4)
        #np.testing.assert_almost_equal(sim2 - sim1, 0.3246, decimal=4)
        self.assertGreater(sim1, sim2)
        self.assertGreater(sim2, sim3)


        
    def test_get_simMatrix(self):
        r.enable_simmat_cache = False
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        simMatrix = r.get_simMatrix(r.CosSim, [[1, 2], [1, 2], [2, 1]])
        np.testing.assert_array_equal(simMatrix, np.array([[1.0, 1.0, 0.8], [1.0, 1.0, 0.8], [0.8, 0.8, 1.0]]))
        print("simMatrix=", simMatrix)
        r.simMatrixPlot(fig, ax, simMatrix)

    def test_simMatByCos(self):
        r.enable_simmat_cache = False
        sim = r.simMat(r.getRwkvEmbeddings, r.CosSim, 2048)
        print(sim)

    def test_plotSimMatByCos(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        mat = r.simMat(r.getRwkvEmbeddings, r.CosSim, 2048)
        r.simMatrixPlot(fig, ax, mat)

    def test_simMatByJFIP(self):
        r.enable_simmat_cache = False
        sim = r.simMat(r.getRwkvEmbeddings, r.JFIP, 1024)
        print(sim)

    def test_PDsimMatByJFIP(self):
        r.enable_simmat_cache = False
        sim = r.simMat(r.getHeadPersistenceDiagramEmbeddings, r.JFIP, 1024)
        print(sim)

    def test_PDsimMatByCos(self):
        r.enable_simmat_cache = False        
        sim = r.simMat(r.getHeadPersistenceDiagramEmbeddings, r.CosSim, 2048)
        print(sim)

    def test_PDsimMatByBottleneck(self):
        r.enable_simmat_cache = False
        sim = r.simMat(r.getHeadPersistenceDiagramEmbeddings, r.BottleneckSim, 2048)
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

    def test_getAllScore(self):
        scoreArray = r.getAllScores()
        print(scoreArray)


    def test_encoding(self):
        text = """
            Alice was beginning to get very tired of sitting by her sister on the
            bank, and of having nothing to do: once or twice she had peeped into
            the book her sister was reading, but it had no pictures or
            conversations in it, “and what is the use of a book,” thought Alice
            “without pictures or conversations?”
               """
        print(text)
        tokenIds = r.encoding(text)
        print(tokenIds)
        print(r.tokenizer.__class__)
        print(r.tokenizer.__class__.__name__)


    def test_getCacheKey(self):
        fileName = "data/carroll/Alice's Adventures in Wonderland by Lewis Carroll"
        with open(fileName, "r") as f:
            #### old version check
            
            key = r.getCacheKey("rwkvemb", None, f, 1024, None, None)
            print(key)
            self.assertEqual(key,
              ":file=data/carroll/Alice's Adventures in Wonderland by Lewis Carroll"
             +":tokens=1024:rwkvemb"
                             )
            key = r.getCacheKey("rwkvemb", None, None, 1024, r.getRwkvEmbeddings,
                                r.CosSim)
            print(key)
            self.assertEqual(key,
              ":getEmb=getRwkvEmbeddings:simFunc=CosSim:tokens=1024:rwkvemb"
                             )
            #### new version check
            
            key = r.getCacheKey("rwkvemb", r.tokenizer, f, 1024, None, None)
            print(key)
            self.assertEqual(key,
              f":tokenizer={r.tokenizer.__class__.__name__}:"
             +"file=data/carroll/Alice's Adventures in Wonderland by Lewis Carroll"
             +":tokens=1024:rwkvemb"
                             )
            key = r.getCacheKey("rwkvemb", r.tokenizer, None, 1024,
                                r.getRwkvEmbeddings,
                                r.CosSim)
            print(key)
            self.assertEqual(key,
              f":tokenizer={r.tokenizer.__class__.__name__}"
             +":getEmb=getRwkvEmbeddings:simFunc=CosSim:tokens=1024:rwkvemb"
                             )
    def test_getRwkvEmbeddingsWithCache(self):
        if not hasattr(r, "model"):
            print("model_load should be True")
            sys.exit()
        print("model=", r.model)
        numTokens = 3
        with open(tmpfilename, "r") as f:
            logger.info(f"raw tmp file test={f.readlines()}")
            f.seek(0)
            r.tokenizer = Tokenizer.from_file(tokenizer_name)
            vec = r.getRwkvEmbeddings(f, numTokens)
            logger.info(f"tokenizer={r.tokenizer}, vec={vec}")
            #self.assertEqual(vec.shape, torch.Size([50277]))
            self.assertEqual(vec.shape, torch.Size([12800]))
            r.tokenizer = rwkv_tokenizer("rwkv_vocab_v20230424.txt")
            vec = r.getRwkvEmbeddings(f, numTokens)
            logger.info(f"tokenizer={r.tokenizer}, vec={vec}")            
            #self.assertEqual(vec.shape, torch.Size([50277]))
            self.assertEqual(vec.shape, torch.Size([12800]))

    def test_embedding(self):
        numTokens = 1024 # dummy
        dirname = "test_embedding_tmp"
        #getEmbeddings = r.getRwkvEmbeddings
        getEmbeddings = r.getHeadPersistenceDiagramEmbeddings
        r.enable_rwkvemb_cache = False
        r.enable_swemb_cache = False
        r.enable_pdemb_cache = False
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        text1 = """
                    Alice was beginning to get very tired of sitting by her sister on the
                    bank, and of having nothing to do: once or twice she had peeped into
                    the book her sister was reading, but it had no pictures or
                    conversations in it, “and what is the use of a book,” thought Alice
                    “without pictures or conversations?”
                """
        with open(f"{dirname}/text1", "w", encoding="utf-8") as f:
            f.write(text1)
        with open(f"{dirname}/text1", "r", encoding="utf-8") as f:
            emb1 = getEmbeddings(f, numTokens)

        text2 = """
            In your schooldays most of you who read this book made acquaintance
            with the noble building of Euclid’s geometry, and you remember—perhaps
            with more respect than love—the magnificent structure, on the lofty
            staircase of which you were chased about for uncounted hours by
            conscientious teachers. By reason of our past experience, you would
            certainly regard everyone with disdain who should pronounce even the
            most out-of-the-way proposition of this science to be untrue. But
            perhaps this feeling of proud certainty would leave you immediately if
            some one were to ask you: “What, then, do you mean by the assertion
            that these propositions are true?” Let us proceed to give this question
            a little consideration.

                """    
        with open(f"{dirname}/text2", "w", encoding="utf-8") as f:
            f.write(text2)
        with open(f"{dirname}/text2", "r", encoding="utf-8") as f:
            emb2 = getEmbeddings(f, numTokens)            
        text3 = """
            Alice was not a bit hurt, and she jumped up on to her feet in a moment:
            she looked up, but it was all dark overhead; before her was another
            long passage, and the White Rabbit was still in sight, hurrying down
            it. There was not a moment to be lost: away went Alice like the wind,
            and was just in time to hear it say, as it turned a corner, “Oh my ears
            and whiskers, how late it’s getting!” She was close behind it when she
            turned the corner, but the Rabbit was no longer to be seen: she found
            herself in a long, low hall, which was lit up by a row of lamps hanging
            from the roof.
                """
        with open(f"{dirname}/text3", "w", encoding="utf-8") as f:
            f.write(text3)
        with open(f"{dirname}/text3", "r", encoding="utf-8") as f:
            emb3 = getEmbeddings(f, numTokens)

        all_emb = [emb1, emb2, emb3]
        simMat = []
        for i in all_emb:
            for j in all_emb:
                sim = r.CosSim(i, j)
                simMat.append(sim)
        print(simMat)
        simMat = torch.tensor(simMat).reshape(3, 3)
        logger.info(f"(similarity matrix={simMat}")

    def test_embedding_small(self):
        numTokens = 1 # dummy
        dirname = "test_embedding_tmp"
        #getEmbeddings = r.getRwkvEmbeddings
        getEmbeddings = r.getHeadPersistenceDiagramEmbeddings
        r.enable_rwkvemb_cache = False
        r.enable_swemb_cache = False
        r.enable_pdemb_cache = False
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        text1 = """
                    Alice was beginning to get very tired of sitting by her sister on the
                    bank
                """
        with open(f"{dirname}/text1", "w", encoding="utf-8") as f:
            f.write(text1)
        with open(f"{dirname}/text1", "r", encoding="utf-8") as f:
            emb1 = getEmbeddings(f, numTokens)

        text2 = """
            In your schooldays most of you who read this book made acquaintance
            with the noble building of Euclid’s geometry, and you remember—perhaps
            with more respect than love—the magnificent structure, on the lofty
            staircase of which you were chased about for uncounted hours by
            conscientious teachers
                """    
        with open(f"{dirname}/text2", "w", encoding="utf-8") as f:
            f.write(text2)
        with open(f"{dirname}/text2", "r", encoding="utf-8") as f:
            emb2 = getEmbeddings(f, numTokens)            
        text3 = """
            Alice was not a bit hurt, and she jumped up on to her feet in a moment
                """
        with open(f"{dirname}/text3", "w", encoding="utf-8") as f:
            f.write(text3)
        with open(f"{dirname}/text3", "r", encoding="utf-8") as f:
            emb3 = getEmbeddings(f, numTokens)

        all_emb = [emb1, emb2, emb3]
        simMat = []
        for i in all_emb:
            for j in all_emb:
                sim = r.CosSim(i, j)
                simMat.append(sim)
        print(simMat)
        simMat = torch.tensor(simMat).reshape(3, 3)
        print(simMat)

    def test_embedding_large(self):
        numTokens = 1024 # dummy
        dirname = "test_embedding_tmp"
        #getEmbeddings = r.getRwkvEmbeddings
        getEmbeddings = r.getHeadPersistenceDiagramEmbeddings
        #r.enable_rwkvemb_cache = False
        #r.enable_swemb_cache = False
        #r.enable_pdemb_cache = False
        filename1 = "data/carroll/Alice's Adventures in Wonderland by Lewis Carroll"
        filename2 = 'data/einstein/Relativity: The Special and General Theory by Albert Einstein'
        filename3 = 'data/lovecraft/The Dunwich Horror by H. P. Lovecraft'
        filename4 = 'data/carroll/Through the Looking-Glass by Lewis Carroll'
        with open(filename1, "r", encoding="utf-8") as f:
            emb1 = getEmbeddings(f, numTokens)
        with open(filename2, "r", encoding="utf-8") as f:
            emb2 = getEmbeddings(f, numTokens)            
        with open(filename3, "r", encoding="utf-8") as f:
            emb3 = getEmbeddings(f, numTokens)
        with open(filename4, "r", encoding="utf-8") as f:
            emb4 = getEmbeddings(f, numTokens)

        all_emb = [emb1, emb2, emb3, emb4]
        simMat = []
        for i in all_emb:
            for j in all_emb:
                sim = r.CosSim(i, j)
                #sim = r.BottleneckSim(i, j)
                simMat.append(sim)
        print(simMat)
        simMat = torch.tensor(simMat).reshape(len(all_emb), len(all_emb))
        print(simMat)


    def test_count_source_files(self):
        r.count_source_files()

    def test_describeEmbeddings(self):
        r.enable_rwkvemb_cache = False
        text = "Hello, World"
        tmpdir = "tmp"
        if not os.path.exists(tmpdir):
            os.mkdir(tmpdir)
        tmpfilename = f"{tmpdir}/tmp.txt"
        with open(tmpfilename, "w") as f:
           f.write(text)
        r.data_top_dir = "."
        r.data_subdirs = [tmpdir]
        r.numTokensList = [1024]
        iter = SourceFileIterator(r.data_top_dir, r.data_subdirs,
                                  r.numTokensList)
        out = next(iter, None)
        head = 5
        count = 0
        while out and count < head:
            print(f"out={out}")
            indexed_filename = out[0]
            filename = indexed_filename[1]
            numTokens = out[1][1]
            with open(filename, "r", encoding="utf-8") as f:
                emb = r.getRwkvEmbeddings(f, numTokens)
                r.describeEmbeddings(emb)
            out = next(iter, None)
            count += 1

        r.close()
    def test_all_simMatrixPlot(self):
        r.all_simMatrixPlot()

    def test_normalize(self):
        x = np.array([[0, 1, 2], [3, 5, 7]])
        y = r.normalize(x)
        np.testing.assert_array_equal(y, np.array([[0, 0.5, 1], [0, 0.5, 1]]))

    def test_regularize(self):
        x = np.array([[0, 1, 2], [3, 5, 7]])
        y = r.regularize(x)
        np.testing.assert_array_equal(y, np.array([[-1.224745,  0.,  1.224745],
                                                   [-1.224745,  0.,  1.224745]]))

    def test_hash(self):
        numTokens = 1024
        dirname = "test_embedding_tmp"
        #getEmbeddings = r.getRwkvEmbeddings
        getEmbeddings = r.getHeadPersistenceDiagramEmbeddings
        #r.enable_rwkvemb_cache = False
        #r.enable_swemb_cache = False
        #r.enable_pdemb_cache = False
        filename1 = "data/carroll/Alice's Adventures in Wonderland by Lewis Carroll"
        filename2 = 'data/einstein/Relativity: The Special and General Theory by Albert Einstein'
        filename3 = 'data/lovecraft/The Dunwich Horror by H. P. Lovecraft'
        filename4 = 'data/carroll/Through the Looking-Glass by Lewis Carroll'
        with open(filename1, "r", encoding="utf-8") as f:
            emb1 = getEmbeddings(f, numTokens)
        with open(filename2, "r", encoding="utf-8") as f:
            emb2 = getEmbeddings(f, numTokens)            
        with open(filename3, "r", encoding="utf-8") as f:
            emb3 = getEmbeddings(f, numTokens)
        with open(filename4, "r", encoding="utf-8") as f:
            emb4 = getEmbeddings(f, numTokens)
        r.hash_algorithm.update(emb1.tobytes())
        hash1 = r.hash_algorithm.hexdigest()
        r.hash_algorithm.update(emb2.tobytes())
        hash2 = r.hash_algorithm.hexdigest()
        print(hash1)
        print(hash2)

    def test_getSimMatWithoutCache(self):
        getEmbFunc = r.getHeadPersistenceDiagramEmbeddings
        simFunc = r.BottleneckSim
        numTokens = 1024
        simMat = r.getSimMatWithoutCache(getEmbFunc, simFunc, numTokens)
        print("simMat=", simMat)

    def test_cache(self):

        #keys = r.listCache(None, r.getSlidingWindowEmbeddings, None, 1024)
        keys = r.listCache(None, r.getSlidingWindowEmbeddings, None, None)
        logger.info(f"hit keys={keys}")

        
        sys.exit()
        
        key1 = r.getCacheKey("test", None, None, 1024,
                             r.getRwkvEmbeddings, r.CosSim)
        key2 = r.getCacheKey("test", None, None, 1024,
                             r.getRwkvEmbeddings, r.BottleneckSim)
        logger.info(f"test key={key1}")
        val1 = "dummy for test1"
        val2 = "dummy for test2"

        # test 1 全部消す
        r.setDb(key1, val1)
        r.setDb(key2, val2)
        self.assertEqual(r.getDb(key1), val1)
        self.assertEqual(r.getDb(key2), val2)

        keys = r.listCache("test", r.getRwkvEmbeddings, None, 1024)
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys, ['rwkv:RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth::getEmb=getRwkvEmbeddings:simFunc=BottleneckSim:tokens=1024:test', 'rwkv:RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth::getEmb=getRwkvEmbeddings:simFunc=CosSim:tokens=1024:test'])
        c = r.deleteCache("test", r.getRwkvEmbeddings, None, 1024)
        self.assertEqual(c, 2)
        keys = r.listCache("test", r.getRwkvEmbeddings, None, 1024)
        self.assertEqual(len(keys), 0)

        # test2 一部消す
        r.setDb(key1, val1)
        r.setDb(key2, val2)
        self.assertEqual(r.getDb(key1), val1)
        self.assertEqual(r.getDb(key2), val2)

        keys = r.listCache("test", r.getRwkvEmbeddings, None, 1024)
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys, ['rwkv:RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth::getEmb=getRwkvEmbeddings:simFunc=BottleneckSim:tokens=1024:test', 'rwkv:RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth::getEmb=getRwkvEmbeddings:simFunc=CosSim:tokens=1024:test'])
        keys = r.listCache("test", None, r.CosSim, 1024)
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys, ['rwkv:RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth::getEmb=getRwkvEmbeddings:simFunc=CosSim:tokens=1024:test'])

        c = r.deleteCache("test", None, r.CosSim, 1024)
        self.assertEqual(c, 1)
        keys = r.listCache("test", r.getRwkvEmbeddings, None, 1024)
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys, ['rwkv:RWKV-4-Raven-3B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230429-ctx4096.pth::getEmb=getRwkvEmbeddings:simFunc=BottleneckSim:tokens=1024:test'])
        
        
if __name__ == "__main__":
    unittest.main()

