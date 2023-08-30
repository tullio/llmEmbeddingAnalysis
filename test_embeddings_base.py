import unittest
from embeddings_base import embeddings_base
import numpy as np
import logging
from logging import config
config.fileConfig("logging.conf")

logger = logging.getLogger(__name__)
model_filename = "universal-sentence-encoder-multilingual-large_3"
tokenizer_name = '20B_tokenizer.json'
e = embeddings_base(model_filename, tokenizer_name, model_load = True)

text = "Hello, World"
tmpfilename = "tmp.txt"
with open(tmpfilename, "w") as f:
    f.write(text)


class TestEmbeddingsBase(unittest.TestCase):
    def test_init(self):
        logger.info(f"embeddings base instance={e}")
        self.assertNotEqual(e, None)

    def test_encoding(self):
        enc = e.encoding(text)
        print(enc)
        self.assertEqual(enc.ids, [12092, 13, 3645])

    def test_decoding(self):
        dec = e.decoding([12092, 13, 3645])
        print(dec)
        self.assertEqual(dec, text)

    def test_Bottleneck(self):
        e.enable_bottleneck_cache = True
        sim1 = e.BottleneckSim(np.array([[1, 2], [1, 3]]), np.array([[1, 2], [1, 3]]))
        logger.debug(f"most high={sim1}")
        self.assertEqual(sim1, 1)
        sim2 = e.BottleneckSim(np.array([[1, 2], [1, 3]]), np.array([[1, 2], [1, 4]]))
        logger.debug(f"middle={sim2}")
        sim3 = e.BottleneckSim(np.array([[1, 2], [1, 3]]), np.array([[1, 2], [1, 5]]))
        logger.debug(f"most low={sim3}")
        #np.testing.assert_almost_equal(sim, 0.3246, decimal=4)
        #np.testing.assert_almost_equal(sim2 - sim1, 0.3246, decimal=4)
        self.assertGreater(sim1, sim2)
        self.assertGreater(sim2, sim3)


if __name__ == "__main__":
    unittest.main()
