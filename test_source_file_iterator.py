import unittest
from source_file_iterator import SourceFileIterator

data_top_dir = "./data"
data_sub_dirs = ["carroll", "einstein"]
numTokens = [1024, 2048]

class TestSourceFileIterator(unittest.TestCase):
    
    def test_init(self):
        iter = SourceFileIterator(data_top_dir, data_sub_dirs, numTokens)
        #print(iter)
        self.assertNotEqual(iter, None)
    def test_init_dir(self):
        iter = SourceFileIterator(data_top_dir, data_sub_dirs, numTokens)
        #print(iter)
        self.assertEqual(next(iter), ((0, 'Through the Looking-Glass by Lewis Carroll'), (0, 1024)))
        self.assertEqual(next(iter), ((0, 'Through the Looking-Glass by Lewis Carroll'), (1, 2048)))
        self.assertEqual(next(iter), ((1, "Alice's Adventures in Wonderland by Lewis Carroll"), (0, 1024)))

        self.assertEqual(next(iter), ((1, "Alice's Adventures in Wonderland by Lewis Carroll"), (1, 2048)))
        self.assertEqual(next(iter), ((2, "Alice's Adventures Under Ground by Lewis Carroll"), (0, 1024)))
        self.assertEqual(next(iter), ((2, "Alice's Adventures Under Ground by Lewis Carroll"), (1, 2048)))
        self.assertEqual(next(iter, None), None)

if __name__ == "__main__":
    unittest.main()
