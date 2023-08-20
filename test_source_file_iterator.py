import unittest
from source_file_iterator import SourceFileIterator
import os

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
        out = next(iter)
        self.assertEqual(out, ((0, './data/carroll/Through the Looking-Glass by Lewis Carroll'), (0, 1024)))
        with open(out[0][1], "r") as f:
            print(f.name)
        self.assertEqual(next(iter), ((0, './data/carroll/Through the Looking-Glass by Lewis Carroll'), (1, 2048)))
        self.assertEqual(next(iter), ((1, "./data/carroll/Alice's Adventures in Wonderland by Lewis Carroll"), (0, 1024)))

        self.assertEqual(next(iter), ((1, "./data/carroll/Alice's Adventures in Wonderland by Lewis Carroll"), (1, 2048)))
        self.assertEqual(next(iter), ((2, "./data/carroll/Alice's Adventures Under Ground by Lewis Carroll"), (0, 1024)))
        self.assertEqual(next(iter), ((2, "./data/carroll/Alice's Adventures Under Ground by Lewis Carroll"), (1, 2048)))
        self.assertEqual(next(iter), ((0, './data/einstein/The Principle of Relativity by Albert Einstein and H. Minkowski'), (0, 1024)))
    def test_double_iter(self):
        iter1 = SourceFileIterator(data_top_dir, data_sub_dirs, numTokens)

        out1 = next(iter1)
        while out1:
            iter2 = SourceFileIterator(data_top_dir, data_sub_dirs, numTokens)
            out2 = next(iter2)
            while out2:
                print(f"1={out1}")
                print(f"2={out2}\n")
                out2 = next(iter2)
            out1 = next(iter1)

    def test_simpleiter(self):
        text = "Hello, World"
        tmpdir = "tmp"
        if not os.path.exists(tmpdir):
            os.mkdir(tmpdir)
        tmpfilename = f"{tmpdir}/tmp.txt"
        with open(tmpfilename, "w") as f:
           f.write(text)
        data_top_dir = "."
        data_subdirs = [tmpdir]
        numTokensList = [1024]
        iter = SourceFileIterator(data_top_dir, data_subdirs,
                                  numTokensList)
        out = next(iter)
        print(f"out={out}")
        out = next(iter)
        print(f"out={out}")

if __name__ == "__main__":
    unittest.main()
