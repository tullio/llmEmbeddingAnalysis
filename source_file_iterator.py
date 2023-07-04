import os
import glob
class SourceFileIterator():
    # data_top_dir: string
    # data_sub_dir: list(string)
    # numTokensList: list(int)
    def __init__(self, data_top_dir, data_sub_dir, numTokensList):
        self.data_top_dir = data_top_dir
        self.data_sub_dir = data_sub_dir
        self.numTokensList = numTokensList

        # 初期設定として，file_iterとnumTokens_iterを準備
        # numTokens_iter以外は，最初の一つは取り出された状態
        self.init_dir()
        dir = next(self.subdir_iter) # get first one
        self.init_file(dir)
         
        self.init_numTokens()
        self.current_file = next(self.file_iter) # get first one
        #self.current_numTokens = next(self.numTokens_iter)# get first one
    def __iter__(self):
        return self

    def init_dir(self):
        subdir_path = [os.path.join(self.data_top_dir, subdir) for subdir in self.data_sub_dir]
        self.subdir_iter = iter(subdir_path)
    def init_file(self, dir):
        #files = os.listdir(dir)
        files = glob.glob(f"{dir}/*")
        indexed_files = enumerate(files)
        self.file_iter = iter(indexed_files)
    def init_numTokens(self):
        indexed_numTokensList = enumerate(self.numTokensList)
        self.numTokens_iter = iter(indexed_numTokensList)
        
    def __next__(self):
        out = None
        indexed_numTokens = next(self.numTokens_iter, None)
        if indexed_numTokens:
            self.current_numTokens = indexed_numTokens
            out = (self.current_file, self.current_numTokens)
        else: # numTokensListを消費したので，次のファイルに移る
            self.init_numTokens()
            indexed_numTokens = next(self.numTokens_iter, None)
            self.current_numTokens = indexed_numTokens
            indexed_file = next(self.file_iter, None)
            if indexed_file:
                self.current_file = indexed_file
                out = (self.current_file, self.current_numTokens)
            else: # 両方消費したので，終わり
                raise StopIteration()
        return out
