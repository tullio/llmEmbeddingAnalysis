from gtda.time_series import TakensEmbedding
import hashlib

class embedding_base():
    def __init__(self, model_filename, tokenizer_filename, model_load = True):
        embedding_dimension_periodic = 3
        #embedding_time_delay_periodic = 10
        self.embedding_time_delay_periodic = 16
        #self.embedding_time_delay_periodic = 20
        stride = 1
        self.sw_embedder = TakensEmbedding(
            #parameters_type="fixed",
            #n_jobs=2,
            time_delay=self.embedding_time_delay_periodic,
            dimension=embedding_dimension_periodic,
            stride=stride,
         )

        # cache disableはキャッシュを使わない，にしてたけど，
        # キャッシュから読み出さない，にしたほうが良さそうだ
        self.enable_rwkvemb_cache = True
        self.enable_swemb_cache = True
        self.enable_pdemb_cache = True

        self.enable_bottleneck_cache = True
        self.enable_wasserstein_cache = True
        self.enable_simmat_cache = True
        self.enable_pdemb_visualize = True

        self.sigma = 100

        #self.pdemb_postfunc = self.identical
        #self.pdemb_postfunc = self.normalize
        self.pdemb_postfunc = self.scaling
        self.scaling_const = 20
        self.scaling_const0 = 20
        self.data_top_dir = "./data"
        self.data_subdirs = ["carroll", "einstein", "lovecraft"]
        self.numTokensList = [1024, 2048, 4096, 8192, 16384]
        #self.numTokensList = [1024]
        #self.topN = 10 # PDから取るベクトルの数
        self.topN = 1 # PDから取るベクトルの数

        # ディレクトリ情報から自動生成したい
        self.cl = [0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2]  # インデックスが属するクラスタID情報


        # キャッシュのキーに使うハッシュ関数
        self.hash_algorithm = hashlib.sha256
