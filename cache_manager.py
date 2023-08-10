from params import params
import shelve


class CacheManager():
    def __init__(self):
        filename = params.cache_filename
        self.db = shelve.open(filename)

    def listKeys(self):
        keys = self.db.keys()
        return keys
