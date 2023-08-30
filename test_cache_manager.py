import unittest
import logging
from logging import config
from custom_formatter import CustomFormatter
import sys
import os
from cache_manager import CacheManager

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


logger.info(f"unittest initialize")


cm = CacheManager()


class TestCacheManager(unittest.TestCase):

    def test_listKeys(self):
        print(cm.listKeys())
        for i in cm.listKeys():
            print(i)
