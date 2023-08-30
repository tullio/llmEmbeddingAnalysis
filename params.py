import toml

import logging
from logging import config

config.fileConfig("logging.conf", disable_existing_loggers = False)

logger = logging.getLogger(__name__)

class Params():
    def __init__(self, filename):
        self.filename = filename

        #config = toml.load("config.toml")
        self.config = toml.load(self.filename)
        self.cache_filename = self.config["cache"]["filename"]
