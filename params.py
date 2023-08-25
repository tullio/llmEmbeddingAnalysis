import toml


class params():
    config = toml.load("config.toml")
    cache_filename = config["cache"]["filename"]
