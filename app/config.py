import json
import os


class Config:

    def __init__(self, config_path="config.json"):

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def get(self, *keys, default=None):

        value = self.data

        try:
            for key in keys:
                value = value[key]

            return value

        except (KeyError, TypeError):
            return default