import os
import json
import pprint

class ConfigLoader:
    config_formats = ['json']
    config_root = os.path.dirname(os.path.dirname(__file__))


    def __init__(self, fn):
        assert fn.endswith('.json'), 'format implemented: '+' | '.join(self.config_formats)
        with open(os.path.join(self.config_root, fn)) as f:
            self.payload = json.load(f)

    def get_json(self):
        return self.payload