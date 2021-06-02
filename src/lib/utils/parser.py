# -*- coding: utf-8 -*-
import os
import yaml


class YamlParser(dict):
    """
    This is yaml parser to access data with attribute or dict
    """
    def __init__(self, config_file=None, data=None):
        if data is None:
            data = {}
        for k, v in data.items():
            setattr(self, k, v)
        if config_file:
            self.load_yaml(config_file)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = self.__class__(data=value)
        super(YamlParser, self).__setitem__(name, value)
        super(YamlParser, self).__setattr__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None):
        data = e or dict()
        for k, v in data.items():
            setattr(self, k, v)

    def load_yaml(self, config_file):
        assert config_file.endswith('yaml'), f'file is not yaml format : {config_file}'
        assert os.path.isfile(config_file), f'file not exist : "{config_file}"'
        with open(config_file, 'r') as fo:
            config_dict = yaml.safe_load(fo.read())
            self.update(config_dict)

    def merge_from_file(self, config_file):
        self.load_yaml(config_file)

if __name__ == '__main__':
    cfg = YamlParser(config_file='../configs/trtpose.yaml')
    cfg.merge_from_file('../configs/deepsort.yaml')
    data = {
        "a": "aval",
        "b": {
            "b1": {
                "b1a": "b1aval",
                "b1b": "b2aval",
                }
            }
        }
    data1 = YamlParser(data)
    print(cfg)
    print(data1)
