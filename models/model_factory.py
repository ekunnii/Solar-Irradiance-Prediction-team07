import typing
import datetime
import json

from models.dummy_model import DummyModel
from models.cnn2d import cnn2d
from models.resnet import resnet
from models.double_resnet import double_resnet
from models.cnn3d import cnn3d
from models.cnn_lstm import cnn_lstm
from models.double_cnn_lstm import double_cnn_lstm
import tensorflow as tf


class ModelFactory():
    """
    This is a model factory: it's responsible of building Models from based on the information that will be provided 
    at evaluation time. From this, the model should be ready to train. 

    Args:
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.
    """

    def __init__(self,
                 stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
                 target_time_offsets: typing.List[datetime.timedelta],
                 config: typing.Dict[typing.AnyStr, typing.Any],
                 ):
        self.stations = stations
        self.target_time_offsets = target_time_offsets
        self.config = config

        # Declare new models here and map builder function to it.
        self.models = {
            'DummyModel': DummyModel,
            'CNN2D': cnn2d,
            'cnn_lstm': cnn_lstm,
            'double_cnn_lstm': double_cnn_lstm,
            'pretrained_resnet': resnet,
            'resnet': resnet,
            'double_pretrained_resnet': double_resnet,
            'CNN3D': cnn3d,
        }

    def build(self, modelName):
        if modelName == 'pretrained_resnet':
            return self.models[modelName](self.target_time_offsets, pretrained=True)
        elif modelName == 'resnet':
            return self.models[modelName](self.target_time_offsets, pretrained=False)
        
        return self.models[modelName](self.target_time_offsets)

    def load_model_from_config(self):
        modelName = self.config.get("model_name") or "DummyModel"
        print("Loading {model}...")
        model = self.models[modelName](self.target_time_offsets)
        return model.load_config(model, self.config)
    

