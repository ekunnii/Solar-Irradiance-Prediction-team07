import typing
import datetime
import json

from models.dummy_model import DummyModel
from models.cnn2d import cnn2d
from models.resnet import resnet
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
            "DummyModel": self.BuildDummyModel,
            "CNN2D": self.BuildCNN2DModel,
            'pretrained_resnet': self.BuildResnet
        }

    def build(self, modelName):
        return self.models[modelName]()

    def BuildDummyModel(self) -> tf.keras.Model:
        """
        A model example to test out workflow.

        Returns:
            A ``tf.keras.Model`` object that can be used to generate new GHI predictions given imagery tensors.
        """
        return DummyModel(self.target_time_offsets)

    def BuildCNN2DModel(self) -> tf.keras.Model:
        """
        A model example to test out workflow.

        Returns:
            A ``tf.keras.Model`` object that can be used to generate new GHI predictions given imagery tensors.
        """
        return cnn2d(self.target_time_offsets)

    def BuildResnet(self) -> tf.keras.Model:
        """
        Pre-trained resnet50

        Returns:
            A ``tf.keras.Model`` object that can be used to generate new GHI predictions given imagery tensors.
        """
        return resnet(self.target_time_offsets)