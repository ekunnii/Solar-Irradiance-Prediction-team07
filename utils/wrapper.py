# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=protected-access
"""Wrapper layers: layers that augment the functionality of another layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import _standardize_args
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect



class Wrapper(Layer):
  """Abstract wrapper base class.

  Wrappers take another layer and augment it in various ways.
  Do not use this class as a layer, it is only an abstract base class.
  Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.

  Arguments:
    layer: The layer to be wrapped.
  """

  def __init__(self, layer, **kwargs):
    assert isinstance(layer, Layer)
    self.layer = layer
    super(Wrapper, self).__init__(**kwargs)

  def build(self, input_shape=None):
    if not self.layer.built:
      self.layer.build(input_shape)
      self.layer.built = True
    self.built = True

  @property
  def activity_regularizer(self):
    if hasattr(self.layer, 'activity_regularizer'):
      return self.layer.activity_regularizer
    else:
      return None

  def get_config(self):
    config = {
        'layer': {
            'class_name': self.layer.__class__.__name__,
            'config': self.layer.get_config()
        }
    }
    base_config = super(Wrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    # Avoid mutating the input dict
    config = config.copy()
    layer = deserialize_layer(
        config.pop('layer'), custom_objects=custom_objects)
    return cls(layer, **config)


class TimeDistributed(Wrapper):
  """This wrapper allows to apply a layer to every temporal slice of an input.

  The input should be at least 3D, and the dimension of index one
  will be considered to be the temporal dimension.

  Consider a batch of 32 video samples, where each sample is a 128x128 RGB image
  with `channels_last` data format, across 10 timesteps.
  The batch input shape is `(32, 10, 128, 128, 3)`.

  You can then use `TimeDistributed` to apply a `Conv2D` layer to each of the
  10 timesteps, independently:

  >>> inputs = tf.keras.Input(shape=(10, 128, 128, 3))
  >>> conv_2d_layer = tf.keras.layers.Conv2D(64, (3, 3))
  >>> outputs = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
  >>> outputs.shape
  TensorShape([None, 10, 126, 126, 64])

  Arguments:
    layer: a `tf.keras.layers.Layer` instance.

  Call arguments:
    inputs: Input tensor.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the
      wrapped layer (only if the layer supports this argument).
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked. This argument is passed to the
      wrapped layer (only if the layer supports this argument).

  Raises:
    ValueError: If not initialized with a `tf.keras.layers.Layer` instance.
  """

  def __init__(self, layer, **kwargs):
    if not isinstance(layer, Layer):
      raise ValueError(
          'Please initialize `TimeDistributed` layer with a '
          '`tf.keras.layers.Layer` instance. You passed: {input}'.format(
              input=layer))
    super(TimeDistributed, self).__init__(layer, **kwargs)
    self.supports_masking = True
    self._supports_ragged_inputs = True

    # It is safe to use the fast, reshape-based approach with all of our
    # built-in Layers.
    self._always_use_reshape = (
        layer_utils.is_builtin_layer(layer) and
        not getattr(layer, 'stateful', False))

  def _get_shape_tuple(self, init_tuple, tensor, start_idx, int_shape=None):
    """Finds non-specific dimensions in the static shapes.

    The static shapes are replaced with the corresponding dynamic shapes of the
    tensor.

    Arguments:
      init_tuple: a tuple, the first part of the output shape
      tensor: the tensor from which to get the (static and dynamic) shapes
        as the last part of the output shape
      start_idx: int, which indicate the first dimension to take from
        the static shape of the tensor
      int_shape: an alternative static shape to take as the last part
        of the output shape

    Returns:
      The new int_shape with the first part from init_tuple
      and the last part from either `int_shape` (if provided)
      or `tensor.shape`, where every `None` is replaced by
      the corresponding dimension from `tf.shape(tensor)`.
    """
    # replace all None in int_shape by K.shape
    if int_shape is None:
      int_shape = K.int_shape(tensor)[start_idx:]
    if not any(not s for s in int_shape):
      return init_tuple + tuple(int_shape)
    shape = K.shape(tensor)
    int_shape = list(int_shape)
    for i, s in enumerate(int_shape):
      if not s:
        int_shape[i] = shape[start_idx + i]
    return init_tuple + tuple(int_shape)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if len(input_shape) < 3:
      raise ValueError(
          '`TimeDistributed` Layer should be passed an `input_shape ` '
          'with at least 3 dimensions, received: ' + str(input_shape))
    # Don't enforce the batch or time dimension.
    self.input_spec = InputSpec(shape=[None, None] + input_shape[2:])
    child_input_shape = [input_shape[0]] + input_shape[2:]
    super(TimeDistributed, self).build(tuple(child_input_shape))
    self.built = True

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    child_input_shape = tensor_shape.TensorShape([input_shape[0]] +
                                                 input_shape[2:])
    child_output_shape = self.layer.compute_output_shape(child_input_shape)
    if not isinstance(child_output_shape, tensor_shape.TensorShape):
      child_output_shape = tensor_shape.TensorShape(child_output_shape)
    child_output_shape = child_output_shape.as_list()
    timesteps = input_shape[1]
    return tensor_shape.TensorShape([child_output_shape[0], timesteps] +
                                    child_output_shape[1:])

  def call(self, inputs, training=None, mask=None):
    kwargs = {}
    if generic_utils.has_arg(self.layer.call, 'training'):
      kwargs['training'] = training

    input_shape = K.int_shape(inputs)
    if input_shape[0] and not self._always_use_reshape:
      inputs, row_lengths = K.convert_inputs_if_ragged(inputs)
      is_ragged_input = row_lengths is not None

      # batch size matters, use rnn-based implementation
      def step(x, _):
        output = self.layer(x, **kwargs)
        return output, []

      _, outputs, _ = K.rnn(
          step,
          inputs,
          initial_states=[],
          input_length=row_lengths[0] if is_ragged_input else input_shape[1],
          mask=mask,
          unroll=False)
      y = K.maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths)
    else:
      # No batch size specified, therefore the layer will be able
      # to process batches of any size.
      # We can go with reshape-based implementation for performance.
      if isinstance(inputs, ragged_tensor.RaggedTensor):
        y = self.layer(inputs.values, **kwargs)
        y = ragged_tensor.RaggedTensor.from_row_lengths(
            y,
            inputs.nested_row_lengths()[0])
      else:
        input_length = input_shape[1]
        if not input_length:
          input_length = array_ops.shape(inputs)[1]
        inner_input_shape = self._get_shape_tuple((-1,), inputs, 2)
        # Shape: (num_samples * timesteps, ...). And track the
        # transformation in self._input_map.
        inputs = array_ops.reshape(inputs, inner_input_shape)
        # (num_samples * timesteps, ...)
        if generic_utils.has_arg(self.layer.call, 'mask') and mask is not None:
          inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
          kwargs['mask'] = K.reshape(mask, inner_mask_shape)

        y = self.layer(inputs, **kwargs)

        # Shape: (num_samples, timesteps, ...)
        output_shape = self.compute_output_shape(input_shape).as_list()
        output_shape = self._get_shape_tuple((-1, input_length), y, 1,
                                             output_shape[2:])
        y = array_ops.reshape(y, output_shape)

    return y

  def compute_mask(self, inputs, mask=None):
    """Computes an output mask tensor for Embedding layer.

    This is based on the inputs, mask, and the inner layer.
    If batch size is specified:
    Simply return the input `mask`. (An rnn-based implementation with
    more than one rnn inputs is required but not supported in tf.keras yet.)
    Otherwise we call `compute_mask` of the inner layer at each time step.
    If the output mask at each time step is not `None`:
    (E.g., inner layer is Masking or RNN)
    Concatenate all of them and return the concatenation.
    If the output mask at each time step is `None` and the input mask is not
    `None`:(E.g., inner layer is Dense)
    Reduce the input_mask to 2 dimensions and return it.
    Otherwise (both the output mask and the input mask are `None`):
    (E.g., `mask` is not used at all)
    Return `None`.

    Arguments:
      inputs: Tensor with shape [batch size, timesteps, ...] indicating the
        input to TimeDistributed. If static shape information is available for
        "batch size", `mask` is returned unmodified.
      mask: Either None (indicating no masking) or a Tensor indicating the
        input mask for TimeDistributed. The shape can be static or dynamic.

    Returns:
      Either None (no masking), or a [batch size, timesteps, ...] Tensor with
      an output mask for the TimeDistributed layer with the shape beyond the
      second dimension being the value of the input mask shape(if the computed
      output mask is none), an output mask with the shape beyond the first
      dimension being the value of the mask shape(if mask is not None) or
      output mask with the shape beyond the first dimension being the
      value of the computed output shape.

    """
    # cases need to call the layer.compute_mask when input_mask is None:
    # Masking layer and Embedding layer with mask_zero
    input_shape = K.int_shape(inputs)
    if input_shape[0] and not self._always_use_reshape or isinstance(
        inputs, ragged_tensor.RaggedTensor):
      # batch size matters, we currently do not handle mask explicitly, or if
      # the layer always uses reshape approach, or the input is a ragged tensor.
      return mask
    inner_mask = mask
    if inner_mask is not None:
      inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
      inner_mask = K.reshape(inner_mask, inner_mask_shape)
    inner_input_shape = self._get_shape_tuple((-1,), inputs, 2)
    inner_inputs = array_ops.reshape(inputs, inner_input_shape)
    output_mask = self.layer.compute_mask(inner_inputs, inner_mask)
    if output_mask is None:
      if mask is None:
        return None
      # input_mask is not None, and output_mask is None:
      # we should return a not-None mask
      output_mask = mask
      for _ in range(2, len(K.int_shape(mask))):
        output_mask = K.any(output_mask, axis=-1)
    else:
      # output_mask is not None. We need to reshape it
      input_length = input_shape[1]
      if not input_length:
        input_length = K.shape(inputs)[1]
      output_mask_int_shape = K.int_shape(output_mask)
      if output_mask_int_shape is None:
        # if the output_mask does not have a static shape,
        # its shape must be the same as mask's
        if mask is not None:
          output_mask_int_shape = K.int_shape(mask)
        else:
          output_mask_int_shape = K.compute_output_shape(input_shape)[:-1]
      output_mask_shape = self._get_shape_tuple(
          (-1, input_length), output_mask, 1, output_mask_int_shape[1:])
      output_mask = K.reshape(output_mask, output_mask_shape)
    return output_mask