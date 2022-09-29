"""
A novel RNN that calculates a running average, sum, or max of inputs.
This RNN is an integral part of the architecture proposed in the AMRL paper (https://iclr.cc/virtual_2020/poster_Bkl7bREtDr.html).
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

@tf.custom_gradient
def divide_grad_straight_through(x, y):
    def grad(dy):
        return (dy, tf.zeros_like(y))
    return x/y, grad

@tf.custom_gradient
def maximum_grad_straight_through(x, y):
    def grad(dy):
        return (dy, dy)
    return tf.maximum(x,y), grad

class AverageRNNCell(rnn.LayerRNNCell):
    def __init__(self, input_size, eps=None, count=True, straight_through=False, sum_instead=False, max_instead=False,
                 reuse=None, name=None, dtype=None, **kwargs):
        super().__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        self.input_size = input_size
        self.sum_instead = sum_instead
        self.max_instead = max_instead
        self.straight_through = straight_through
        self.count = count
        # When copmuting an avergage, if not None, the sum will be divided by t^(1+eps) to prevent the logarithmic accumulation of gradient on the first input
        # None or 0.001 Is probably a decent value. For example, sum(1/t^(1+0.001)) converges quickly but t^(1+0.001) is approximately t for well into trillion of steps
        self.eps = eps
        assert not (sum_instead and max_instead), "Can only do sum or max, not both"
        assert not (straight_through and sum_instead), "straight_through is only for average or max"

    @property
    def state_size(self):
        return LSTMStateTuple(self.input_size+1, self.input_size+1)

    @property
    def output_size(self):
        return self.input_size+1 if self.count else self.input_size

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % str(inputs_shape))

        input_depth = inputs_shape[-1]
        assert inputs_shape[-1] == self.input_size, "The input size {} must be equal to self.input_size {}".format(inputs_shape[-1], self.input_size)
        self.built = True

    def call(self, inputs, state):

        c = state.c # previous h (output) not used
        prev_num_steps = c[:,self.input_size:self.input_size+1]  # [batch_sz, 1]
        new_num_steps = prev_num_steps+1.0 # [batch_sz, 1]
        prev_running_sums = c[:,0:self.input_size] # [batch_sz, ecoding_sz]
        

        if self.max_instead:
            if self.straight_through:
                new_running_sums = avg = maximum_grad_straight_through(prev_running_sums, inputs)
            else:
                new_running_sums = avg = tf.maximum(prev_running_sums, inputs)
        else:
            new_running_sums = prev_running_sums + inputs
            if self.sum_instead:
                avg = new_running_sums
            elif self.straight_through:
                avg = divide_grad_straight_through(new_running_sums, new_num_steps)
            else:
                if self.eps is None:
                    avg = new_running_sums/new_num_steps
                else:
                    avg = new_running_sums/tf.pow(new_num_steps, (1.+self.eps) )

        new_state = tf.concat([new_running_sums, new_num_steps], axis=-1)
        h = tf.concat([avg, new_num_steps], axis=-1)
        if self.count:
            output = h
        else:
            output = avg # Cut off count
        return output, LSTMStateTuple(new_state, h)

    def get_config(self):
        config = {
            "input_size": self.input_size,
            "reuse": self._reuse,
            "straight_through": self.straight_through,
            "sum_instead": self.sum_instead,
            "max_instead": self.max_instead,
            "eps": self.eps,
            "count": self.count,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))