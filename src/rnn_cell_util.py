from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from tensorflow.python.framework import ops
from tensorflow.python.ops.rnn_cell_impl import RNNCell


def mean_field_cell(logits, state):
    return logits, state


class GeneratorCellBuilder(object):
    def __init__(self, cell_type, **kwargs):
        self.c  = cell_type
        self.kw = kwargs

    def _check_feed_actions(self, feed_actions):
        if feed_actions:
            raise Exception("Cannot feed actions, only logits!")

    def _build_cell(self, **kwargs):
        return self.c

    def _init_state(self, cell, batch_size):
        return None

    def build_cell_and_init_state(self, batch_size, feed_actions):
        self._check_feed_actions(feed_actions)
        cell = self._build_cell(**self.kw)
        return cell, self._init_state(cell, batch_size)


class GeneratorRNNCellBuilder(GeneratorCellBuilder):
    def _check_feed_actions(self, feed_actions):
        pass

    def _build_cell(self, m, n_stack=1, wrappers=[]):
        if n_stack == 1:
            cell = self.c(m)
        cell = rnn.MultiRNNCell([self.c(m) for _ in range(n_stack)])
        # Apply wrappers; use functools.partial to bind other arguments
        for wrapper in wrappers:
            cell = wrapper(cell)
        return cell

    def _init_state(self, cell, batch_size):
        return cell.zero_state(batch_size, tf.float32)


class OutputRangeWrapper(RNNCell):
    def __init__(self, cell, output_range, norm_op=None):
        """Rescales output range of @cell
            @cell: an RNN cell
            @output_range: range of outputs, e.g. 4 produces outputs in [-2, 2]
            @norm_op: function to map @cell outputs to range [0, 1]
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        if output_range < 0:
            raise ValueError("Logit range must be > 0: %d." % output_range)
        self._cell  = cell
        self._range = output_range
        self._norm  = norm_op

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell._output_size

    def zero_state(self, n, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[n]):
            return self._cell.zero_state(n, dtype)

    def __call__(self, inputs, state, scope=None):
        output, res_state = self._cell(inputs, state)
        if self._norm:
            output = self._norm(output)
        return self._range * (output - 0.5), res_state
