# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf
from tensorflow import keras
# -----------------------------------------------------------


class Pbmm(keras.layers.Layer):
    def __init__(self, shape, **kwargs):
        super(Pbmm, self).__init__()
        th_init = tf.random_normal_initializer()
        cbt_init = tf.random_normal_initializer()
        oim_init = tf.zeros_initializer()
        self.th = tf.Variable(initial_value=th_init(shape=(), dtype="float32"), trainable=True)
        self.cbt = tf.Variable(initial_value=cbt_init(shape=(), dtype="float32"), trainable=True)
        self.oim = tf.Variable(initial_value=oim_init(shape=(), dtype="uint8"), trainable=True)
        self.shape = shape
        self.kwargs = kwargs

    def call(self, inputs: tf.Tensor, *args, **kwargs):
        # Init:
        failure_counter = 0
        last_index = -1
        current_index = 0
        appindex = 0
        d = tf.Variable(initial_value=(), trainable=False)

        # Algorithm loop:
        while tf.math.less((current_index := current_index + 1), inputs.shape[0]):
            # Compute mean
            elements = inputs[current_index, last_index + 1:current_index]
            if current_index - last_index - 1 <= 0:
                mean = 1
            else:
                mean = sum(elements) / (current_index - last_index - 1)
            # Algorithm control:
            if tf.math.less(mean, self.th):
                failure_counter += 1
            else:
                appindex = current_index
                failure_counter = 0

            if tf.math.greater(failure_counter, self.oim):
                d = tf.concat([d, appindex + 1], 0)
                len_cb = int(d[-1]) - last_index - 1  # Checkback init.
                init_cb = last_index + 1  # Checkback init.
                last_index = appindex
                current_index = appindex

                # Checkback...
                if len_cb > 1:
                    cb_mean = tf.Variable(initial_value=0.0, trainable=False)
                    for i in range(len_cb - 1):
                        cb_mean = cb_mean.asign_add(inputs[init_cb, init_cb + i + 1])
                    cb_mean = tf.math.divide(cb_mean, len_cb - 1)
                    if tf.math.less(cb_mean, self.cbt):
                        # Check back integrity...
                        cb_mean_back = inputs[init_cb, init_cb - 1] if tf.math.greater(init_cb, 0) else -1
                        if tf.math.less(cb_mean_back, self.cbt):
                            aux = d.pop(-1)
                            if d:
                                d = tf.concat([d, [d[-1] + 1]], 0)
                            else:
                                d = tf.concat([d, [1]], 0)
                            d = tf.concat([d, [aux]])
                        else:
                            if d:
                                d[-2] = tf.add(d[-2], 1)
                            else:
                                d = tf.concat([d, [1]], 0)
        # Last element:
        d = tf.concat([d, [current_index]])
        return d
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
