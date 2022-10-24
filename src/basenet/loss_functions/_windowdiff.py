# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf


# -----------------------------------------------------------
def window_diff(a: tf.Tensor, b: tf.Tensor, th=0.5):
    # Data fromatting:
    _b = tf.cast(tf.convert_to_tensor(b), dtype='float32')
    _a = tf.cast(tf.convert_to_tensor(a), dtype='float32')
    _b = tf.where(_b > th, 1, 0)
    _a = tf.where(_a > th, 1, 0)
    _b = tf.cast(_b, dtype='float32')
    _a = tf.cast(_a, dtype='float32')
    access__ = 1
    # Window size:
    _aux = tf.math.multiply(2.0, (tf.math.add(1.0, tf.math.reduce_sum(_b))))
    _aux = tf.math.divide(_b.shape[1] * _b.shape[0], _aux)
    w_size = tf.round(_aux)
    w_size = tf.cast(w_size, dtype='int32')
    # Divider is the number of convolutions.
    _aux = tf.math.subtract(_b.shape[access__], w_size)
    _divider = tf.math.add(_aux, 1)
    # We create the masks of the window.
    _n_shifts = tf.math.subtract(w_size, 1)
    _mask = tf.eye(_divider, num_columns=_b.shape[access__], dtype='float32')
    _masks = tf.eye(_divider, num_columns=_b.shape[access__], dtype='float32')
    while tf.greater(_n_shifts, 0):
        _this_roll = tf.roll(_mask, shift=_n_shifts, axis=1)
        _masks = tf.math.add(_masks, _this_roll)
        _n_shifts = tf.math.subtract(_n_shifts, 1)
    # Addup is the sum of the total error, masked.
    _a_masked = tf.linalg.matvec(_masks, _a)
    _b_masked = tf.linalg.matvec(_masks, _b)
    _aux = tf.subtract(_a_masked, _b_masked)
    _diff_unnorm = tf.math.abs(_aux)
    _aux = tf.where(_diff_unnorm > 0, 1, 0)
    _addup = tf.reduce_sum(_aux, axis=access__)
    _addup = tf.cast(_addup, dtype='int32')
    # Return the windowdiff.
    _result = tf.math.divide(_addup, _divider)
    return tf.reduce_mean(_result)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
