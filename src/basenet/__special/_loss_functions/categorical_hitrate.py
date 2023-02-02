# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def categorical_hitrate(a: tf.Tensor, b: tf.Tensor, th=None):
    """
    This function computes the categorical hit rate (1 - error_rate) with Tensorflow.
    The function goes from 0 to 1, being 0 the same performance as a random predictor.
    :param a: Hypothesis classification.
    :param b: Reference classification.
    :param th: Not used.
    :return: 1 - error_rate from (1/nclases mapped to 0), to 1.
    """
    _b = tf.cast(tf.convert_to_tensor(b), dtype='float32')
    _a = tf.cast(tf.convert_to_tensor(a), dtype='float32')
    categorical_a = tf.argmax(a, axis=-1)
    categorical_b = tf.argmax(b, axis=-1)
    errors = tf.reduce_sum(tf.abs(categorical_a - categorical_b)) / 2
    error_rate = errors / len(a)
    hit_rate = 1 - error_rate
    categorical_hit_rate = ((a.shape[-1] * hit_rate) - 1) / (a.shape[-1] - 1)
    return categorical_hit_rate
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
