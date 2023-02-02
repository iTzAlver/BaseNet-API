# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import tensorflow as tf
import numpy as np
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #


def confusion_matrix(a: tf.Tensor, b: tf.Tensor, th=None) -> None:
    _b = tf.cast(tf.convert_to_tensor(b), dtype='float32')
    _a = tf.cast(tf.convert_to_tensor(a), dtype='float32')
    categorical_a = tf.argmax(a, axis=-1)
    categorical_b = tf.argmax(b, axis=-1)
    # Numpy zone...
    confusion = np.zeros((a.shape[-1], b.shape[-1]))
    for aa, ab in zip(categorical_a, categorical_b):
        confusion[aa, ab] += 1
    # Tf zone...
    return tf.convert_to_tensor(confusion)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
