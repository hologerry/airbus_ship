"""
Deep model
"""
import tensorflow as tf

class Model():
    def __init__(self, args):
        self.X = tf.placeholder(tf.float32, shape=[None, args.size, args.size, 3], name="X")
        