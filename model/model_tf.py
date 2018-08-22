"""
TensorFlow U-net baseline model
"""
import tensorflow as tf
import os

class UnetModel():
    def __init__(self, args):
        self.args = args

    def net(self, X):
        with tf.variable_scope("encoder"):
            conv1_1 = tf.layers.conv2d(X, filters=16, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv1_1")
            conv1_2 = tf.layers.conv2d(conv1_1, filters=16, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv1_2")
            pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=(2,2), strides=2, name="pool1")

            conv2_1 = tf.layers.conv2d(pool1, filters=32, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv2_1")
            conv2_2 = tf.layers.conv2d(conv2_1, filters=32, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv2_2")
            pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=(2,2), strides=2, name="pool2")

            conv3_1 = tf.layers.conv2d(pool2, filters=64, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv3_1")
            conv3_2 = tf.layers.conv2d(conv3_1, filters=64, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv3_2")
            pool3 = tf.layers.max_pooling2d(conv3_2, pool_size=(2,2), strides=2, name="pool3")

            conv4_1 = tf.layers.conv2d(pool3, filters=128, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv4_1")
            conv4_2 = tf.layers.conv2d(conv4_1, filters=128, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv4_2")
            pool4 = tf.layers.max_pooling2d(conv4_2, pool_size=(2,2), strides=2, name="pool4")

            conv5_1 = tf.layers.conv2d(pool4, filters=256, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv5_1")
            conv5_2 = tf.layers.conv2d(conv5_1, filters=256, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv5_2")
            pool5 = tf.layers.max_pooling2d(conv5_2, pool_size=(2,2), strides=2, name="pool5")

            conv6_1 = tf.layers.conv2d(pool5, filters=512, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv6_1")
            conv6_2 = tf.layers.conv2d(conv6_1, filters=512, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv6_2")

        with tf.variable_scope("decoder"):
            dconv7 = tf.layers.conv2d_transpose(conv6_2, filters=256, kernel_size=(2,2), strides=(2,2), padding="same", name="dconv7")
            concat7 = tf.concat(axis=-1, values=[dconv7, conv5_2])
            conv7_1 = tf.layers.conv2d(concat7, filters=256, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv7_1")
            conv7_2 = tf.layers.conv2d(conv7_1, filters=256, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv7_2")

            dconv8 = tf.layers.conv2d_transpose(conv7_2, filters=128, kernel_size=(2,2), strides=(2,2), padding="same", name="dconv8")
            concat8 = tf.concat(axis=-1, values=[dconv8, conv4_2])
            conv8_1 = tf.layers.conv2d(concat8, filters=128, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv8_1")
            conv8_2 = tf.layers.conv2d(conv8_1, filters=128, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv8_2")

            dconv9 = tf.layers.conv2d_transpose(conv8_2, filters=128, kernel_size=(2,2), strides=(2,2), padding="same", name="dconv9")
            concat9 = tf.concat(axis=-1, values=[dconv9, conv3_2])
            conv9_1 = tf.layers.conv2d(concat9, filters=64, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv9_1")
            conv9_2 = tf.layers.conv2d(conv9_1, filters=64, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv9_2")

            dconv10 = tf.layers.conv2d_transpose(conv9_2, filters=32, kernel_size=(2,2), strides=(2,2), padding="same", name="dconv10")
            concat10 = tf.concat(axis=-1, values=[dconv10, conv2_2])
            conv10_1 = tf.layers.conv2d(concat10, filters=32, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv10_1")
            conv10_2 = tf.layers.conv2d(conv10_1, filters=32, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv10_2")

            dconv11 = tf.layers.conv2d_transpose(conv10_2, filters=16, kernel_size=(2,2), strides=(2,2), padding="same", name="dconv11")
            concat11 = tf.concat(axis=-1, values=[dconv11, conv1_2])
            conv11_1 = tf.layers.conv2d(concat11, filters=16, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv11_1")
            conv11_2 = tf.layers.conv2d(conv11_1, filters=16, kernel_size=(3,3), padding="same", activation=tf.nn.relu, name="conv11_2")

            out = tf.layers.conv2d(conv11_2, filters=1, kernel_size=(1,1), padding="same", activation=tf.nn.sigmoid, name="out_conv")
        
        return out


    def cal_loss(self, y_true, y_pred):
        return self.IoU(y_true, y_pred) + self.args.lambda_bce * self.binary_crossentropy(y_true, y_pred)

    
    def save_checkpoint(self, saver, sess, model_epoch):
        saver.save(sess, os.path.join(self.args.ckpt_dir, "model_"+str(model_epoch)+".ckpt"))

    
    def save_summary(self):
        pass


    def IoU(self, y_true, y_pred, eps=1e-8):
        if tf.reduce_max(y_true) == 0:
            return self.IoU(1-y_true, 1-y_pred)
        inter = tf.reduce_sum(y_true*y_pred, axis=[1,2,3])
        union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3]) - inter
        return -tf.reduce_mean((inter+eps)/(union+eps), axis=0)


    def binary_crossentropy(self, y_true, y_pred, eps=1e-8):
        y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
        y_pred = tf.log(y_pred/(1-y_pred))
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))