from option import Options
from data.dataset import Dataset
from model.model_tf import UnetModel

import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def train(args):
    print("Training the network ...")
    dataset = Dataset(args)
    unet = UnetModel(args)

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        X, y = dataset.iter.get_next()
        y_pred = unet.net(X)
        variables = tf.trainable_variables()
        saver = tf.train.Saver()
        loss = unet.cal_loss(y, y_pred)

        optim = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss, var_list=variables)

        sess.run(tf.global_variables_initializer())

        start_time = datetime.now()
        for epoch in range(args.epochs):

            epoch_s_time = datetime.now()
            sess.run(dataset.train_init_op)
            for batch in range(args.batches_per_epoch):
                _, batch_loss = sess.run([optim, loss])
                curtime = datetime.now()
                print("epoch:", epoch, "of", args.epochs, " batch:", batch, "of", args.batches_per_epoch,
                      "batch loss:", batch_loss, "  elapsed time:", curtime-epoch_s_time)
            epoch_e_time = datetime.now()

            print("Validating for current epoch...")
            sess.run(dataset.valid_init_op)
            tot_valid_loss = 0.0
            for _ in range(args.valid_batches):
                valid_loss = sess.run(loss)
                tot_valid_loss += valid_loss
            mean_v_loss = tot_valid_loss / args.valid_batches
            valid_time = datetime.now()
            print("epoch:", epoch, "mean valid loss:", mean_v_loss, "validate elapsed time: ", valid_time-epoch_e_time)

            if (epoch+1) % args.ckpt_fr == 0:
                print("saving model for epoch:", epoch)
                unet.save_checkpoint(saver, sess, epoch)

        end_time = datetime.now()
        print("train finished..", "elpased time:", end_time-start_time)


def test(args):
    # TODO: test and generate csv
    unet = UnetModel(args)
    latest_model_epoch = 95

    with tf.Session() as sess:
        saver = tf.train.Saver()
        unet.restore_checkpoint(sess, saver, latest_model_epoch)




def main():
    args = Options().parse()
    if args.subcommand is None:
        raise ValueError('ERROR: specify the experiment type')
    if args.gpu and not tf.test.is_gpu_available():
        raise ValueError('ERROR: gpu is not available, try running on cpu')

    if args.subcommand == 'train':
        train(args)
    elif args.subcommand == 'test':
        test(args)
    else:
        raise ValueError('ERROR: Unknow experiment type')


if __name__ == "__main__":
    main()