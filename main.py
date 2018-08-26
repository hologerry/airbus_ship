from option import Options
from data.dataset import Dataset
from data.rle import multi_rle_encode
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
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        file_writer = tf.summary.FileWriter(args.summary_dir)
        X, y = dataset.iter.get_next()
        y_pred = unet.full_res_net(X)
        variables = tf.trainable_variables()
        saver = tf.train.Saver()
        loss = unet.cal_loss(y, y_pred)

        optim = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss, var_list=variables)

        sess.run(tf.global_variables_initializer())

        start_time = datetime.now()
        for epoch in range(args.epochs):
            print("Start training the network ...")
            epoch_s_time = datetime.now()
            sess.run(dataset.train_init_op)
            for batch in range(args.batches_per_epoch):
                _, batch_loss = sess.run([optim, loss])
                curtime = datetime.now()
                print("epoch:", epoch, "of", args.epochs, " batch:", batch, "of", args.batches_per_epoch,
                      "batch loss:", batch_loss, "  elapsed time:", curtime-epoch_s_time)
                if batch % 10 == 0:
                    loss_sum = tf.summary.Summary()
                    loss_sum.value.add(tag='train_loss', simple_value=float(batch_loss))  # pylint: disable=E1101
                    file_writer.add_summary(loss_sum)
                    file_writer.flush()
            epoch_e_time = datetime.now()

            print("Validating for current epoch...")
            sess.run(dataset.valid_init_op)
            tot_valid_loss = 0.0
            for valid_b in range(args.valid_batches):
                valid_loss = sess.run(loss)
                tot_valid_loss += valid_loss
                if valid_b % 10 == 0:
                    vloss_sum = tf.summary.Summary()
                    vloss_sum.value.add(tag='valid_loss', simple_value=float(valid_loss))  # pylint: disable=E1101
                    file_writer.add_summary(vloss_sum)
            mean_v_loss = tot_valid_loss / args.valid_batches
            valid_time = datetime.now()
            print("epoch:", epoch, "mean valid loss:", mean_v_loss, "validate elapsed time: ", valid_time-epoch_e_time)

            if (epoch+1) % args.ckpt_fr == 0:
                print("saving model for epoch:", epoch)
                unet.save_checkpoint(saver, sess, epoch)

        end_time = datetime.now()
        print("train finished..", "elpased time:", end_time-start_time)


def test(args):
    print("Testing the network")
    dataset = Dataset(args)
    unet = UnetModel(args)

    latest_model_epoch = 30

    with tf.Session() as sess:
        test_X, test_img_id = dataset.test_iter.get_next()
        y_pred = unet.full_res_net(test_X)
        saver = tf.train.Saver()
        unet.restore_checkpoint(saver, sess, latest_model_epoch)
        sess.run(dataset.test_init_op)

        out_pred_rows = []

        while True:
            try:
                curbatch_seg = sess.run(y_pred)
                for idx, one_seg in enumerate(curbatch_seg):
                    cur_rles = multi_rle_encode(one_seg)
                    if cur_rles is not None:
                        for one_rle in cur_rles:
                            out_pred_rows += [[test_img_id[idx], one_rle]]

            except:
                print("Finished test all images")
                test_df = pd.DataFrame(out_pred_rows)
                test_df.columns = ['ImageId', 'EncodedPixels']
                test_df.to_csv(os.path.join(args.result_dir, "submission.csv"))
                print(test_df.head())


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