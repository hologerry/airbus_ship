"""Project configurations
"""

import argparse


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='parser of airbus ship compition project')

        subparser = self.parser.add_subparsers(title='subcommands', dest='subcommand')

        # Training args
        train_arg = subparser.add_parser("train", help='parser for training arguments')
        # train_arg.add_argument("--dataset_dir", type=str, default="/media/gerry/Data_2/kaggle_airbus_data",
        #                        help="directory of dataset")
        train_arg.add_argument("--dataset_dir", type=str, default="/S1/CSCL/gaoy/kaggle_airbus_data",
                               help="directory of dataset gpu cluster")
        train_arg.add_argument("--train_masks", type=str, default='train_ship_segmentations.csv',
                               help="train masks csv name")
        train_arg.add_argument("--gpu", type=int, default=1, help="whether to use gpu")
        train_arg.add_argument("--train_img_dir", type=str, default='train', help="train image dir")
        train_arg.add_argument("--shuffle", type=bool, default=True, help="dataloader shuffle")
        train_arg.add_argument("--batch_size", type=int, default=16, help="train batch size")
        train_arg.add_argument("--val_batch_size", type=int, default=1, help="validate batch size")
        train_arg.add_argument("--samples_per_ship_group", type=int, default=2000, help="")
        train_arg.add_argument("--epochs", type=int, default=32, help="number of initial taining epochs")
        train_arg.add_argument("--lr", type=float, default=0.00001, help="learning rate")
        train_arg.add_argument("--decay_fr", type=int, default=8, help="frequency of learning rate decay")
        train_arg.add_argument("--log_fr", type=int, default=50, help="frequency of saving log")
        train_arg.add_argument("--debug", type=bool, default=False, help="debug?")
        train_arg.add_argument("--train_valid_ratio", type=float, default=0.05, help="train valid tratio")
        train_arg.add_argument("--optim", type=int, default=0, help="0 for 256 res, 1 for 384 res, 2 for 768 res")
        # Testing args
        test_arg = subparser.add_parser("test", help='parser for testing arguments')
        test_arg.add_argument("--batch_size", type=int, default=1, help="batch size")
        test_arg.add_argument("--dataset_dir", type=str, default="/S1/CSCL/gaoy/kaggle_airbus_data",
                              help="directory of dataset gpu cluster")
        test_arg.add_argument("--test_img_dir", type=str, default='test', help="train image dir")
        test_arg.add_argument("--gpu", type=int, default=1, help="gpu?")

    def parse(self):
        return self.parser.parse_args()
