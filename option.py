import argparse
import os


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="parser of airbus ship competition project")

        subparsers = self.parser.add_subparsers(title="subcommands", dest="subcommand")

        # training args
        train_arg = subparsers.add_parser("train", help="parser for training arguments")
        train_arg.add_argument("--batch_size", type=int, default=64, help="batch size")
        train_arg.add_argument("--epochs", type=int, default=30, help="number of epoch")
        train_arg.add_argument("--max_train_steps", type=int, default=300, help="maxium number of steps_per_epoch in training")
        train_arg.add_argument("--img_size", type=tuple, default=(768,768), help="image shape/size")
        train_arg.add_argument("--img_scaling", type=tuple, default=(4,4), help="downsampling during preprocessing")
        train_arg.add_argument("--size", type=int, default=768/4, help="image size down sampled")
        train_arg.add_argument("--dataset_dir", type=str, default="/media/gerry/Data_2/kaggle_airbus_data", help="directory of dataset")
        train_arg.add_argument("--train_masks_csv", type=str, default='train_ship_segmentations.csv', help="train masks csv name")
        train_arg.add_argument("--train_img_dir", type=str, default="train", help="path to train set")
        train_arg.add_argument("--edge_crop", type=int, default=16, help="not understood")
        train_arg.add_argument("--guassian_noise", type=float, default=0.1, help="guassian noise")
        train_arg.add_argument("--samples_per_ship_group", type=int, default=2000, help="group by ships(ship number), upper bound")
        train_arg.add_argument("--train_valid_ratio", type=float, default=0.3, help="ration when split train and test set")

        train_arg.add_argument("--valid_img_count", type=int, default=600, help="number of validation images to use")
        train_arg.add_argument("--augment_brightness", type=bool, default=False, help="whether augment the img brightness")
        train_arg.add_argument("--gpu", type=int, default=1, help="whether to use gpu")
        train_arg.add_argument("--debug", type=bool, default=True, help="debug?")
        
        # testing args
        test_arg = subparsers.add_parser("test", help="parser for testing arguments")
        test_arg.add_argument("--test_img_dir", type=str, default="test", help="path to test set")
        test_arg.add_argument("--csv_dir", type=str, default="results")
        test_arg.add_argument("--debug", type=bool, default=True, help="debug?")
        test_arg.add_argument("--samples_per_ship_group", type=int, default=2000, help="group by ships(ship number), upper bound")
        test_arg.add_argument("--img_scaling", type=tuple, default=(4,4), help="downsampling during preprocessing")


    def parse(self):
        return self.parser.parse_args()
