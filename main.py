from option import Options

import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn


def train(args):
    pass


def test(args):
    pass


def main():
    args = Options().parse()
    if args.subcommand is None:
        raise ValueError('ERROR: specify the experiment type')
    if args.cuda and not tf.test.is_gpu_available():
        raise ValueError('ERROR: gpu is not available, try running on cpu')
    
    if args.subcomand == 'train':
        train(args)
    elif args.subcommand == 'test':
        test(args)
    else:
        raise ValueError('ERROR: Unknow experiment type')
