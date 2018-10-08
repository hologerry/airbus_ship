import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from data.data_aug import (DualCompose, HorizontalFlip, RandomRotate90, Resize,
                           Shift, Transpose, VerticalFlip)
from data.dataset import make_dataloader
from data.load_data import get_balanced_train_valid, get_unique_img_ids
from data.rle import multi_rle_encode
from model.loss import LossBinary, get_jaccard
from model.unet import UNet
from option import Options
from utils import save_model, write_event

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def train(args):
    print("Traning")

    print("Prepaing data")
    masks = pd.read_csv(os.path.join(args.dataset_dir, args.train_masks))
    unique_img_ids = get_unique_img_ids(masks, args)
    train_df, valid_df = get_balanced_train_valid(masks, unique_img_ids, args)

    if args.stage == 0:
        train_shape = (256, 256)
        batch_size = args.stage0_batch_size
        extra_epoch = args.stage0_epochs
    elif args.stage == 1:
        train_shape = (384, 384)
        batch_size = args.stage1_batch_size
        extra_epoch = args.stage1_epochs
    elif args.stage == 2:
        train_shape = (512, 512)
        batch_size = args.stage2_batch_size
        extra_epoch = args.stage2_epochs
    elif args.stage == 3:
        train_shape = (768, 768)
        batch_size = args.stage3_batch_size
        extra_epoch = args.stage3_epochs

    print("Stage {}".format(args.stage))

    train_transform = DualCompose([
        Resize(train_shape),
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        Shift(),
        Transpose(),
        # ImageOnly(RandomBrightness()),
        # ImageOnly(RandomContrast()),
    ])
    val_transform = DualCompose([
        Resize(train_shape),
    ])

    train_dataloader = make_dataloader(
        train_df, args, batch_size, args.shuffle, transform=train_transform)
    val_dataloader = make_dataloader(
        valid_df, args, batch_size//2, args.shuffle, transform=val_transform)

    # Build model
    model = UNet()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.decay_fr, gamma=0.1)
    if args.gpu and torch.cuda.is_available():
        model = model.cuda()

    # Restore model ...
    run_id = 4

    model_path = Path('model_{run_id}.pt'.format(run_id=run_id))
    if not model_path.exists() and args.stage > 0:
        raise ValueError(
            'model_{run_id}.pt does not exist, initial train first.'.format(run_id=run_id))
    if model_path.exists():
        state = torch.load(str(model_path))
        last_epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restore model, epoch {}, step {:,}'.format(last_epoch, step))
    else:
        last_epoch = 1
        step = 0

    log_file = open('train_{run_id}.log'.format(
        run_id=run_id), 'at', encoding='utf8')

    loss_fn = LossBinary(jaccard_weight=args.iou_weight)

    valid_losses = []

    print("Start training ...")
    for _ in range(last_epoch):
        scheduler.step()

    for epoch in range(last_epoch, last_epoch+extra_epoch):
        scheduler.step()
        model.train()
        random.seed()
        tq = tqdm(total=len(train_dataloader)*batch_size)
        tq.set_description('Run Id {}, Epoch {} of {}, lr {}'.format(run_id, epoch, last_epoch+extra_epoch,
                                                                     args.lr * (0.1 ** (epoch // args.decay_fr))))
        losses = []
        try:
            mean_loss = 0.
            for i, (inputs, targets) in enumerate(train_dataloader):
                inputs, targets = torch.tensor(inputs), torch.tensor(targets)
                if args.gpu and torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-args.log_fr:])
                tq.set_postfix(loss="{:.5f}".format(mean_loss))

                if i and (i % args.log_fr) == 0:
                    write_event(log_file, step, loss=mean_loss)
            write_event(log_file, step, loss=mean_loss)
            tq.close()
            save_model(model, epoch, step, model_path)

            valid_metrics = validation(args, model, loss_fn, val_dataloader)
            write_event(log_file, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)

        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save_model(model, epoch, step, model_path)
            print('Terminated.')
    print('Done.')


def validation(args, model: torch.nn.Module, criterion, valid_loader):
    print("Validating Network...")
    if args.gpu and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    losses = []
    jaccard = []
    for inputs, targets in valid_loader:
        inputs, targets = torch.tensor(inputs), torch.tensor(targets)
        if args.gpu and torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        jaccard += [get_jaccard(targets, (outputs > 0).float()).item()]

    valid_loss = np.mean(losses)  # type: float

    valid_jaccard = np.mean(jaccard)

    print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(
        valid_loss, valid_jaccard))
    metrics = {'valid_loss': valid_loss, 'jaccard': valid_jaccard}
    return metrics


def test(args):
    print("Predicting ...")
    test_paths = os.listdir(os.path.join(args.dataset_dir, args.test_img_dir))
    print(len(test_paths), 'test images found')
    test_df = pd.DataFrame({'ImageId': test_paths, 'EncodedPixels': None})

    from skimage.morphology import binary_opening, disk

    test_df = test_df[:5000]
    test_loader = make_dataloader(test_df, args, batch_size=args.batch_size,
                                  shuffle=False, transform=None, mode='predict')

    model = UNet()
    if args.gpu and torch.cuda.is_available():
        model = model.cuda()
    run_id = 1
    print("Resuming run #{}...".format(run_id))
    model_path = Path('model_{run_id}.pt'.format(run_id=run_id))
    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key,
             value in state['model'].items()}
    model.load_state_dict(state)

    out_pred_rows = []

    for batch_id, (inputs, image_paths) in enumerate(tqdm(test_loader, desc='Predict')):
        if args.gpu and torch.cuda.is_available():
            inputs = inputs.cuda()
        inputs = torch.tensor(inputs)
        outputs = model(inputs)
        for i, image_name in enumerate(image_paths):
            mask = torch.sigmoid(outputs[i, 0]).data.cpu().numpy()
            cur_seg = binary_opening(mask > 0.5, disk(2))
            cur_rles = multi_rle_encode(cur_seg)
            if len(cur_rles) > 0:
                for c_rle in cur_rles:
                    out_pred_rows += [{'ImageId': image_name,
                                       'EncodedPixels': c_rle}]
            else:
                out_pred_rows += [{'ImageId': image_name,
                                   'EncodedPixels': None}]

    submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
    submission_df.to_csv('submission.csv', index=False)
    print("done.")


def main():
    args = Options().parse()
    if args.subcommand is None:
        raise ValueError('ERROR: specify the experiment type')
    if args.gpu and not torch.cuda.is_available():
        raise ValueError('ERROR: gpu is not available, try running on cpu')

    if args.subcommand == 'train':
        train(args)
    elif args.subcommand == 'test':
        test(args)
    else:
        raise ValueError('ERROR: unknown experiment type')


if __name__ == '__main__':
    main()
