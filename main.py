import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.functional as F
from torch.optim import Adam
from tqdm import tqdm

from data.data_aug import (CenterCrop, DualCompose, HorizontalFlip, RandomCrop,
                           VerticalFlip)
from data.dataset import make_dataloader
from data.load_data import get_balanced_train_valid, get_unique_img_ids
from data.rle import multi_rle_encode
from model.unet import UNet
from model.loss import LossBinary
from model.loss import get_jaccard
from option import Options
from utils import write_event
from utils import save_model


def train(args):
    print("Traning Network...")
    masks = pd.read_csv(os.path.join(args.dataset_dir, args.train_masks))
    unique_img_ids = get_unique_img_ids(masks, args)
    train_df, valid_df = get_balanced_train_valid(masks, unique_img_ids, args)
    train_transform = DualCompose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomCrop((256, 256, 3)),
        # ImageOnly(RandomBrightness()),
        # ImageOnly(RandomContrast()),
    ])

    val_transform = DualCompose([
        CenterCrop((512, 512, 3)),
    ])

    train_dataloader = make_dataloader(train_df, args, args.batch_size, args.shuffle, transform=train_transform)
    val_dataloader = make_dataloader(valid_df, args, args.batch_size//2, args.shuffle, transform=val_transform)

    model = UNet()
    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.gpu:
        model.cuda()
    step = 0
    fold = 1
    model_path = Path('model_{fold}.pt'.format(fold=fold))
    log_file = open('train_{fold}.log'.format(fold=fold), 'at', encoding='utf8')
    loss_fn = LossBinary(jaccard_weight=5)

    valid_losses = []

    for epoch in range(1, args.epochs+1):
        model.train()
        random.seed()
        tq = tqdm(total=len(train_dataloader)*args.batch_size)
        tq.set_description('Epoch {} of {}, lr {}'.format(epoch, args.epochs, args.lr))
        losses = []
        try:
            mean_loss = 0.
            for i, (inputs, targets) in enumerate(train_dataloader):
                inputs, targets = torch.tensor(inputs), torch.tensor(targets)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(args.batch_size)
                losses.append(loss.data[0])
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
            save_model(log_file, epoch, step, model_path)
            print('done.')
            return


def validation(args, model: torch.nn.Module, criterion, valid_loader):
    model.eval()
    losses = []
    jaccard = []
    for inputs, targets in valid_loader:
        inputs, targets = torch.tensor(inputs), torch.tensor(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        jaccard += [get_jaccard(targets, (outputs > 0).float()).item()]

    valid_loss = np.mean(losses)  # type: float

    valid_jaccard = np.mean(jaccard)

    print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
    metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard}
    return metrics


def test(args):
    test_paths = os.listdir(os.path.join(args.dataset_dir, args.test_img_dir))
    print(len(test_paths), 'test images found')
    test_df = pd.DataFrame({'ImageId': test_paths, 'EncodedPixels': None})
    from skimage.morphology import binary_opening, disk
    test_df = test_df[:5000]
    test_loader = make_dataloader(test_df, batch_size=args.batch_size, shuffle=False, transform=None, mode='predict')

    model = UNet()
    fold = 1
    model_path = Path('model_{fold}.pt'.format(fold=fold))
    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    out_pred_rows = []

    for batch_id, (inputs, image_paths) in enumerate(tqdm(test_loader, desc='Predict')):
        inputs = torch.tensor(inputs)
        outputs = model(inputs)
        for i, image_name in enumerate(image_paths):
            mask = F.sigmoid(outputs[i, 0]).data.cpu().numpy()
            cur_seg = binary_opening(mask > 0.5, disk(2))
            cur_rles = multi_rle_encode(cur_seg)
            if len(cur_rles) > 0:
                for c_rle in cur_rles:
                    out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': c_rle}]
            else:
                out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': None}]

    submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
    submission_df.to_csv('submission.csv', index=False)


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
