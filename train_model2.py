#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from pprint import pprint
import attr
import numpy as np
import torch
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import utils
from data_loader import Model, SatelliteImageDataset
from HyperParams import HyperParams

logger = utils.get_logger(__name__)
from utils import dataset_path


def start():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('logdir', help='Path to log directory')
    arg('--hps', help='Change hyperparameters in k1=v1,k2=v2 format')
    arg('--all', action='store_true',
        help='Train on all images without validation')
    arg('--validation', choices=['random', 'stratified', 'square', 'custom'],
        default='custom', help='validation strategy')
    arg('--valid-only', action='store_true')
    arg('--only',
        help='Train on this image ids only (comma-separated) without validation')
    arg('--clean', action='store_true', help='Clean logdir')
    arg('--no-mp', action='store_true', help='Disable multiprocessing')
    arg('--model-path', type=Path)
    arg('--data-path', type=Path, default=dataset_path)
    args = parser.parse_args()
    print('test')
    logdir = Path(args.logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    if args.clean:
        for p in logdir.iterdir():
            p.unlink()

    if args.hps == 'load':
        hps = HyperParams.from_dir(logdir)
    else:
        hps = HyperParams()
        hps.update(args.hps)
        logdir.joinpath('hps.json').write_text(
            json.dumps(attr.asdict(hps), indent=True, sort_keys=True))
    pprint(attr.asdict(hps))

    model = Model(hps=hps)
    all_im_ids = list(utils.get_wkt_data())



    # Get the training ids of the images from the dataset classes stats file.
    mask_stats = json.loads(Path('cls-stats.json').read_text())


    valid_ids = []

    if args.only:
        train_ids = args.only.split(',')
    elif args.all:
        train_ids = all_im_ids
    #     Checks the validation of the accuracy based off the IoU, dice loss
    #     jaccard etc which based on the pixel distance values and the
    #     bounding box pixel positions from the original.
    elif args.validation == 'square':
        train_ids = valid_ids = all_im_ids
    #     Checks the validation based off the random selection of the images
    elif args.validation == 'stratified':
        # Get the mean area of the classes for each image.
        im_area = [(im_id, np.mean([mask_stats[im_id][str(cls)]['area']
                                    for cls in hps.classes]))
                   for im_id in all_im_ids]
        area_by_id = dict(im_area)

        logger.info('Train area mean: {:.6f}'.format(
            np.mean([area_by_id[im_id] for im_id in valid_ids])))

        # Split the images into train and validation sets by the area.
        train_ids, valid_ids = [], []
        for idx, (im_id, _) in enumerate(
                sorted(im_area, key=lambda x: (x[1], x[0]), reverse=True)):
            (valid_ids if (idx % 4 == 1) else train_ids).append(im_id)
    else:
        raise ValueError(
            'Unexpected validation kind: {}'.format(args.validation))

    if args.valid_only:
        train_ids = []

    transform_pipline = []

    if self.hps.needs_dist:
        dist_mask = im.dist_mask[:, x - margin: x + p + margin,
                    y - margin: y + p + margin]

    if hps.augment_flips:
        [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
        if random.random() < 0.5:
            transforms.RandomHorizontalFlip(),
            patch = np.flip(patch, 1)
            mask = np.flip(mask, 1)
            if self.hps.needs_dist:
                dist_mask = np.flip(dist_mask, 1)
        if random.random() < 0.5:
            patch = np.flip(patch, 2)
            mask = np.flip(mask, 2)
            if self.hps.needs_dist:
                dist_mask = np.flip(dist_mask, 2)

    if self.hps.augment_rotations:
        assert self.hps.augment_rotations != 1  # old format
        angle = (2 * random.random() - 1.) * self.hps.augment_rotations
        patch = utils.rotate(patch, angle)
        mask = utils.rotate(mask, angle)
        if self.hps.needs_dist:
            dist_mask = utils.rotate(dist_mask, angle)

    if self.hps.augment_channels:
        ch_shift = np.random.normal(
            1, self.hps.augment_channels, patch.shape[0])
        patch = patch * ch_shift[:, None, None]

    inputs.append(patch[:, margin: -margin, margin: -margin].astype(np.float32))
    outputs.append(mask[:, margin: -margin, margin: -margin].astype(np.float32))
    if self.hps.needs_dist:
        dist_outputs.append(
            dist_mask[:, margin: -margin, margin: -margin].astype(np.float32))

    transform_pipline.append(transforms.ToTensor())
    transform = transforms.Compose(transform_pipline)

    data_paths = [args.data_path.joinpath(im_id + '.tif') for im_id in train_ids]

    dataset = SatelliteImageDataset(data_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=load_from_disk)

    model = Model(hps)
    model.train(dataloader, None)

    train_area_by_class, valid_area_by_class = [
        {cls: np.mean(
            [mask_stats[im_id][str(cls)]['area'] for im_id in im_ids])
            for cls in hps.classes}
        for im_ids in [train_ids, valid_ids]]

    logger.info('Train: {}'.format(' '.join(sorted(train_ids))))
    logger.info('Valid: {}'.format(' '.join(sorted(valid_ids))))

    logger.info('Train area by class: {}'.format(
        ' '.join('{}: {:.6f}'.format(cls, train_area_by_class[cls])
                 for cls in hps.classes)))
    logger.info('Valid area mean: {:.6f}'.format(
        np.mean([area_by_id[im_id] for im_id in train_ids])))
    logger.info('Valid area by class: {}'.format(
        ' '.join('cls-{}: {:.6f}'.format(cls, valid_area_by_class[cls])
                 for cls in hps.classes)))

    model.train(logdir=logdir,
                train_ids=train_ids,
                valid_ids=valid_ids,
                validation=args.validation,
                no_mp=args.no_mp,
                validate_only=args.valid_only,
                model_path=args.model_path
                )

# Custom collate function to load batches from disk
def load_from_disk(batch):
    return torch.stack(batch)

# Main
if __name__ == '__main__':
    start()