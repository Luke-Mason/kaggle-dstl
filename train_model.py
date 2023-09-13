#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from pprint import pprint
import attr
import numpy as np
from sklearn.model_selection import ShuffleSplit
import utils
from model import Model
from HyperParams import HyperParams

logger = utils.get_logger(__name__)

def start():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('logdir', help='Path to log directory')
    arg('--hps', help='Change hyperparameters in k1=v1,k2=v2 format')
    arg('--all', action='store_true', help='Train on all images without validation')
    arg('--validation', choices=['random', 'stratified', 'square', 'custom'], default='custom', help='validation strategy')
    arg('--valid-only', action='store_true')
    arg('--only', help='Train on this image ids only (comma-separated) without validation')
    arg('--clean', action='store_true', help='Clean logdir')
    arg('--no-mp', action='store_true', help='Disable multiprocessing')
    arg('--model-path', type=Path)
    args = parser.parse_args()

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
    mask_stats = json.loads(Path('cls-stats.json').read_text())
    im_area = [(im_id, np.mean([mask_stats[im_id][str(cls)]['area']
                                for cls in hps.classes]))
               for im_id in all_im_ids]
    area_by_id = dict(im_area)
    valid_ids = []

    if args.only:
        train_ids = args.only.split(',')
    elif args.all:
        train_ids = all_im_ids
    elif args.validation == 'stratified':
        train_ids, valid_ids = [], []
        for idx, (im_id, _) in enumerate(
                sorted(im_area, key=lambda x: (x[1], x[0]), reverse=True)):
            (valid_ids if (idx % 4 == 1) else train_ids).append(im_id)
    elif args.validation == 'square':
        train_ids = valid_ids = all_im_ids
    elif args.validation == 'random':
        forced_train_ids = {'6070_2_3', '6120_2_2', '6110_4_0'}
        other_ids = list(set(all_im_ids) - forced_train_ids)
        train_ids, valid_ids = [[other_ids[idx] for idx in g] for g in next(
            ShuffleSplit(random_state=1, n_splits=4).split(other_ids))]
        train_ids.extend(forced_train_ids)
    elif args.validation == 'custom':
        valid_ids = ['6140_3_1', '6110_1_2', '6160_2_1', '6170_0_4', '6100_2_2']
        train_ids = [im_id for im_id in all_im_ids if im_id not in valid_ids]
    else:
        raise ValueError('Unexpected validation kind: {}'.format(args.validation))

    if args.valid_only:
        train_ids = []

    train_area_by_class, valid_area_by_class = [
        {cls: np.mean(
            [mask_stats[im_id][str(cls)]['area'] for im_id in im_ids])
         for cls in hps.classes}
        for im_ids in [train_ids, valid_ids]]

    logger.info('Train: {}'.format(' '.join(sorted(train_ids))))
    logger.info('Valid: {}'.format(' '.join(sorted(valid_ids))))
    logger.info('Train area mean: {:.6f}'.format(
        np.mean([area_by_id[im_id] for im_id in valid_ids])))
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
