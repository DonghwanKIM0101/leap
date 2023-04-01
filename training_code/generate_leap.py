import os
import argparse
import logging

import numpy as np
import torch
import torch.optim as optim
import tensorboardX

import config
import checkpoints
import utils

from datasets.humman_seq import TestSampler


def main(cfg, num_workers):
    
    out_dir = cfg['training']['out_dir']
    mesh_out_dir = os.path.join(out_dir, 'mesh')
    batch_size = cfg['training']['batch_size'] # use the same batch size (seq_len)
    utils.save_config(os.path.join(out_dir, 'config.yml'), cfg)

    model_selection_metric = cfg['training']['model_selection_metric']
    model_selection_sign = 1 if cfg['training']['model_selection_mode'] == 'maximize' else -1

    utils.cond_mkdir(mesh_out_dir)

    vis_dataset = config.get_dataset('test', cfg)

    vis_loader = torch.utils.data.DataLoader(
        vis_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=TestSampler(vis_dataset, seq_num=3000)) # WARNING

    # Model
    model = config.get_model(cfg)
    trainer = config.get_trainer(model, None, cfg)

    # Print model
    print(model)
    logger = logging.getLogger(__name__)
    logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')

    # load pretrained model
    ckp = checkpoints.CheckpointIO(out_dir, model, None, cfg)
    try:
        load_dict = ckp.load('model_best.pt')
        logger.info('Model loaded')
    except FileExistsError:
        logger.info('Model NOT loaded')
        load_dict = dict()

    metric_val_best = load_dict.get('loss_val_best', -model_selection_sign * np.inf)

    logger.info(f'Current best validation metric ({model_selection_metric}): {metric_val_best:.6f}')

    eval_dict = trainer.visualize(vis_loader, mesh_out_dir)

    print(eval_dict)

    print('Evaluation done!')

    '''
    metric_val = eval_dict[model_selection_metric]
    logger.info(f'Validation metric ({model_selection_metric}): {metric_val:.8f}')

    eval_dict_path = os.path.join(out_dir, 'eval_dict.yml')
    with open(eval_dict_path, 'w') as f:
        yaml.dump(config, f)

    print(f'Results saved in {eval_dict_path}')
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type=str,
        help='Path to a config file.')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=6,
        help='Number of workers for datasets loaders.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(cfg=config.load_config(args.config),
         num_workers=args.num_workers)
