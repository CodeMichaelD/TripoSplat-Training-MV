import os
import sys
import glob
import argparse
from deg.utils.general_utils import edict

import torch
import torch.multiprocessing as mp
import numpy as np
import random
import subprocess
import tarfile
import tempfile
import os
from deg import models, datasets, trainers
from deg.utils.dist_utils import setup_dist, endup_dist

def copy_git_snapshot(output_dir):
    print("Snapshotting code (make sure ignore files in .gitignore)...")
    dest_tar_path = ""
    untar_cmd = ""
    try:
        base_dir = os.path.join(output_dir, "code_snapshot")
        os.makedirs(base_dir, exist_ok=True)
        for ver in range(10):
            dest_tar_path = os.path.join(output_dir, f'code_snapshot_v{ver}.tar.gz')
            if not os.path.exists(dest_tar_path):
                break
        untar_path = base_dir
        
        # Get current working directory as repo_dir
        repo_dir = os.getcwd()

        # Step 1a: Get tracked files (use -C to avoid safe.directory issues)
        result_tracked = subprocess.run(
            ['git', '-C', repo_dir, 'ls-files'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        tracked_files = result_tracked.stdout.strip().split('\n')

        # Step 1b: Get untracked files (non-ignored)
        result_untracked = subprocess.run(
            ['git', '-C', repo_dir, 'ls-files', '-o', '--exclude-standard'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        untracked_files = result_untracked.stdout.strip().split('\n')

        # Step 1c: Get deleted tracked files
        result_deleted = subprocess.run(
            ['git', '-C', repo_dir, 'ls-files', '--deleted'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        deleted_files = set(result_deleted.stdout.strip().split('\n'))

        # Combine tracked + untracked, exclude deleted
        all_files = set(tracked_files + untracked_files)
        all_files.discard('')
        all_files -= deleted_files  # EXCLUDE deleted tracked files

        if not all_files:
            print("No files to include in snapshot.")
            return

        # Step 2: Create tar.gz archive
        subprocess.run(
            ['tar', '-czf', dest_tar_path, '-T', '-'],
            input='\n'.join(all_files),
            text=True,
            cwd=repo_dir,
            check=True
        )

        # Step 3: Print the untar command
        untar_cmd = f"tar -xzf {dest_tar_path} -C {untar_path}"
    except Exception as e:
        print(f"Warning: Error snapshotting code: {e}")
        print(f"Warning: snapshotting code failed, please check your code and retry.")
        print(f"Warning: If you get this snapshotting error, please try commit your code and retry.")
    print(f"\n✅ Snapshot archive created at: {dest_tar_path}")
    print(f"💡 To untar it later, run:\n\n    {untar_cmd}\n")
    return untar_cmd

def find_ckpt(cfg):
    # Load checkpoint
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            files = glob.glob(os.path.join(cfg.load_dir, 'ckpts', 'misc_*.pt'))
            if len(files) != 0:
                cfg.load_ckpt = max([
                    int(os.path.basename(f).split('step')[-1].split('.')[0])
                    for f in files
                ])
        elif cfg.ckpt == 'none':
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def get_model_summary(model):
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += '\n'
    model_summary += f'Number of parameters: {num_params}\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params}\n'
    return model_summary


def main(local_rank, cfg):
    # Set up distributed training
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    # Seed rngs
    setup_rng(rank)

    # Load data
    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)
    if cfg.data_train_val_dir is not None:
        dataset_train_val = getattr(datasets, cfg.train_val_dataset.name)(cfg.data_train_val_dir, **cfg.train_val_dataset.args)
    else:
        dataset_train_val = None
    
    # Build model
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }

    # Code copy
    if rank == 0:
        untar_cmd = copy_git_snapshot(cfg.output_dir)
        with open(os.path.join(cfg.output_dir, 'command_untar_code_snapshot.txt'), 'w') as fp:
            print(untar_cmd, file=fp)

    # Build trainer
    trainer = getattr(trainers, cfg.trainer.name)(model_dict, dataset, **cfg.trainer.args, output_dir=cfg.output_dir, load_dir=cfg.load_dir, step=cfg.load_ckpt, dataset_train_val=dataset_train_val, debug=cfg.debug)

    # Model summary
    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            # print(f'\n\nBackbone: {name}\n' + model_summary)
            with open(os.path.join(cfg.output_dir, f'{name}_model_summary.txt'), 'w') as fp:
                print(model_summary, file=fp)
    
    # Log sync command
    if rank == 0:
        log_tb_dir = os.path.join(cfg.output_dir, "tb_logs")
        log_run_id = cfg.output_dir.split('/')[-1]
        with open(os.path.join(cfg.output_dir, 'command_log_sync.txt'), 'w') as fp:
            print(f'wandb sync {log_tb_dir} --id {log_run_id}', file=fp)
    # Train
    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()
    # End distributed training
    endup_dist()


if __name__ == '__main__':
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument('--config', type=str, required=True, help='Experiment config file')
    ## io and resume
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory, default to output_dir')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step to resume training, default to latest')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory')
    parser.add_argument('--data_train_val_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--auto_retry', type=int, default=0, help='Number of retries on error')
    ## dubug
    parser.add_argument('--tryrun', action='store_true', help='Try run without training')
    parser.add_argument('--profile', action='store_true', help='Profile training')
    parser.add_argument('--debug', action='store_true', help='Debug mode, single GPU, no multiprocessing, no distributed')
    ## multi-node and multi-gpu
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs per node, default to all')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12333', help='Port for distributed training')
    opt = parser.parse_args()
    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    if opt.debug:
        opt.num_gpus = 1
        os.environ["WANDB_MODE"] = "offline"
    ## Load config
    if opt.config.endswith('.json'):
        import json
        config = json.load(open(opt.config, 'r'))
    elif opt.config.endswith('.yaml'):
        import yaml
        import re
        # Add resolver for scientific notation floats
        yaml.FullLoader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(r'''^(?:[-+]?(?:[0-9][0-9_]*)?\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                        |[-+]?[0-9][0-9_]*(?:[eE][-+]?[0-9]+)
                        |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?)$''', re.X),
            list('-+0123456789.')
        )
        config = yaml.load(open(opt.config, 'r'), Loader=yaml.FullLoader)
    else:
        raise ValueError('Unsupported config file format')
    ## Combine arguments and config
    import json
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    print('\n\nConfig:')
    print('=' * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    # Prepare output directory
    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        ## Save command and config
        with open(os.path.join(cfg.output_dir, 'command.txt'), 'w') as fp:
            print(' '.join(['python'] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as fp:
            json.dump(config, fp, indent=4)

    # Run
    if cfg.auto_retry == 0:
        cfg = find_ckpt(cfg)
        if cfg.num_gpus > 1:
            mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main(0, cfg)
    else:
        for rty in range(cfg.auto_retry):
            try:
                cfg = find_ckpt(cfg)
                if cfg.num_gpus > 1:
                    mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
                else:
                    main(0, cfg)
                break
            except Exception as e:
                print(f'Error: {e}')
                print(f'Retrying ({rty + 1}/{cfg.auto_retry})...')
            