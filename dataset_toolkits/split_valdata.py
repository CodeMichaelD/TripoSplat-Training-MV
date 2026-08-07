import os
import sys
import argparse
from easydict import EasyDict as edict
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--num_val', type=int, default=512,)
    opt = parser.parse_args()
    opt = edict(vars(opt))
    
    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))

    output_dir_train = opt.output_dir + '_train'
    output_dir_train_val = opt.output_dir + '_train_val'

    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_train_val, exist_ok=True)
    
    id_train = metadata['sha256'].values[:-opt.num_val]
    id_val = metadata['sha256'].values[-opt.num_val:]
    
    metadata_train = metadata.iloc[:-opt.num_val]
    metadata_val = metadata.iloc[-opt.num_val:]
    
    metadata_train.to_csv(os.path.join(output_dir_train, 'metadata.csv'), index=False)
    metadata_val.to_csv(os.path.join(output_dir_train_val, 'metadata.csv'), index=False)
    
    # create soft links
    # recursively find the the directorys in output_dir
    
    folders = [
        "latents",
        "renders",
        "renders_cond",
        "pcd_features",
    ]
    print(f"Folders in {opt.output_dir}: {folders}")
    for folder in folders:
        if not os.path.exists(os.path.join(opt.output_dir, folder)):
            print(f"Folder {os.path.join(opt.output_dir, folder)} does not exist, skipped")
            continue
        try:
            os.symlink(os.path.join(opt.output_dir, folder), os.path.join(output_dir_train, folder), target_is_directory=True)
            os.symlink(os.path.join(opt.output_dir, folder), os.path.join(output_dir_train_val, folder), target_is_directory=True)
        except FileExistsError:
            print(f"Folder {os.path.join(output_dir_train, folder)} or {os.path.join(output_dir_train_val, folder)} already exists, skipped")
            continue
        print(f"Creating soft link for {folder}...")
