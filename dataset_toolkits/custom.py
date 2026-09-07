import os
import pandas as pd
from tqdm import tqdm

def add_args(parser): 
    pass

def get_metadata(**kwargs): 
    return pd.read_csv(kwargs['output_dir'] + '/metadata.csv')

def download(metadata, output_dir, **kwargs): 
    return metadata[['sha256', 'local_path']]

def foreach_instance(metadata, output_dir, func, **kwargs):
    records = []
    for _, row in tqdm(metadata.iterrows(), desc="Processing instances"):
        file_path = os.path.join(output_dir, row['local_path'])
        sha256 = row['sha256']
        res = func(file_path, sha256)
        if res is not None:
            records.append(res)
    return pd.DataFrame.from_records(records)
