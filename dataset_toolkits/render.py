import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence, set_permissions_recursively

# ----------------------------------------------------------------------
# Removed all Blender‑specific constants and installation functions
# ----------------------------------------------------------------------

def _render(file_path, sha256, output_dir, num_views, engine, output_folder,
            new_light=False, verbose=False, unit_radius=False):
    """
    Render a 3D model using the OpenGL‑based renderer (render_gl.py).
    This replaces the original Blender call.
    """
    output_folder = os.path.join(output_dir, output_folder, sha256)

    # Build camera parameters (same as original)
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [
        {
            'yaw': y, 'pitch': p, 'radius': r, 'fov': f,
            'ortho': False, 'ortho_scale': None
        }
        for y, p, r, f in zip(yaws, pitchs, radius, fov)
    ]

    # Path to our new render script (placed in blender_script/)
    script_path = os.path.join(os.path.dirname(__file__), 'blender_script', 'render_gl.py')

    # Build command line
    args = [
        'python', script_path,
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', os.path.expanduser(output_folder),
        '--save_depth',   # always save depth to match original behaviour
    ]
    if unit_radius:
        args.append('--unit_radius')
    # The original also passed --save_mesh and --engine; our script ignores engine,
    # but we can pass it for compatibility (it will be ignored).
    args.extend(['--engine', engine])
    if new_light:
        args.append('--new_light')  # our script ignores it

    if verbose:
        print(' '.join(args))
        ret = call(args)
    else:
        ret = call(args, stdout=DEVNULL, stderr=DEVNULL)

    if ret == 0 and os.path.exists(os.path.join(output_folder, 'transforms.json')):
        set_permissions_recursively(output_folder)
        return {'sha256': sha256, 'rendered': True}
    return None


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    dataset_utils.add_args(parser)
    parser.add_argument('--engine', type=str, default='BLENDER_EEVEE',
                        help='Ignored (kept for compatibility)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=64)
    parser.add_argument('--unit_radius', action='store_true',
                        help='Normalize to unit radius.')
    parser.add_argument('--new_light', action='store_true',
                        help='Ignored (kept for compatibility)')

    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    output_folder = "verbose/renders" if opt.verbose else "renders"

    os.makedirs(os.path.join(opt.output_dir, output_folder), exist_ok=True)

    # Load metadata
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))

    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' in metadata.columns and not opt.verbose:
            metadata = metadata[metadata['rendered'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    # Distribute work across processes
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    records = []

    # Skip already rendered objects
    if opt.verbose:
        metadata = metadata[:100]   # debug mode: only first 100
        import time
        start_time = time.time()
    else:
        for sha256 in copy.copy(metadata['sha256'].values):
            if os.path.exists(os.path.join(opt.output_dir, output_folder, sha256, 'transforms.json')):
                records.append({'sha256': sha256, 'rendered': True})
                metadata = metadata[metadata['sha256'] != sha256]

    print(f'Processing {len(metadata)} objects...')

    # Render each object in parallel
    func = partial(
        _render,
        output_dir=opt.output_dir,
        num_views=opt.num_views,
        engine=opt.engine,
        output_folder=output_folder,
        new_light=opt.new_light,
        verbose=opt.verbose,
        unit_radius=opt.unit_radius
    )
    rendered = dataset_utils.foreach_instance(
        metadata, opt.output_dir, func,
        max_workers=opt.max_workers,
        desc='Rendering objects'
    )

    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    if (not opt.verbose) and opt.num_views > 0:
        rendered.to_csv(os.path.join(opt.output_dir, f'rendered_{opt.rank}.csv'), index=False)
    else:
        end_time = time.time()
        print(f'Rendering {len(metadata)} objects took {end_time - start_time} seconds')
