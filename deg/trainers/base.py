from abc import abstractmethod
import os
import time
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np
import einops

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms.functional as TF

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler

from tqdm import tqdm

class Trainer:
    """
    Base class for training.
    """
    def __init__(self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode='inflat_all',
        fp16_scale_growth=1e-3,
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        dataset_train_val=None,
        gradient_checkpointing=False,
        shuffle_data=True,
        i_print=1000,
        i_log=500,
        i_train_val=10000,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        snapshot_num_samples=64,
        snapshot_dataset_num_samples=100,
        num_workers=None,
        use_wandb=False,
        wandb_run_name=None,
        wandb_project_name=None,
        wandb_run_tags=None,
        graceful_stop=False,
        debug=False,
        prefetch_factor=None,
        sync_data=False, 
        **kwargs
    ):
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.models = models
        self.dataset = dataset
        self.dataset_train_val = dataset_train_val
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = ema_rate if isinstance(ema_rate, list) else [ema_rate]
        self.fp16_mode = fp16_mode
        self.fp16_scale_growth = fp16_scale_growth
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        self.snapshot_num_samples = snapshot_num_samples
        self.snapshot_dataset_num_samples = snapshot_dataset_num_samples
        self.num_workers = num_workers if num_workers is not None else int(np.ceil(os.cpu_count() / torch.cuda.device_count()))
        self.gradient_checkpointing = gradient_checkpointing
        self.use_wandb = use_wandb
        
        # LoRA configuration
        self.train_lora_only = kwargs.get('train_lora_only', False)
        self.lora_rank = kwargs.get('lora_rank', 16)
        self.lora_alpha = kwargs.get('lora_alpha', 1.0)
        self.lora_blocks = kwargs.get('lora_blocks', list(range(20, 24)))
        
        self.graceful_stop = graceful_stop
        self.shuffle_data = shuffle_data
        self.prefetch_factor = prefetch_factor
        self.sync_data = sync_data
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_train_val = i_train_val
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck
        self.debug = debug    

        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        assert self.batch_size % self.world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        assert self.batch_size_per_gpu % self.batch_split == 0, 'Batch size per GPU must be divisible by batch split.'

        self.init_models_and_more(finetune_ckpt=finetune_ckpt, **kwargs)
        self.prepare_dataloader(sync_data=sync_data, **kwargs)
        
        # Load checkpoint
        self.step = 0
        
        if finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'tb_logs'))
            if self.use_wandb:
                import wandb
                wandb_project_name = os.path.basename(os.getcwd()) if wandb_project_name is None else wandb_project_name
                wandb_run_name = os.path.basename(self.output_dir) if wandb_run_name is None else wandb_run_name
                wandb_run_tags = wandb_run_name.split("-") if wandb_run_tags is None else wandb_run_tags
                wandb.init(
                    project=wandb_project_name,
                    name=wandb_run_name,
                    tags=wandb_run_tags,
                    dir=os.path.join(self.output_dir, 'wandb_logs')
                )
                    

        if self.world_size > 1:
            self.check_ddp()
            
        if self.is_master:
            print('\n\nTrainer initialized.')
            print(self)
            
    def set_models_train(self):
        """
        Set models to train mode.
        """
        for _, model in self.models.items():
            model.train()
            
    def set_models_eval(self):
        """
        Set models to eval mode.
        """
        for _, model in self.models.items():
            model.eval()
            
    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, 'device'):
                return model.device
        return next(list(self.models.values())[0].parameters()).device
            
    @abstractmethod
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        pass
    
    def prepare_dataloader(self, sync_data=False, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=self.shuffle_data,
            sync=sync_data,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            sampler=self.data_sampler,
            prefetch_factor=self.prefetch_factor,
        )
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(getattr(self.data_sampler, 'epoch', 0))
        self.data_iterator = cycle(self.dataloader)
        if self.dataset_train_val is not None:
            self.data_sampler_train_val = ResumableSampler(self.dataset_train_val)
            self.dataloader_train_val = DataLoader(
                self.dataset_train_val,
                batch_size=self.batch_size_per_gpu,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=self.num_workers > 0,
                collate_fn=self.dataset_train_val.collate_fn if hasattr(self.dataset_train_val, 'collate_fn') else None,
                sampler=self.data_sampler_train_val,
                prefetch_factor=self.prefetch_factor,
            )

    @abstractmethod
    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        pass
    
    @abstractmethod
    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        pass
    
    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        """
        pass
    
    def run_train_val_step(self, init=False, **kwargs):
        """
        Run a training validation step.
        return keys value, reduce
        """
        return {}
    
    def run_train_val(self, init=False, **kwargs):
        """
        Run a snapshot of the model.
        """
        if self.dataset_train_val is None:
            if self.is_master:
                print('\n\nNo validation dataset provided, skipping validation.')
            return {}
        
        if self.is_master:
            print('\n\nRunning validation...\n')
        step_log = self.run_train_val_step(init=init, **kwargs)
        # sort the keys
        step_log = {k: step_log[k] for k in sorted(step_log.keys())}
        
        # Filter out images and save them
        metrics = {}
        for key, value in step_log.items():
            if isinstance(value['value'], Image.Image):
                if self.is_master:
                    os.makedirs(os.path.join(self.output_dir, 'train_val'), exist_ok=True)
                    try:
                        value['value'].save(os.path.join(self.output_dir, 'train_val', f'{key}_step{self.step:07d}.webp'))
                    except Exception:
                        value['value'].save(os.path.join(self.output_dir, 'train_val', f'{key}_step{self.step:07d}.jpg'))
            else:
                metrics[key] = value
        
        step_log = metrics

        # Gather results
        if self.world_size > 1:
            for key in step_log.keys():
                step_log[key]['value'] = step_log[key]['value'].contiguous()
                if self.is_master:
                    all_values = [torch.empty_like(step_log[key]['value']) for _ in range(self.world_size)]
                else:
                    all_values = []
                dist.gather(step_log[key]['value'], all_values, dst=0)
                if self.is_master:
                    step_log[key]['value'] = einops.reduce(torch.stack(all_values, dim=0), 'B ... -> ...', step_log[key]['reduce'])
        step_log_item = {
            key: value['value'].item() if isinstance(value['value'], torch.Tensor) else value['value']
            for key, value in step_log.items()
        }
        return step_log_item
            

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        """
        if hasattr(self.dataset, 'visualize_sample'):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=None):
        """
        Sample images from the dataset.
        """
        num_samples = self.snapshot_dataset_num_samples if num_samples is None else num_samples
        if self.is_master:
            print(f'\nSampling {num_samples} images from dataset...', end='')
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=num_samples,
            num_workers=0,
            shuffle=self.shuffle_data,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        data = next(iter(dataloader))
        data = recursive_to_device(data, self.device)
        vis = self.visualize_sample(data)
        if isinstance(vis, dict):
            save_cfg = [(f'dataset_{k}', v) for k, v in vis.items()]
        else:
            save_cfg = [('dataset', vis)]
        for name, image in save_cfg:
            utils.save_image(
                image,
                os.path.join(self.output_dir, 'samples', f'{name}.jpg'),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=None, batch_size=None, verbose=False):
        """
        Sample images from the model.
        NOTE: This function should be called by all processes.
        """
        batch_size = self.batch_size_per_gpu if batch_size is None else batch_size
        num_samples = self.snapshot_num_samples if num_samples is None else num_samples
        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')

        if suffix is None:
            suffix = f'step{self.step:07d}'

        # Assign tasks
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose)
        # sort the keys
        samples = {k: samples[k] for k in sorted(samples.keys())}

        # Preprocess images
        for key in list(samples.keys()):
            if samples[key]['type'] == 'sample':
                vis = self.visualize_sample(samples[key]['value'])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                    del samples[key]
                else:
                    samples[key] = {'value': vis, 'type': 'image'}

        # Gather results
        if self.world_size > 1:
            for key in samples.keys():
                if samples[key]['type'] == 'pil_image':
                    continue

                if samples[key]['type'] == 'pil_image_concat':
                    samples[key]['value'] = TF.to_tensor(samples[key]['value']).unsqueeze(0).to(self.device)
                    samples[key]['type'] = 'image'

                samples[key]['value'] = samples[key]['value'].contiguous()
                if self.is_master:
                    all_images = [torch.empty_like(samples[key]['value']) for _ in range(self.world_size)]
                else:
                    all_images = []
                dist.gather(samples[key]['value'], all_images, dst=0)
                if self.is_master:
                    samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples]

        # Save images
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            for key in samples.keys():
                if samples[key]['type'] == 'pil_image':
                    try:
                        samples[key]['value'].save(os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.webp'))
                    except Exception as e:
                        samples[key]['value'].save(os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'))
                elif samples[key]['type'] == 'image':
                    B, C, H, W = samples[key]['value'].shape
                    if H == W:
                        make_grid_kwargs = {"nrow": int(np.sqrt(num_samples)) }
                    elif H > W:
                        make_grid_kwargs  = {"nrow": num_samples}
                    else:
                        make_grid_kwargs = {"nrow": 1}
                    try:
                        utils.save_image(
                            samples[key]['value'],
                            os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.webp'),
                            normalize=True,
                            value_range=self.dataset.value_range,
                            **make_grid_kwargs
                        )
                    except Exception as e:
                        utils.save_image(
                            samples[key]['value'],
                            os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                            normalize=True,
                            value_range=self.dataset.value_range,
                            **make_grid_kwargs
                        )
                elif samples[key]['type'] == 'number':
                    min = samples[key]['value'].min()
                    max = samples[key]['value'].max()
                    images = (samples[key]['value'] - min) / (max - min)
                    images = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,
                    )
                    save_image_with_notes(
                        images,
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.webp'),
                        notes=f'{key} min: {min}, max: {max}',
                    )

        if self.is_master:
            print(' Done.')

    @abstractmethod
    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        pass

    @abstractmethod
    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        pass

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses.
        """
        pass
    
    def load_data(self):
        """
        Load data.
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        
        # if the data is a dict, we need to split it into multiple dicts with batch_size_per_gpu
        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0]
                data_list = [
                    {k: v[i * batch_size // self.batch_split:(i + 1) * batch_size // self.batch_split] for k, v in data.items()}
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError('Data must be a dict or a list of dicts.')
        
        return data_list

    @abstractmethod
    def run_step(self, data_list):
        """
        Run a training step.
        """
        pass
    
    def write_log(self, log):
        if len(log) == 0:
            return
        ## save to log file
        log_str = '\n'.join([
            f'{step}: {json.dumps(log)}' for step, log in log
        ])
        with open(os.path.join(self.output_dir, 'log.txt'), 'a') as log_file:
            log_file.write(log_str + '\n')

        # show with mlflow
        log_show = [l for _, l in log if not dict_any(l, lambda x: np.isnan(x))]
        log_show = dict_reduce(log_show, lambda x: np.mean(x))
        log_show = dict_flatten(log_show, sep='/')
        for key, value in log_show.items():
            self.writer.add_scalar(key, value, self.step)
        if self.use_wandb:
            import wandb
            wandb.log(log_show, step=self.step)

    def run(self):
        """
        Run training.
        """
        self.set_models_eval()
        if self.is_master:
            print('\nStarting training...')
            self.snapshot_dataset()
        if self.step == 0:
            self.snapshot(suffix='init')
            snapshot_log = self.run_train_val(init=True)
        else: # resume
            self.snapshot(suffix=f'resume_step{self.step:07d}')
            snapshot_log = self.run_train_val()

        log = [(self.step, snapshot_log)]
        time_last_print = 0.0
        time_elapsed = 0.0
        graceful_stop_triggered = False
        debug_stop_step = self.step + 100 if self.debug else None
        for _ in tqdm(range(self.step, self.max_steps), total=self.max_steps, initial=self.step, ncols=80, desc="Training..."):
            time_start = time.time()

            self.set_models_train()
            data_list = self.load_data()
            step_log = self.run_step(data_list)
            self.set_models_eval()

            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1

            # Print progress
            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f'Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)',
                    f'Elapsed: {time_elapsed / 3600:.2f} h',
                    f'Speed: {speed:.2f} steps/h',
                    f'ETA: {(self.max_steps - self.step) / speed:.2f} h',
                ]
                print(' | '.join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            # Check ddp
            if self.world_size > 1 and self.i_ddpcheck is not None and (self.step % self.i_ddpcheck == 0 or self.step == 100):
                self.check_ddp()

            # Sample images
            is_sample_step = False
            if self.step < self.i_sample:
                for sample_pow in range(6):
                    if self.step == self.i_sample // (2 ** sample_pow):
                        is_sample_step = True
                        break
            if self.step % self.i_sample == 0 or is_sample_step:
                self.snapshot()

            if self.is_master:
                log.append((self.step, {}))

            # Run train validation
            if self.step % self.i_train_val == 0:
                if self.dataset_train_val is None:
                    if self.is_master:
                        print('\n\nNo validation dataset provided, skipping validation.')
                else:
                    val_log = self.run_train_val()
                    if self.is_master:
                        log[-1][1].update(val_log)
                        log[-1][1]['val_step'] = self.step
                        
            if self.is_master:
                # Log time
                log[-1][1]['time'] = {
                    'step': time_end - time_start,
                    'elapsed': time_elapsed,
                    'sample_preceeded': self.world_size * self.batch_size_per_gpu * self.step,
                }
                log[-1][1]['meta'] = {
                    'num_gpus': self.world_size,
                }

                # Log losses
                if step_log is not None:
                    log[-1][1].update(step_log)

                # Log scale
                if self.fp16_mode == 'amp':
                    log[-1][1]['scale'] = self.scaler.get_scale()
                elif self.fp16_mode == 'inflat_all':
                    log[-1][1]['log_scale'] = self.log_scale
                    if hasattr(self, 'log_scale_fake'):
                        log[-1][1]['log_scale_fake'] = self.log_scale_fake

                # Save log
                if self.step % self.i_log == 0:
                    self.write_log(log)
                    log = []

                # Save checkpoint
                if self.step % self.i_save == 0:
                    self.save()
                    
            if self.debug and self.step >= debug_stop_step:
                print('Debug passed 100 steps!')
                play_beeps()
                break

        self.snapshot(suffix='final' if not graceful_stop_triggered else f'graceful_stop_step{self.step:07d}')
        if self.is_master:
            self.write_log(log)
            self.writer.close()
            print('Training finished.')
            
    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop.
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, 'profile')),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()
            
