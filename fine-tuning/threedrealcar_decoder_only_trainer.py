import os
import sys
import copy
import torch
import utils3d.torch
import numpy as np
from typing import Dict, Tuple
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

# Add the TRELLIS root directory to sys.path
trellis_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, trellis_root)

from trellis.trainers.vae.structured_latent_vae_gaussian import SLatVaeGaussianTrainer
from trellis import models


class DecoderOnlyFinetuneTrainer(SLatVaeGaussianTrainer):
    """
    Modified trainer for decoder-only fine-tuning.
    Loads pre-trained encoder weights and freezes them.
    """
    
    def __init__(
        self,
        *args,
        pretrained_decoder_path: str = None,
        val_dataset=None,
        **kwargs
    ):
        self.pretrained_decoder_path = pretrained_decoder_path
        self.val_dataset = val_dataset
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Extract validation-specific parameters before passing to parent
        self.i_val = kwargs.pop('i_val', 500)
        self.early_stopping_config = kwargs.pop('early_stopping', {})
        
        super().__init__(*args, **kwargs)
        
        # Load pretrained decoder weights if provided
        if self.pretrained_decoder_path and os.path.exists(self.pretrained_decoder_path):
            print(f"\nLoading pretrained decoder from: {self.pretrained_decoder_path}")
            state_dict = torch.load(self.pretrained_decoder_path, map_location='cpu')
            
            # Check if decoder is properly initialized
            decoder_state = self.models['decoder'].state_dict()
            print(f"Current decoder has {len(decoder_state)} parameters")
            print(f"Pretrained decoder has {len(state_dict)} parameters")
            
            # Load the pretrained weights
            missing_keys, unexpected_keys = self.models['decoder'].load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in pretrained decoder: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in pretrained decoder: {unexpected_keys}")
                
            print("Pretrained decoder loaded successfully!")
            
            # Verify the decoder has transformer blocks
            if hasattr(self.models['decoder'], 'blocks'):
                print(f"Decoder has {len(self.models['decoder'].blocks)} transformer blocks")
            else:
                print("WARNING: Decoder has no transformer blocks!")
        
    def init_models_and_more(self, **kwargs):
        """
        Initialize models with special handling for decoder-only training.
        """
        # First, initialize as normal
        super().init_models_and_more(**kwargs)
        
        # Create a dummy encoder that just returns pre-computed latents
        class DummyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = torch.nn.Parameter(torch.zeros(1))
                
            def forward(self, x, sample_posterior=True, return_raw=False):
                # x is already the latent representation (SparseTensor)
                if return_raw:
                    # Return dummy mean and logvar
                    mean = torch.zeros_like(x.feats)
                    logvar = torch.zeros_like(x.feats)
                    return x, mean, logvar
                return x
        
        # Replace encoder with dummy
        self.models['encoder'] = DummyEncoder().cuda()
        
        # Rebuild training models
        if self.world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.training_models = {
                name: DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    bucket_cap_mb=128,
                    find_unused_parameters=False
                )
                for name, model in self.models.items()
            }
        else:
            self.training_models = self.models
        
        # Update model_params to only include decoder
        self.model_params = [
            p for p in self.models['decoder'].parameters() 
            if p.requires_grad
        ]
        
        # Rebuild master params with only decoder params
        if self.fp16_mode == 'inflat_all':
            from trellis.trainers.utils import make_master_params
            self.master_params = make_master_params(self.model_params)
        else:
            self.master_params = self.model_params
        
        # Rebuild optimizer with only decoder params
        if hasattr(torch.optim, self.optimizer_config['name']):
            self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(
                self.master_params, **self.optimizer_config['args']
            )
        
        # Update EMA params if master
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]
        
        print(f"Decoder-only training configured with {len(self.model_params)} parameter groups")
        print(f"Decoder has {sum(p.numel() for p in self.model_params)} trainable parameters")
    
    def training_losses(
        self,
        latents,  # This is a SparseTensor from the dataset
        image: torch.Tensor,
        alpha: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_aux: bool = False,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Override training losses to use pre-computed latents directly.
        The dataset provides 'latents' as a SparseTensor.
        """
        # The latents are already a SparseTensor, just pass through dummy encoder
        z = latents  # No encoding needed, these are pre-computed
        
        # Decode with trainable decoder
        reps = self.training_models['decoder'](z)
        self.renderer.rendering_options.resolution = image.shape[-1]
        render_results = self._render_batch(reps, extrinsics, intrinsics)
        
        terms = edict(loss=0.0, rec=0.0)
        
        rec_image = render_results['color']
        gt_image = image * alpha[:, None] + (1 - alpha[:, None]) * render_results['bg_color'][..., None, None]
        
        # Compute losses (same as parent)
        if self.loss_type == 'l1':
            terms["l1"] = torch.nn.functional.l1_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l1"]
        elif self.loss_type == 'l2':
            terms["l2"] = torch.nn.functional.mse_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l2"]
        
        if self.lambda_ssim > 0:
            from trellis.utils.loss_utils import ssim
            terms["ssim"] = 1 - ssim(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_ssim * terms["ssim"]
        
        if self.lambda_lpips > 0:
            from trellis.utils.loss_utils import lpips
            terms["lpips"] = lpips(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_lpips * terms["lpips"]
        
        terms["loss"] = terms["loss"] + terms["rec"]
        
        # No KL loss since we're not training the encoder
        terms["kl"] = torch.tensor(0.0).cuda()
        
        # Regularization losses
        reg_loss, reg_terms = self._get_regularization_loss(reps)
        terms.update(reg_terms)
        terms["loss"] = terms["loss"] + reg_loss
        
        status = self._get_status(z, reps)
        
        if return_aux:
            return terms, status, {'rec_image': rec_image, 'gt_image': gt_image}
        return terms, status
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        """
        Override run_snapshot to work with pre-computed latents.
        """
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        ret_dict = {}
        gt_images = []
        exts = []
        ints = []
        reps = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() for k, v in data.items()}
            gt_images.append(args['image'] * args['alpha'][:, None])
            exts.append(args['extrinsics'])
            ints.append(args['intrinsics'])
            
            # Use 'latents' key - it's a SparseTensor
            if 'latents' in args:
                z = args['latents']  # Already a SparseTensor, no encoding needed
            else:
                raise KeyError(f"'latents' not found in data. Keys: {list(args.keys())}")
                
            reps.extend(self.models['decoder'](z))
            
        gt_images = torch.cat(gt_images, dim=0)
        ret_dict.update({f'gt_image': {'value': gt_images, 'type': 'image'}})

        # render single view
        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)
        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = gt_images.shape[-1]
        render_results = self._render_batch(reps, exts, ints)
        ret_dict.update({f'rec_image': {'value': render_results['color'], 'type': 'image'}})

        # render multiview
        self.renderer.rendering_options.resolution = 512
        ## Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        ## render each view
        miltiview_images = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            extrinsics = extrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            intrinsics = intrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            render_results = self._render_batch(reps, extrinsics, intrinsics)
            miltiview_images.append(render_results['color'])

        ## Concatenate views
        miltiview_images = torch.cat([
            torch.cat(miltiview_images[:2], dim=-2),
            torch.cat(miltiview_images[2:], dim=-2),
        ], dim=-1)
        ret_dict.update({f'miltiview_image': {'value': miltiview_images, 'type': 'image'}})

        self.renderer.rendering_options.bg_color = 'random'
                                    
        return ret_dict
    
    @torch.no_grad()
    def run_validation(self):
        """Run validation and return average loss"""
        if self.val_dataset is None:
            return None
        
        # Create validation dataloader
        val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_per_gpu,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.val_dataset.collate_fn if hasattr(self.val_dataset, 'collate_fn') else None,
        )
        
        val_losses = []
        num_batches = min(20, len(val_dataloader))
        
        for i, data in enumerate(val_dataloader):
            if i >= num_batches:
                break
                
            args = {k: v.cuda(self.local_rank, non_blocking=True) for k, v in data.items()}
            
            # Use pre-computed latents
            terms, _ = self.training_losses(
                args['latents'],
                args['image'],
                args['alpha'],
                args['extrinsics'],
                args['intrinsics']
            )
            
            val_losses.append(terms['loss'].item())
        
        if not val_losses:
            return None
            
        avg_val_loss = np.mean(val_losses)
        
        # Check for early stopping
        if avg_val_loss < self.best_val_loss - self.early_stopping_config.get('min_delta', 0.0001):
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0
            
            # Save best model
            if self.is_master:
                torch.save(
                    self.models['decoder'].state_dict(),
                    os.path.join(self.output_dir, 'ckpts', 'best_decoder.pt')
                )
        else:
            self.patience_counter += 1
        
        return avg_val_loss

    def log(self, writer=None):
        """Override to add validation logging"""
        # Call parent's log method
        super().log(writer)
        
        # Check if it's time for validation
        if hasattr(self, 'i_val') and self.step % self.i_val == 0 and self.val_dataset is not None:
            val_loss = self.run_validation()
            
            if self.is_master and val_loss is not None:
                print(f'\n[Validation] Step {self.step}: Loss = {val_loss:.6f}, Best = {self.best_val_loss:.6f}')
                
                # Log to tensorboard if available
                if writer is not None:
                    writer.add_scalar('val/loss', val_loss, self.step)
                    writer.add_scalar('val/best_loss', self.best_val_loss, self.step)
                    writer.add_scalar('val/patience', self.patience_counter, self.step)
                
                # Check early stopping
                patience = self.early_stopping_config.get('patience', 5000)
                if self.patience_counter >= patience:
                    print(f'\n[Early Stopping] Validation loss has not improved for {self.patience_counter} steps')
                    print(f'Best validation loss: {self.best_val_loss:.6f}')
                    self.finish = True
    
    def save(self):
        """
        Override save to only save decoder weights.
        """
        if not self.is_master:
            return
            
        print(f'\nSaving decoder checkpoint at step {self.step}...', end='')
        
        # Save only decoder
        torch.save(
            self.models['decoder'].state_dict(),
            os.path.join(self.output_dir, 'ckpts', f'decoder_step{self.step:07d}.pt')
        )
        
        # Save EMA decoder
        for i, ema_rate in enumerate(self.ema_rate):
            # Extract decoder params from EMA
            ema_decoder_state = self._master_params_to_state_dicts(self.ema_params[i])['decoder']
            torch.save(
                ema_decoder_state,
                os.path.join(self.output_dir, 'ckpts', f'decoder_ema{ema_rate}_step{self.step:07d}.pt')
            )
        
        # Save misc (optimizer, etc.)
        misc_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'data_sampler': self.data_sampler.state_dict(),
        }
        if self.fp16_mode == 'inflat_all':
            misc_ckpt['log_scale'] = self.log_scale
        if self.lr_scheduler_config is not None:
            misc_ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        if self.elastic_controller_config is not None:
            misc_ckpt['elastic_controller'] = self.elastic_controller.state_dict()
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            misc_ckpt['grad_clip'] = self.grad_clip.state_dict()
        
        torch.save(
            misc_ckpt,
            os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt')
        )
        
        print(' Done.')