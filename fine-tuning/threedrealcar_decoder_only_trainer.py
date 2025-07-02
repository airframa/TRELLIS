import os
import sys
import copy
import torch
import numpy as np
import utils3d.torch
from typing import Dict, Tuple
from easydict import EasyDict as edict
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
        pretrained_encoder_path: str = None,
        freeze_encoder: bool = True,
        **kwargs
    ):
        self.pretrained_encoder_path = pretrained_encoder_path
        self.freeze_encoder = freeze_encoder
        super().__init__(*args, **kwargs)
        
    def init_models_and_more(self, **kwargs):
        """
        Initialize models with special handling for decoder-only training.
        """
        # First, initialize as normal
        super().init_models_and_more(**kwargs)
        
        # Since we're using pre-computed latents, we don't need the encoder
        # Just ensure only decoder parameters are trainable
        print("Setting up decoder-only training")
        
        # Freeze encoder (even if it's a dummy)
        for param in self.models['encoder'].parameters():
            param.requires_grad = False
        
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
            # Get base LR from config
            lr = self.optimizer_config['args'].get('lr', 5e-05)
            
            # Scale LR by number of GPUs
            if self.world_size > 1:
                lr *= self.world_size
            
            # Create new args dict without invalid parameters
            optimizer_args = {k: v for k, v in self.optimizer_config['args'].items() 
                            if k not in ['lr_scale_by_gpus']}
            optimizer_args['lr'] = lr
            
            self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(
                self.master_params, **optimizer_args
            )
        
        # Update EMA params if master
        if self.is_master:
            import copy
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]
        
        print(f"Decoder-only training configured with {len(self.model_params)} parameter groups")
    
    def training_losses(self, *args, **kwargs) -> Tuple[Dict, Dict]:
        """
        Override to handle pre-computed latents from the dataset.
        """
        # The dataset already provides latents, not features
        # So we directly use them instead of encoding
        latents = kwargs.get('latents')
        
        # Remove 'latents' from kwargs and pass the rest
        kwargs_without_latents = {k: v for k, v in kwargs.items() if k != 'latents'}
        
        # Call our modified training losses with pre-computed latents
        return self._training_losses_with_z(latents, **kwargs_without_latents)
    
    def _training_losses_with_z(
        self,
        z,  # Pre-computed latents
        image: torch.Tensor,
        alpha: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_aux: bool = False,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Modified training losses that accepts pre-computed latents.
        """
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
        # Add this at the start of the function:
        torch.cuda.empty_cache()
        dataloader = torch.utils.data.DataLoader(
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
            
            # Use pre-computed latents directly
            z = args['latents']
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
        self.renderer.rendering_options.resolution = 256 # 512
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