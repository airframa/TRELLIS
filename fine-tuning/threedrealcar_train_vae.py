#!/usr/bin/env python3
"""
Training script for TRELLIS decoder fine-tuning with full VAE (encoder + decoder)
"""

import os
os.environ['ATTN_BACKEND'] = 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'
import sys
import json
import argparse


sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.multiprocessing as mp
from easydict import EasyDict as edict

# Add the TRELLIS root directory to sys.path
trellis_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, trellis_root)
from trellis import models, datasets
from trellis.utils.dist_utils import setup_dist
from trellis.trainers.vae.structured_latent_vae_gaussian import SLatVaeGaussianTrainer


def find_ckpt(cfg):
    """Find the latest checkpoint if resuming"""
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            import glob
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


def main_worker(local_rank, cfg):
    # Calculate global rank
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus

    print(f"[Worker {local_rank}] Starting with rank={rank}, world_size={world_size}")

    # Set device FIRST
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Then setup distributed training
    if world_size > 1:
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)
        print(f"[Worker {local_rank}] Distributed setup complete")

    # Set random seeds
    torch.manual_seed(rank + cfg.get('seed', 42))
    torch.cuda.manual_seed_all(rank + cfg.get('seed', 42))

    # Load dataset - SparseFeat2Render for on-the-fly encoding
    dataset = datasets.SparseFeat2Render(
        roots=cfg.data_dir,
        **cfg.dataset.args
    )
    print(f"[Worker {local_rank}] Training dataset loaded: {len(dataset)} samples")

    # Load validation dataset if specified
    val_dataset = None
    if cfg.get('val_dir'):
        val_dataset = datasets.SparseFeat2Render(
            roots=cfg.val_dir,
            **cfg.dataset.args
        )
        print(f"[Worker {local_rank}] Validation dataset loaded: {len(val_dataset)} samples")

    # Build models
    # Build encoder - IMPORTANT: Use pretrained encoder
    encoder = models.ElasticSLatEncoder(**cfg.models.encoder.args)
    encoder = encoder.cuda(local_rank)
    
    # Load pretrained encoder weights
    if cfg.get('pretrained_encoder'):
        print(f"[Worker {local_rank}] Loading pretrained encoder from: {cfg.pretrained_encoder}")
        encoder_state = torch.load(cfg.pretrained_encoder, map_location=f'cuda:{local_rank}')
        encoder.load_state_dict(encoder_state)
        print(f"[Worker {local_rank}] Pretrained encoder loaded successfully")

    # In threedrealcar_train_vae.py, after loading the encoder, add:
    # Freeze encoder parameters
    if cfg.get('freeze_encoder', True):  # Default to True for fine-tuning
        print(f"[Worker {local_rank}] Freezing encoder parameters")
        for param in encoder.parameters():
            param.requires_grad = False
        
        # Verify freezing
        encoder_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"[Worker {local_rank}] Encoder trainable parameters: {encoder_trainable} (should be 0)")
    
    # Build decoder
    decoder = models.ElasticSLatGaussianDecoder(**cfg.models.decoder.args)
    decoder = decoder.cuda(local_rank)
    
    # Load pretrained decoder weights if starting from pretrained
    if cfg.get('pretrained_decoder'):
        print(f"[Worker {local_rank}] Loading pretrained decoder from: {cfg.pretrained_decoder}")
        decoder_state = torch.load(cfg.pretrained_decoder, map_location=f'cuda:{local_rank}')
        decoder.load_state_dict(decoder_state)
        print(f"[Worker {local_rank}] Pretrained decoder loaded successfully")

    model_dict = {
        'encoder': encoder,
        'decoder': decoder
    }

    # Model summary (only on rank 0)
    if rank == 0:
        print(f"\nModels initialized:")
        print(f"Encoder: {encoder.__class__.__name__}")
        print(f"Decoder: {decoder.__class__.__name__}")
        
        encoder_params = sum(p.numel() for p in encoder.parameters())
        encoder_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in decoder.parameters())
        decoder_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        
        print(f"\nParameter counts:")
        print(f"Encoder: {encoder_params:,} (trainable: {encoder_trainable:,})")
        print(f"Decoder: {decoder_params:,} (trainable: {decoder_trainable:,})")

    # Synchronize before creating trainer
    if world_size > 1:
        torch.distributed.barrier()

    # Build trainer - use the original SLatVaeGaussianTrainer
    try:
        trainer = SLatVaeGaussianTrainer(
            model_dict,
            dataset,
            output_dir=cfg.output_dir,
            load_dir=cfg.load_dir,
            step=cfg.load_ckpt,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            **cfg.trainer.args
        )
        print(f"[Worker {local_rank}] Trainer initialized successfully")
    except Exception as e:
        print(f"[Worker {local_rank}] Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Train
    if not cfg.get('tryrun', False):
        print(f"[Worker {local_rank}] Starting training...")
        trainer.run()
    else:
        print(f"[Worker {local_rank}] Dry run mode - skipping training")


def main():
    parser = argparse.ArgumentParser()
    
    # Config file
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, required=True, help='Path to preprocessed data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--val_dir', type=str, default=None, help='Path to validation data')
    
    # Pre-trained models
    parser.add_argument('--pretrained_encoder', type=str, required=True,
                        help='Path to pretrained encoder checkpoint')
    parser.add_argument('--pretrained_decoder', type=str, required=True,
                        help='Path to pretrained decoder checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                    help='Freeze encoder parameters (default: True for fine-tuning)')
    
    # Resume training
    parser.add_argument('--load_dir', type=str, default='', help='Directory to load checkpoint from')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint to load')
    
    # Multi-GPU settings
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs (-1 for all)')
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='12345')
    
    # Debug
    parser.add_argument('--tryrun', action='store_true', help='Dry run without training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Auto-detect GPUs
    if args.num_gpus == -1:
        args.num_gpus = torch.cuda.device_count()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Combine args and config
    cfg = edict()
    cfg.update(vars(args))
    cfg.update(config)
    cfg.load_dir = args.load_dir if args.load_dir else args.output_dir
    
    # Print configuration
    if cfg.node_rank == 0:
        print('\n' + '='*80)
        print('Configuration:')
        print('='*80)
        print(json.dumps(cfg, indent=2))
        print('='*80 + '\n')
    
    # Create output directory
    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save command
        with open(os.path.join(cfg.output_dir, 'command.txt'), 'w') as f:
            f.write(' '.join(['python'] + sys.argv))
    
    # Find checkpoint if resuming
    cfg = find_ckpt(cfg)
    
    # Check if launched by torchrun (preferred method)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Already launched by torchrun
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        print(f"Detected torchrun launch, using local_rank={local_rank}")
        main_worker(local_rank, cfg)
    else:
        # Launch with spawn (fallback)
        if cfg.num_gpus > 1:
            print(f"Launching {cfg.num_gpus} processes with mp.spawn")
            mp.spawn(main_worker, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main_worker(0, cfg)


if __name__ == '__main__':
    main()