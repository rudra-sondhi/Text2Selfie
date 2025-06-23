# -*- coding: utf-8 -*-
"""
Refactored version using custom transformer model from model.py
Updated with gradient accumulation and removed gradient norm code
Added option to resume training from checkpoint
"""

import os
import json
import random
import torch
import shutil
import argparse

import yaml
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model import Text2SelfiesTransformer, TransformerConfig
import numpy as np
from tqdm import tqdm

# Set environment variable for better CUDA error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #ARNAV CHANGE THIS

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading tokenizers...")
gpt_tok = AutoTokenizer.from_pretrained("distilgpt2")
gpt_tok.add_special_tokens({'pad_token': '[PAD]'})
mol_tok = AutoTokenizer.from_pretrained("ibm/materials.selfies-ted")

# Define special tokens
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SEP_TOKEN_ID = 3  # Separator between text and molecule


def dict2ns(d):
    """Recursively convert dicts to simple namespaces for dot-access."""
    from argparse import Namespace
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict2ns(v)
    return Namespace(**d)

def validate_vocabulary_setup():
    """Validate that vocabulary mappings are correct"""
    print("Validating vocabulary setup...")
    
    # Check text vocab bounds
    max_text_token = gpt_tok.vocab_size - 1
    mapped_max_text = max_text_token + 4
    
    if mapped_max_text >= mol_offset:
        raise ValueError(f"Text vocab overflow! Max mapped text token {mapped_max_text} >= mol_offset {mol_offset}")
    
    # Check mol vocab bounds  
    max_mol_token = mol_tok.vocab_size - 1
    mapped_max_mol = max_mol_token + mol_offset
    
    if mapped_max_mol >= total_vocab_size:
        raise ValueError(f"Mol vocab overflow! Max mapped mol token {mapped_max_mol} >= total_vocab_size {total_vocab_size}")
    
    print(f"✓ Text tokens: [4, {mapped_max_text}]")
    print(f"✓ Mol tokens: [{mol_offset}, {mapped_max_mol}]")
    print(f"✓ Total vocab size: {total_vocab_size}")
    print("Vocabulary validation passed!")

# Calculate vocab sizes
text_vocab_size = gpt_tok.vocab_size
mol_vocab_size = mol_tok.vocab_size
mol_offset = text_vocab_size + 4  # Account for special tokens
total_vocab_size = text_vocab_size + mol_vocab_size + 4  # 4 special tokens

print(f"Text vocab size: {text_vocab_size}")
print(f"Mol vocab size: {mol_vocab_size}")
print(f"Total vocab size: {total_vocab_size}")

validate_vocabulary_setup()

def load_data_safely(data_path, max_examples=10000):
    """Load data with comprehensive error handling"""
    try:
        print(f"Loading data from: {data_path}")
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        
        # Handle different data formats
        if isinstance(raw_data, dict) and 'molecules' in raw_data:
            example_data = list(raw_data["molecules"].values())
        elif isinstance(raw_data, list):
            example_data = raw_data
        else:
            raise ValueError("Unexpected data format")
        
        print(f"Loaded {len(example_data)} examples")
        
        # Validate data quality
        valid_examples = []
        required_fields = ['selfies', 'c_nmr_text', 'h_nmr_text', 'functional_groups']
        
        for i, ex in enumerate(example_data[:max_examples]):
            if all(field in ex and ex[field] for field in required_fields):
                # Additional validation
                if len(ex['selfies']) > 0 and len(ex['c_nmr_text']) > 0:
                    valid_examples.append(ex)
            else:
                if i < 10:  # Only print first 10 warnings
                    print(f"Skipping invalid example {i}: missing fields")
        
        print(f"Valid examples after filtering: {len(valid_examples)}")
        
        if len(valid_examples) < 100:
            print("Warning: Very few valid examples found!")
        
        return valid_examples
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating minimal dummy data for testing...")
        return [
            {
                "selfies": "[C][C][Branch1][C][C]",
                "c_nmr_text": "13C NMR: δ 168.15, 155.89, 141.69",
                "h_nmr_text": "1H NMR: δ 8.15(1H, d, J = 1.63 Hz), 7.55(1H, d, J = 1.63 Hz)",
                "functional_groups": "alkene, aromatics"
            }
        ] * 100


def encode_sequence(text_part, mol_part, gpt_tok, mol_tok, mol_offset, max_length=256):
    """
    Encode a sequence with text and molecule parts.
    MODIFIED: Format text to match pretrained encoder expectations
    """
    # NEW: Format text to match pretrained encoder training format
    if text_part and isinstance(text_part, dict):
        # If text_part is a dict with NMR components
        c_nmr = text_part.get('c_nmr_text', '')
        h_nmr = text_part.get('h_nmr_text', '')
        # Format exactly like the pretrained encoder expects
        formatted_text = f"C NMR: {c_nmr} H NMR: {h_nmr}"
    elif text_part:
        # If it's already a string, assume it's properly formatted
        formatted_text = text_part
    else:
        formatted_text = ""
    
    # Encode text part with bounds checking
    text_tokens = []
    if formatted_text:
        text_encoded = gpt_tok.encode(formatted_text, add_special_tokens=False)
        # Safer mapping with bounds checking
        for t in text_encoded:
            if t < text_vocab_size:  # Only process valid tokens
                mapped_token = t + 4
                # Double-check the mapped token is within bounds
                if mapped_token < mol_offset:
                    text_tokens.append(mapped_token)
                else:
                    print(f"Warning: Text token {t} maps to {mapped_token} which exceeds text space")
    
    # Encode molecule part with bounds checking
    mol_tokens = []
    if mol_part:
        mol_encoded = mol_tok.encode(mol_part, add_special_tokens=False)
        # Safer mapping with bounds checking
        mol_tokens = []
        for m in mol_encoded:
            if m < mol_vocab_size:  # Only process valid tokens
                mapped_token = m + mol_offset
                # Double-check the mapped token is within total vocab
                if mapped_token < total_vocab_size:
                    mol_tokens.append(mapped_token)
                else:
                    # Fallback to a safe token or skip
                    print(f"Warning: Mol token {m} maps to {mapped_token} which exceeds vocab size")
    
    # Create source sequence: [BOS] text [SEP]
    src_tokens = [BOS_TOKEN_ID] + text_tokens + [SEP_TOKEN_ID]
    
    # Create target sequence: [BOS] molecule [EOS]
    tgt_tokens = [BOS_TOKEN_ID] + mol_tokens + [EOS_TOKEN_ID]
    
    # Truncate if needed
    src_tokens = src_tokens[:max_length]
    tgt_tokens = tgt_tokens[:max_length]
    
    return src_tokens, tgt_tokens

class MolTextDataset(Dataset):
    def __init__(self, data, gpt_tok, mol_tok, max_length=256):
        self.data = data
        self.gpt_tok = gpt_tok
        self.mol_tok = mol_tok
        self.mol_offset = text_vocab_size + 4
        self.max_length = max_length
        # Store the pad token id from the encoder's training
        self.encoder_pad_token_id = 50257  # This is what your encoder was trained with
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        
        # Check if required fields exist
        required_fields = ['selfies', 'c_nmr_text', 'h_nmr_text', 'functional_groups']
        if not all(field in ex for field in required_fields):
            # Return padding if data is missing
            return {
                "src_ids": torch.tensor([self.encoder_pad_token_id] * self.max_length, dtype=torch.long),
                "tgt_ids": torch.tensor([PAD_TOKEN_ID] * self.max_length, dtype=torch.long),
                "src_padding_mask": torch.ones(self.max_length, dtype=torch.bool),
                "tgt_padding_mask": torch.ones(self.max_length, dtype=torch.bool),
            }
        
        # Format molecule SELFIES
        slfs = ex['selfies'].replace("][", "] [")
        
        # Format NMR text to match encoder training format
        formatted_text = f"C NMR: {ex['c_nmr_text']} H NMR: {ex['h_nmr_text']}"
        
        # CRITICAL FIX: Encode text for encoder WITHOUT shifting
        # Use the tokenizer exactly as it was during encoder training
        src_ids = self.gpt_tok.encode(formatted_text, add_special_tokens=True, 
                                      max_length=self.max_length, 
                                      truncation=True)
        
        # Manually pad with the encoder's expected pad token id
        src_len = len(src_ids)
        if src_len < self.max_length:
            src_ids = src_ids + [self.encoder_pad_token_id] * (self.max_length - src_len)
        
        # Encode molecule for decoder WITH shifting to unified vocabulary
        mol_encoded = self.mol_tok.encode(slfs, add_special_tokens=False)
        mol_tokens = [m + self.mol_offset for m in mol_encoded if m < mol_vocab_size]
        
        # Create target sequence: [BOS] molecule [EOS]
        tgt_ids = [BOS_TOKEN_ID] + mol_tokens + [EOS_TOKEN_ID]
        tgt_ids = tgt_ids[:self.max_length]
        
        # Pad target sequences
        tgt_len = len(tgt_ids)
        tgt_ids = tgt_ids + [PAD_TOKEN_ID] * (self.max_length - tgt_len)
        
        # Create padding masks (True = padded position)
        src_padding_mask = [s == self.encoder_pad_token_id for s in src_ids]
        tgt_padding_mask = [False] * tgt_len + [True] * (self.max_length - tgt_len)
        
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_padding_mask": torch.tensor(src_padding_mask, dtype=torch.bool),
            "tgt_padding_mask": torch.tensor(tgt_padding_mask, dtype=torch.bool),
        }

def compute_accuracy(predictions, targets, ignore_index=0):
    """
    Compute token-level accuracy, ignoring padding tokens.
    """
    mask = targets != ignore_index
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()

def compute_seq_token_accuracy(predictions, targets, ignore_index=0):
    """
    For each sequence in the batch, compute:
      (# correct tokens) / (# non-pad tokens),
    then return the mean over all sequences.
    """
    # predictions, targets: [batch_size, seq_len]
    mask = targets != ignore_index                        # [B, T] boolean
    correct = (predictions == targets) & mask              # [B, T] boolean

    # Sum correct & total for each sequence
    correct_per_seq = correct.sum(dim=1).float()           # [B]
    total_per_seq   = mask.sum(dim=1).float()              # [B]

    # Avoid divide-by-zero (if any all-pad seqs exist)
    seq_acc = torch.where(total_per_seq>0,
                          correct_per_seq / total_per_seq,
                          torch.zeros_like(total_per_seq))
    
    return seq_acc.mean().item()                           # scalar


def compute_loss_safely(logits, targets, criterion, ignore_index=0):
    """Compute loss with additional safety checks"""
    # Flatten for loss computation
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    
    # Check for any invalid targets
    valid_mask = (targets_flat >= 0) & (targets_flat < logits.size(-1))
    if not valid_mask.all():
        invalid_count = (~valid_mask).sum().item()
        print(f"Warning: {invalid_count} invalid target tokens found, setting to ignore_index")
        targets_flat = torch.where(valid_mask, targets_flat, ignore_index)
    
    loss = criterion(logits_flat, targets_flat)
    return loss

def train_epoch(model, dataloader, optimizer, criterion, device, accumulation_steps=4):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_seq_token_acc = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    # Initialize accumulated loss
    accumulated_loss = 0
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move to device
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            src_mask = batch["src_padding_mask"].to(device)
            tgt_mask = batch["tgt_padding_mask"].to(device)
            
            # Validate input shapes
            #assert src_ids.size(1) <= 256, f"Source sequence too long: {src_ids.size(1)}"
            #assert tgt_ids.size(1) <= 256, f"Target sequence too long: {tgt_ids.size(1)}"
            
            # Teacher forcing: use ground truth as input (shifted right)
            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:]
            tgt_mask_input = tgt_mask[:, :-1]
            
            # Forward pass
            logits = model(src_ids, tgt_input, src_mask, tgt_mask_input)
            
            # Compute loss with safety checks
            loss = compute_loss_safely(logits, tgt_output, criterion, PAD_TOKEN_ID)
            
            # Scale loss by accumulation steps (important for proper averaging)
            loss = loss / accumulation_steps
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss detected in batch {batch_idx}, skipping...")
                continue
            
            # Compute accuracy (for display only)
            predictions = logits.argmax(dim=-1)
            accuracy = compute_accuracy(predictions, tgt_output, ignore_index=PAD_TOKEN_ID)
            seq_token_acc = compute_seq_token_accuracy(predictions, tgt_output, ignore_index=PAD_TOKEN_ID)
            
            # Backward pass (accumulate gradients)
            loss.backward()
            
            # Accumulate loss for display
            accumulated_loss += loss.item()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Perform optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Update metrics
                total_loss += accumulated_loss
                total_accuracy += accuracy
                total_seq_token_acc += seq_token_acc
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{accumulated_loss:.4f}',
                    'acc': f'{accuracy:.4f}',
                    'seq_token_acc': f'{seq_token_acc:.4f}',
                    'step': f'{num_batches}'
                })
                
                # Reset accumulated loss
                accumulated_loss = 0
            else:
                # Update progress bar with current accumulation
                progress_bar.set_postfix({
                    'acc_loss': f'{accumulated_loss:.4f}',
                    'acc': f'{accuracy:.4f}',
                    'seq_token_acc': f'{seq_token_acc:.4f}',
                    'acc_step': f'{(batch_idx + 1) % accumulation_steps}/{accumulation_steps}'
                })


        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {e}")
            print("Skipping this batch...")
            continue
    
    # Handle any remaining accumulated gradients
    if (len(dataloader) % accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()
        total_loss += accumulated_loss
        num_batches += 1
    
    if num_batches == 0:
        print("Warning: No valid batches processed!")
        return float('inf'), 0.0
    
    return total_loss / num_batches, total_accuracy / num_batches

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_seq_token_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                src_ids = batch["src_ids"].to(device)
                tgt_ids = batch["tgt_ids"].to(device)
                src_mask = batch["src_padding_mask"].to(device)
                tgt_mask = batch["tgt_padding_mask"].to(device)
                
                # Teacher forcing
                tgt_input = tgt_ids[:, :-1]
                tgt_output = tgt_ids[:, 1:]
                tgt_mask_input = tgt_mask[:, :-1]
                
                # Forward pass
                logits = model(src_ids, tgt_input, src_mask, tgt_mask_input)
                
                # Compute loss with safety checks
                loss = compute_loss_safely(logits, tgt_output, criterion, PAD_TOKEN_ID)
                
                # Compute accuracy
                predictions = logits.argmax(dim=-1)
                accuracy = compute_accuracy(predictions, tgt_output, ignore_index=PAD_TOKEN_ID)
                seq_token_acc = compute_seq_token_accuracy(predictions, tgt_output, ignore_index=PAD_TOKEN_ID)
                
                # Update metrics
                total_loss += loss.item()
                total_accuracy += accuracy
                total_seq_token_acc += seq_token_acc
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.4f}',
                    'seq_token_acc': f'{seq_token_acc:.4f}'
                })


                
            except RuntimeError as e:
                print(f"Error in evaluation batch: {e}")
                continue
    
    if num_batches == 0:
        return float('inf'), 0.0
    
    return total_loss / num_batches, total_accuracy / num_batches, total_seq_token_acc / num_batches

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load model checkpoint and return starting epoch, best accuracy, and scheduler info"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state")
    else:
        print("Warning: No optimizer state found in checkpoint")
    
    # Get starting epoch and best accuracy
    start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
    best_val_accuracy = checkpoint.get('val_accuracy', 0.0)
    
    # Check if scheduler state exists
    has_scheduler_state = 'scheduler_state_dict' in checkpoint
    scheduler_state = checkpoint.get('scheduler_state_dict', None) if has_scheduler_state else None
    
    print(f"Resuming from epoch {start_epoch}")
    print(f"Best validation accuracy so far: {best_val_accuracy:.4f}")
    
    return start_epoch, best_val_accuracy, has_scheduler_state, scheduler_state

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Text2SELFIES Transformer')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training from (e.g., ./checkpoints/best_model.pt)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = dict2ns(cfg_dict)

    # NEW: Add pretrained encoder path to config if not present
    if not hasattr(cfg.model, 'pretrained_encoder_path'):
        cfg.model.pretrained_encoder_path = None
    if not hasattr(cfg.model, 'freeze_encoder'):
        cfg.model.freeze_encoder = False
    if not hasattr(cfg.model, 'encoder_d_model'):
        cfg.model.encoder_d_model = 384  # Should match your pretrained encoder

    # Create save directory (only if not resuming)
    if args.resume is None:
        os.makedirs(cfg.save_path, exist_ok=False)
        shutil.copy(args.config, os.path.join(cfg.save_path, "config.yaml"))
    else:
        # Ensure save path exists when resuming
        os.makedirs(cfg.save_path, exist_ok=True)

    # Load data
    print("Loading data...")
    train_data = load_data_safely(cfg.dataset.train.path, max_examples=cfg.dataset.train.max)
    val_data = load_data_safely(cfg.dataset.val.path, max_examples=cfg.dataset.val.max)

    train_dataset = MolTextDataset(train_data, gpt_tok, mol_tok, max_length=cfg.model.max_length)
    val_dataset = MolTextDataset(val_data, gpt_tok, mol_tok, max_length=cfg.model.max_length)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Gradient accumulation settings
    EFFECTIVE_BATCH_SIZE = cfg.training.batch_size * cfg.training.accumulation_steps

    print(f"Base batch size: {cfg.training.batch_size}")
    print(f"Accumulation steps: {cfg.training.accumulation_steps}")
    print(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.hardware.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.hardware.num_workers)

    # Initialize model
    print("Initializing model...")
    config = TransformerConfig(
        src_vocab_size    = total_vocab_size,
        tgt_vocab_size    = total_vocab_size,
        d_model           = cfg.model.transformer.d_model,
        nhead             = cfg.model.transformer.nhead,
        num_encoder_layers= cfg.model.transformer.num_encoder_layers,
        num_decoder_layers= cfg.model.transformer.num_decoder_layers,
        dim_feedforward   = cfg.model.transformer.dim_feedforward,
        dropout           = cfg.model.transformer.dropout,
        max_seq_length    = cfg.model.transformer.max_seq_length,
        # NEW: Add pretrained encoder configuration
        pretrained_encoder_path = cfg.model.pretrained_encoder_path,
        freeze_encoder    = cfg.model.freeze_encoder,
        encoder_d_model   = cfg.model.encoder_d_model
    )

    model = Text2SelfiesTransformer(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # NEW: Print info about frozen vs trainable parameters
    if config.pretrained_encoder_path:
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Initialize optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.optimizer.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    print(f"Learning rate: {cfg.training.lr}")

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.scheduler.t_max, eta_min=cfg.training.scheduler.eta_min)

    # Initialize training variables
    start_epoch = 0
    best_val_accuracy = 0

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch, best_val_accuracy, has_scheduler_state, scheduler_state = load_checkpoint(args.resume, model, optimizer, device)
            
            # Handle scheduler state
            if has_scheduler_state and scheduler_state:
                scheduler.load_state_dict(scheduler_state)
                print("Loaded scheduler state")
            else:
                print("Warning: No scheduler state found in checkpoint - adjusting scheduler for resumed training")
                # Calculate remaining epochs and adjust T_max
                remaining_epochs = cfg.training.epochs - start_epoch
                if remaining_epochs > 0:
                    # Create a new scheduler for the remaining epochs
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, 
                        T_max=remaining_epochs, 
                        eta_min=cfg.training.scheduler.eta_min
                    )
                    print(f"Scheduler reinitialized with T_max={remaining_epochs} for remaining epochs")
        else:
            print(f"Error: Checkpoint file {args.resume} not found!")
            return

    # Training loop
    print("Starting training...")
    num_epochs = cfg.training.epochs

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train with gradient accumulation
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, cfg.training.accumulation_steps)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Evaluate
        val_loss, val_acc, seq_token_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Seq Tok Accuracy: {seq_token_acc:.4f}")
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr:.2e}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'accumulation_steps': cfg.training.accumulation_steps,
            }, os.path.join(cfg.save_path, 'best_model.pt')) 
            print(f"Saved best model with accuracy: {val_acc:.4f}")

        # Save checkpoint every few epochs
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'accumulation_steps': cfg.training.accumulation_steps,
            }, os.path.join(cfg.save_path, f'checkpoint_epoch_{epoch+1}.pt'))

    # Save final model
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'accumulation_steps': cfg.training.accumulation_steps,
    },  os.path.join(cfg.save_path,'final_model.pt'))

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Effective batch size used: {EFFECTIVE_BATCH_SIZE}")

if __name__ == "__main__":
    main()