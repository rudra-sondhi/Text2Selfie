# -*- coding: utf-8 -*-
"""
Refactored version using custom transformer model from model.py
Updated with gradient accumulation and removed gradient norm code
"""

import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model import Text2SelfiesTransformer, TransformerConfig
import numpy as np
from tqdm import tqdm

# Set environment variable for better CUDA error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    Returns source (text) and target (molecule) token IDs.
    """
    # Encode text part with bounds checking
    text_tokens = []
    if text_part:
        text_encoded = gpt_tok.encode(text_part, add_special_tokens=False)
        # Safer mapping with bounds checking
        text_tokens = []
        for t in text_encoded:
            if t < text_vocab_size:  # Only process valid tokens
                mapped_token = t + 4
                # Double-check the mapped token is within bounds
                if mapped_token < mol_offset:
                    text_tokens.append(mapped_token)
                else:
                    # Fallback to a safe token or skip
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
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        
        # Check if required fields exist
        required_fields = ['selfies', 'c_nmr_text', 'h_nmr_text', 'functional_groups']
        if not all(field in ex for field in required_fields):
            # Return padding if data is missing
            return {
                "src_ids": torch.tensor([PAD_TOKEN_ID] * self.max_length, dtype=torch.long),
                "tgt_ids": torch.tensor([PAD_TOKEN_ID] * self.max_length, dtype=torch.long),
                "src_padding_mask": torch.ones(self.max_length, dtype=torch.bool),
                "tgt_padding_mask": torch.ones(self.max_length, dtype=torch.bool),
            }
        
        # Format molecule SELFIES
        slfs = ex['selfies'].replace("][", "] [")
        
        # Create text prompt
        text_prompt = f"Predict the molecule given the C NMR, H NMR and Functional Groups: {ex['c_nmr_text']} {ex['h_nmr_text']} {ex['functional_groups']}"
        
        # Encode sequences
        src_ids, tgt_ids = encode_sequence(text_prompt, slfs, self.gpt_tok, self.mol_tok, 
                                         self.mol_offset, self.max_length)
        
        # Pad sequences
        src_len = len(src_ids)
        tgt_len = len(tgt_ids)
        
        src_ids = src_ids + [PAD_TOKEN_ID] * (self.max_length - src_len)
        tgt_ids = tgt_ids + [PAD_TOKEN_ID] * (self.max_length - tgt_len)
        
        # Create padding masks (True = padded position)
        src_padding_mask = [False] * src_len + [True] * (self.max_length - src_len)
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
            assert src_ids.size(1) <= 256, f"Source sequence too long: {src_ids.size(1)}"
            assert tgt_ids.size(1) <= 256, f"Target sequence too long: {tgt_ids.size(1)}"
            
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
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{accumulated_loss:.4f}',
                    'acc': f'{accuracy:.4f}',
                    'step': f'{num_batches}'
                })
                
                # Reset accumulated loss
                accumulated_loss = 0
            else:
                # Update progress bar with current accumulation
                progress_bar.set_postfix({
                    'acc_loss': f'{accumulated_loss:.4f}',
                    'acc': f'{accuracy:.4f}',
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
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
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
                
                # Update metrics
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.4f}'
                })
                
            except RuntimeError as e:
                print(f"Error in evaluation batch: {e}")
                continue
    
    if num_batches == 0:
        return float('inf'), 0.0
    
    return total_loss / num_batches, total_accuracy / num_batches

# Load data
print("Loading data...")
data_path = "/home/rudra/Spectro/downloaded/data/final_data/random_molecules_700000.json"

example_data = load_data_safely(data_path, max_examples=600000)

# Create dataset with train/val split
print("Creating dataset...")
num_examples = min(len(example_data), 600000)
example_data = example_data[:num_examples]
random.shuffle(example_data)

train_size = int(0.9 * len(example_data))
train_data = example_data[:train_size]
val_data = example_data[train_size:]

train_dataset = MolTextDataset(train_data, gpt_tok, mol_tok, max_length=256)
val_dataset = MolTextDataset(val_data, gpt_tok, mol_tok, max_length=256)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# Gradient accumulation settings
ACCUMULATION_STEPS = 8  # Adjust based on your memory constraints
EFFECTIVE_BATCH_SIZE = 16 * ACCUMULATION_STEPS  # 64 effective batch size
BASE_BATCH_SIZE = 16

print(f"Base batch size: {BASE_BATCH_SIZE}")
print(f"Accumulation steps: {ACCUMULATION_STEPS}")
print(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BASE_BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BASE_BATCH_SIZE, shuffle=False, num_workers=4)

# Initialize model
print("Initializing model...")
config = TransformerConfig(
    src_vocab_size=total_vocab_size,
    tgt_vocab_size=total_vocab_size,
    d_model=512,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    max_seq_length=512
)

model = Text2SelfiesTransformer(config).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Initialize optimizer and loss
# Scale learning rate by accumulation steps for equivalent training
lr = 1e-4 
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

print(f"Base learning rate: 1e-4")
print(f"Scaled learning rate: {lr}")

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# Training loop
print("Starting training...")
num_epochs = 10
best_val_accuracy = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Train with gradient accumulation
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, ACCUMULATION_STEPS)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    
    # Evaluate
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    
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
            'config': config,
            'val_accuracy': val_acc,
            'val_loss': val_loss,
            'accumulation_steps': ACCUMULATION_STEPS,
        }, 'best_model.pt')
        print(f"Saved best model with accuracy: {val_acc:.4f}")

# Save final model
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'accumulation_steps': ACCUMULATION_STEPS,
}, 'final_model.pt')

print("\nTraining completed!")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Effective batch size used: {EFFECTIVE_BATCH_SIZE}")