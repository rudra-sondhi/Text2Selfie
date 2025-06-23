import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

# Import the pretrained encoder components
from encoder_model import TransformerEncoder, EncoderConfig  # Your encoder files

@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 768
    nhead: int = 12
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 1024
    # New parameters for pretrained encoder
    pretrained_encoder_path: Optional[str] = None
    freeze_encoder: bool = False
    encoder_d_model: int = 384  # Should match your pretrained encoder

class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with Pre-LayerNorm using PyTorch's built-in attention"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        # Use PyTorch's built-in MultiheadAttention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Self-attention with causal masking
        tgt_norm = self.norm1(tgt)
        # Create causal mask if not provided
        if tgt_mask is None:
            seq_len = tgt.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tgt.device, dtype=tgt_norm.dtype)
        
        # Convert key_padding_mask to float if needed
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(dtype=tgt_norm.dtype)
        
        self_attn_out, _ = self.self_attn(tgt_norm, tgt_norm, tgt_norm,
                                          attn_mask=tgt_mask,
                                          key_padding_mask=tgt_key_padding_mask,
                                          need_weights=False)
        tgt = tgt + self.dropout(self_attn_out)
        
        # Cross-attention
        tgt_norm = self.norm2(tgt)
        # Convert memory_key_padding_mask to float if needed
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.to(dtype=tgt_norm.dtype)
            
        cross_attn_out, _ = self.cross_attn(tgt_norm, memory, memory,
                                            key_padding_mask=memory_key_padding_mask,
                                            need_weights=False)
        tgt = tgt + self.dropout(cross_attn_out)
        
        # Feedforward
        tgt_norm = self.norm3(tgt)
        ff_out = self.linear2(F.gelu(self.linear1(tgt_norm)))  # Using GELU
        tgt = tgt + self.dropout(ff_out)
        
        return tgt

class Text2SelfiesTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # NEW: Use pretrained encoder if provided
        if config.pretrained_encoder_path:
            print(f"Loading pretrained encoder from: {config.pretrained_encoder_path}")
            self._load_pretrained_encoder(config.pretrained_encoder_path, config.freeze_encoder)
            
            # If encoder and decoder have different d_model, add projection layer
            if config.encoder_d_model != config.d_model:
                self.encoder_projection = nn.Linear(config.encoder_d_model, config.d_model)
                print(f"Added encoder projection: {config.encoder_d_model} -> {config.d_model}")
            else:
                self.encoder_projection = None
        else:
            # Original: Use token embeddings for encoder
            self.src_tok_emb = nn.Embedding(config.src_vocab_size, config.d_model)
            # Add learned positional embeddings for encoder when not using pretrained
            self.src_pos_emb = nn.Embedding(config.max_seq_length, config.d_model)
            self.encoder = None
            self.encoder_projection = None

        # Target embeddings (for decoder)
        self.tgt_tok_emb = nn.Embedding(config.tgt_vocab_size, config.d_model)
        
        # LEARNED positional embeddings for decoder
        self.tgt_pos_emb = nn.Embedding(config.max_seq_length, config.d_model)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(config.d_model, config.nhead,
                                  config.dim_feedforward, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])
        
        # Final layer norm for encoder output
        self.encoder_norm = nn.LayerNorm(config.d_model)
        
        # Decoder output layer norm
        self.decoder_norm = nn.LayerNorm(config.d_model)
        
        self.fc_out = nn.Linear(config.d_model, config.tgt_vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()

    def _load_pretrained_encoder(self, checkpoint_path: str, freeze: bool = False):
        """Load the pretrained NMR encoder"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get encoder config from checkpoint
        if 'config' in checkpoint:
            encoder_config = checkpoint['config']
        else:
            # Fallback: create config matching your pretrained model
            encoder_config = EncoderConfig(
                vocab_size=50265,  # DistilGPT2 vocab size
                d_model=384,
                nhead=4,
                num_encoder_layers=3,
                dim_feedforward=1024,
                dropout=0.1,
                max_seq_length=256
            )
        
        # Create encoder instance
        self.encoder = TransformerEncoder(encoder_config)
        
        # Load pretrained weights
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        
        if freeze:
            # Freeze encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder parameters frozen")
        
        print(f"Loaded pretrained encoder with {sum(p.numel() for p in self.encoder.parameters()):,} parameters")

    def _init_weights(self):
        """Initialize weights using best practices for modern transformers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use normal initialization for GELU activations (better than Xavier)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.normal_(module.weight, mean=0, std=0.02)
                    nn.init.constant_(module.bias, 0)
                else:
                    # For linear layers without bias (like projections)
                    nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.Embedding):
                # Standard embedding initialization
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def encode(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence"""
        if self.encoder is not None:
            # Use pretrained encoder
            # Convert padding mask format: False = attend, True = ignore
            attention_mask = None
            if src_key_padding_mask is not None:
                attention_mask = ~src_key_padding_mask  # Invert: True = attend, False = ignore
            
            # Get embeddings from pretrained encoder
            encoded = self.encoder(src, attention_mask=attention_mask)
            
            # Apply projection if needed
            if self.encoder_projection is not None:
                encoded = self.encoder_projection(encoded)
                
        else:
            # Use token embeddings with learned positional embeddings
            batch_size, seq_len = src.shape
            
            # Token embeddings with scaling
            token_embeds = self.src_tok_emb(src) * math.sqrt(self.config.d_model)
            
            # Position embeddings
            position_ids = torch.arange(seq_len, device=src.device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.src_pos_emb(position_ids)
            
            # Combine embeddings
            encoded = token_embeds + position_embeds
            encoded = self.dropout(encoded)
        
        return self.encoder_norm(encoded)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_key_padding_mask: Optional[torch.Tensor] = None,
               memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode target sequence"""
        batch_size, seq_len = tgt.shape
        
        # Token embeddings with scaling
        token_embeds = self.tgt_tok_emb(tgt) * math.sqrt(self.config.d_model)
        
        # Learned positional embeddings
        position_ids = torch.arange(seq_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.tgt_pos_emb(position_ids)
        
        # Combine token and position embeddings
        tgt_emb = token_embeds + position_embeds
        tgt_emb = self.dropout(tgt_emb)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, memory, 
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
        
        # Final layer norm
        tgt_emb = self.decoder_norm(tgt_emb)
        
        return tgt_emb

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Encode source
        memory = self.encode(src, src_key_padding_mask)
        
        # Decode target
        tgt_out = self.decode(tgt, memory, tgt_key_padding_mask, 
                            memory_key_padding_mask or src_key_padding_mask)
        
        # Final projection
        logits = self.fc_out(tgt_out)
        return logits

    def generate(self, src: torch.Tensor, max_length: int = 100, 
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95,
                 src_key_padding_mask: Optional[torch.Tensor] = None,
                 bos_token_id: int = 1, eos_token_id: int = 2, pad_token_id: int = 0):
        """
        Generate sequences using greedy decoding or sampling
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        memory = self.encode(src, src_key_padding_mask)
        
        # Initialize target sequence with BOS token
        tgt = torch.full((batch_size, 1), bos_token_id, device=device, dtype=torch.long)
        
        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Create target mask (no padding mask needed for generated sequence)
            tgt_mask = None
            
            # Decode
            tgt_out = self.decode(tgt, memory, tgt_mask, src_key_padding_mask)
            
            # Get logits for the last position
            logits = self.fc_out(tgt_out[:, -1, :])  # [batch_size, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Check if all sequences have generated EOS token
            if (next_token == eos_token_id).all():
                break
        
        return tgt