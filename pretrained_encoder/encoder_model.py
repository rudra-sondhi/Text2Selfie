import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class EncoderConfig:
    vocab_size: int
    d_model: int = 768
    nhead: int = 12
    num_encoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 1024

class MultiHeadAttention(nn.Module):
    """Standard Multi-head attention WITHOUT RoPE"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # No RoPE - just standard attention
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        
        batch_size, seq_len, _ = query.size()
        
        # Linear projections and reshape
        Q = self.q_proj(query).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # NO RoPE - Q and K stay as-is
        
        # Convert key_padding_mask to attention mask format if provided
        if key_padding_mask is not None:
            # key_padding_mask: [batch_size, seq_len]
            # Need to convert to [batch_size, 1, 1, seq_len] for broadcasting
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            # Create a large negative value for masked positions
            key_padding_mask = key_padding_mask.to(dtype=Q.dtype) * -1e9
            
            if attn_mask is not None:
                attn_mask = attn_mask + key_padding_mask
            else:
                attn_mask = key_padding_mask
        
        # Use PyTorch's optimized scaled dot product attention
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with Pre-LayerNorm"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LayerNorm: norm -> attention -> residual
        src_norm = self.norm1(src)
        attn_out = self.self_attn(src_norm, src_norm, src_norm, 
                                 attn_mask=src_mask, 
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(attn_out)
        
        # Pre-LayerNorm: norm -> feedforward -> residual
        src_norm = self.norm2(src)
        ff_out = self.linear2(F.gelu(self.linear1(src_norm)))
        src = src + self.dropout(ff_out)
        
        return src

class TransformerEncoder(nn.Module):
    """Transformer Encoder with LEARNED positional embeddings"""
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # LEARNED positional embeddings (instead of RoPE)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config.d_model, config.nhead, 
                                  config.dim_feedforward, config.dropout)
            for _ in range(config.num_encoder_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using best practices for modern transformers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use normal initialization for GELU activations
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

    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_all_hidden_states: bool = False) -> torch.Tensor:
        """
        Forward pass through the encoder
        
        Args:
            input_ids: Token ids of shape [batch_size, seq_len]
            attention_mask: Mask of shape [batch_size, seq_len] where 1 = attend, 0 = ignore
            return_all_hidden_states: Whether to return all layer outputs
            
        Returns:
            If return_all_hidden_states=False: Final hidden states [batch_size, seq_len, d_model]
            If return_all_hidden_states=True: Tuple of (final_states, all_layer_states)
        """
        batch_size, seq_len = input_ids.shape
        
        # Convert attention mask to key_padding_mask format (True = masked/ignore)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        
        # Token embeddings with scaling
        token_embeds = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        
        # LEARNED positional embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine token and position embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Store all layer outputs if requested
        all_hidden_states = [] if return_all_hidden_states else None
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=key_padding_mask)
            if return_all_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        if return_all_hidden_states:
            all_hidden_states.append(hidden_states)
            return hidden_states, all_hidden_states
        
        return hidden_states
    
    def get_embeddings(self, input_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None,
                      pooling_strategy: str = 'mean') -> torch.Tensor:
        """
        Get sentence-level embeddings for contrastive learning
        
        Args:
            input_ids: Token ids of shape [batch_size, seq_len]
            attention_mask: Mask of shape [batch_size, seq_len]
            pooling_strategy: 'mean', 'max', 'cls', or 'last'
            
        Returns:
            Sentence embeddings of shape [batch_size, d_model]
        """
        hidden_states = self.forward(input_ids, attention_mask)
        
        if pooling_strategy == 'mean':
            # Mean pooling with attention mask
            if attention_mask is not None:
                # Mask out padded positions
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                # Sum and divide by actual sequence length
                sentence_embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                sentence_embeddings = hidden_states.mean(dim=1)
                
        elif pooling_strategy == 'max':
            # Max pooling
            if attention_mask is not None:
                # Set padded positions to very negative values
                masked_hidden = hidden_states.clone()
                masked_hidden[attention_mask == 0] = -1e9
                sentence_embeddings = masked_hidden.max(dim=1)[0]
            else:
                sentence_embeddings = hidden_states.max(dim=1)[0]
                
        elif pooling_strategy == 'cls':
            # Use first token (assumes CLS token at position 0)
            sentence_embeddings = hidden_states[:, 0, :]
            
        elif pooling_strategy == 'last':
            # Use last non-padded token
            if attention_mask is not None:
                # Find last non-padded position for each sequence
                seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                sentence_embeddings = hidden_states[batch_indices, seq_lengths]
            else:
                sentence_embeddings = hidden_states[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
            
        return sentence_embeddings

if __name__ == "__main__":
    # Test the encoder
    config = EncoderConfig(
        vocab_size=50265,  # DistilGPT2 vocab size
        d_model=768,
        nhead=12,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=256
    )
    
    encoder = TransformerEncoder(config)
    
    # Test input
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Make some padding
    attention_mask[:, -20:] = 0  # Last 20 tokens are padding
    
    print("Testing TransformerEncoder:")
    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Forward pass
    hidden_states = encoder(input_ids, attention_mask)
    print(f"Output hidden states shape: {hidden_states.shape}")
    
    # Get sentence embeddings
    embeddings = encoder.get_embeddings(input_ids, attention_mask, pooling_strategy='mean')
    print(f"Sentence embeddings shape: {embeddings.shape}")
    
    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")