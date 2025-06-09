import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

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

class LlamaRotaryEmbedding(nn.Module):
    """
    Simplified RoPE from HuggingFace Transformers
    Based on LLaMA implementation
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Applies Rotary Position Embedding to the query and key tensors.
    Fixed to handle cross-attention properly.
    """
    # q, k shape: [batch_size, num_heads, seq_len, head_dim]
    # cos, sin shape: [max_seq_len, head_dim]
    
    q_seq_len = q.size(2)
    k_seq_len = k.size(2)
    
    # For self-attention, use same positions for q and k
    # For cross-attention, use appropriate positions for each
    if q_seq_len == k_seq_len:
        # Self-attention case
        cos_emb = cos[:q_seq_len, :].unsqueeze(0).unsqueeze(0)
        sin_emb = sin[:q_seq_len, :].unsqueeze(0).unsqueeze(0)
        q_embed = (q * cos_emb) + (rotate_half(q) * sin_emb)
        k_embed = (k * cos_emb) + (rotate_half(k) * sin_emb)
    else:
        # Cross-attention case - use separate positions
        cos_q = cos[:q_seq_len, :].unsqueeze(0).unsqueeze(0)
        sin_q = sin[:q_seq_len, :].unsqueeze(0).unsqueeze(0)
        cos_k = cos[:k_seq_len, :].unsqueeze(0).unsqueeze(0)
        sin_k = sin[:k_seq_len, :].unsqueeze(0).unsqueeze(0)
        
        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    
    return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE using PyTorch's optimized SDPA"""
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
        
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        
        batch_size, seq_len, _ = query.size()
        kv_seq_len = key.size(1)
        
        # Linear projections and reshape
        Q = self.q_proj(query).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, kv_seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, kv_seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(V, seq_len=max(seq_len, kv_seq_len))
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        
        # Convert key_padding_mask to attention mask format if provided
        if key_padding_mask is not None:
            # key_padding_mask: [batch_size, kv_seq_len]
            # Need to convert to [batch_size, 1, 1, kv_seq_len] for broadcasting
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            # Create a large negative value for masked positions
            key_padding_mask = key_padding_mask.to(dtype=Q.dtype) * -1e9
            
            if attn_mask is not None:
                attn_mask = attn_mask + key_padding_mask
            else:
                attn_mask = key_padding_mask
        
        # Use PyTorch's optimized scaled dot product attention
        # This automatically handles Flash Attention when available
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
        ff_out = self.linear2(F.gelu(self.linear1(src_norm)))  # Using GELU instead of ReLU
        src = src + self.dropout(ff_out)
        
        return src

class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with Pre-LayerNorm"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
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
        self_attn_out = self.self_attn(tgt_norm, tgt_norm, tgt_norm,
                                      attn_mask=tgt_mask,
                                      key_padding_mask=tgt_key_padding_mask,
                                      is_causal=True)
        tgt = tgt + self.dropout(self_attn_out)
        
        # Cross-attention
        tgt_norm = self.norm2(tgt)
        cross_attn_out = self.cross_attn(tgt_norm, memory, memory,
                                        key_padding_mask=memory_key_padding_mask)
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

        self.src_tok_emb = nn.Embedding(config.src_vocab_size, config.d_model)
        self.tgt_tok_emb = nn.Embedding(config.tgt_vocab_size, config.d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config.d_model, config.nhead, 
                                  config.dim_feedforward, config.dropout)
            for _ in range(config.num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(config.d_model, config.nhead,
                                  config.dim_feedforward, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])
        
        # Final layer norm for encoder output
        self.encoder_norm = nn.LayerNorm(config.d_model)
        
        self.fc_out = nn.Linear(config.d_model, config.tgt_vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()

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
        # Embedding with scaling
        src_emb = self.src_tok_emb(src) * math.sqrt(self.config.d_model)
        src_emb = self.dropout(src_emb)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            src_emb = layer(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        return self.encoder_norm(src_emb)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_key_padding_mask: Optional[torch.Tensor] = None,
               memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode target sequence"""
        # Embedding with scaling
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.config.d_model)
        tgt_emb = self.dropout(tgt_emb)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, memory, 
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
        
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

if __name__ == "__main__":
    # Example usage
    config = TransformerConfig(
        src_vocab_size=32000,
        tgt_vocab_size=1024,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=512
    )
    
    model = Text2SelfiesTransformer(config)
    
    # Test forward pass
    src = torch.randint(0, config.src_vocab_size, (2, 100))  # batch of 2, source length 100
    tgt = torch.randint(0, config.tgt_vocab_size, (2, 30))   # batch of 2, target length 30
    
    # Create padding masks (optional)
    src_padding_mask = (src == 0)  # Assuming 0 is padding token
    tgt_padding_mask = (tgt == 0)
    
    logits = model(src, tgt, 
                  src_key_padding_mask=src_padding_mask,
                  tgt_key_padding_mask=tgt_padding_mask)
    
    print(f"Input shapes - src: {src.shape}, tgt: {tgt.shape}")
    print(f"Output logits shape: {logits.shape}")  # (2, 30, tgt_vocab_size)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test generation
    generated = model.generate(src, max_length=50, temperature=0.8)
    print(f"Generated shape: {generated.shape}")