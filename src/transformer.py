import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
import numpy as np

from transformers import GPT2Tokenizer

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # Check if d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear layers for Q, K, V, and output
        self.q_linear = nn.Linear(d_model, d_model, bias=True)
        self.k_linear = nn.Linear(d_model, d_model, bias=True)
        self.v_linear = nn.Linear(d_model, d_model, bias=True)
        self.output_linear = nn.Linear(d_model, d_model, bias=True)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        attention_score = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        
        """if mask.dim() == 2: 
                mask = mask.unsqueeze(1).unsqueeze(2)
            The mask might initially be 2D with shape (batch_size, seq_len).
            We reshape it to 4D so it matches the shape of attention_score → (batch_size, 1, 1, seq_len).
            Why?
                Attention scores have shape (batch_size, num_heads, seq_len, seq_len).
            Example : 
            If mask is (batch_size, seq_len), let's say:  
            mask = [[1, 1, 1, 0, 0]]

            Applying .unsqueeze(1).unsqueeze(2) changes its shape:
            mask = [[[[1, 1, 1, 0, 0]]]]
            Now, it can properly expand to (batch_size, num_heads, seq_len, seq_len).
        """
        
        if mask is not None:
            attention_score = attention_score + mask  # Mask is broadcasted       
        attention_probs = torch.nn.functional.softmax(attention_score, dim=-1)
        output = torch.matmul(attention_probs, value)
        return output
    """
    Example -> 
            Before masking:
            attention_score = [[
                [2.3, 1.5, 3.2, 4.1, 2.8],
                [1.7, 2.4, 2.9, 3.3, 1.2]
            ]]
            mask = [[[[1, 1, 1, 0, 0]]]]

            After masking:
            attention_score = [[
                [ 2.3,  1.5,  3.2, -1e9, -1e9],
                [ 1.7,  2.4,  2.9, -1e9, -1e9]
            ]]

            Now, after softmax(), the last two positions will have almost zero probability.
    """

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear transformation for Q, K, V. We pass them through three different linear layers (q_linear, k_linear, v_linear).
        # The reason we transform them is that different projections allow the model to focus on different parts of the input.
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split Q, K, V into num_heads for Multihead Attention
        """We reshape each tensor from: (batch_size, seq_len, d_model)
            to: (batch_size, seq_len, num_heads, head_dim)
            
            Here, head_dim = d_model / num_heads (since each head processes a smaller part of the full embedding).

            Also .transpose(1, 2) swaps the seq_len and num_heads dimensions. 
            The reason? Each attention head operates on its own slice of the data, so we need the num_heads dimension to be second.
        """

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.output_linear(attention_output)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiheadAttention(d_model, num_heads)  # Renamed for clarity
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=True)
        )
        # Pre-layer normalization (GPT-2 style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    """src_mask is for masking source sequence. It ensures the model doesn't attend to padding tokens in the input sequence.
        tgt_mask is for masking target sequence. It ensures the model doesn't peek ahead at the target sequence.
    """

    def forward(self, x, mask=None):
        # Pre-LN: Normalize before attention
        x_norm = self.norm1(x)
        attn_output = self.self_attention(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn_output)  # Residual connection
        # Pre-LN: Normalize before FFN
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)  # Residual connection
        return x
    

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)
        # Use learnable positional embeddings instead of fixed ones
        self.positional_encoding = nn.Embedding(max_seq_len, d_model)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(d_model, vocab_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    # def create_positional_encoding(self, max_seq_len, d_model):
    #     pos_encoding = torch.zeros(max_seq_len, d_model)
    #     position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    #     pos_encoding[:, 0::2] = torch.sin(position * div_term)
    #     pos_encoding[:, 1::2] = torch.cos(position * div_term)
    #     return pos_encoding.unsqueeze(0)

    def generate_causal_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.view(1, 1, size, size)

    def forward(self, src, mask=None):
        batch_size, seq_len = src.shape
        if mask is None:
            mask = self.generate_causal_mask(seq_len).to(src.device)
        # Generate position indices
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(src.device)
        x = self.embedding(src) * math.sqrt(self.d_model)

        # Add learnable positional embeddings
        x = x + self.positional_encoding(positions)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, mask)
        return self.final_layer(x)

    def generate(self, src, max_length=100, temperature=0.7, top_k=50, tokenizer=None):
        if tokenizer is None:
            raise ValueError("Tokenizer required")
        self.eval()
        generated = src.clone()
        with torch.no_grad():
            for _ in range(max_length - src.size(1)):
                mask = self.generate_causal_mask(generated.size(1)).to(src.device)
                outputs = self(generated, mask)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply top-k filtering
                top_k_logits, _ = torch.topk(next_token_logits, top_k, dim=-1)
                min_top_k = top_k_logits[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.full_like(next_token_logits, float('-inf')),
                    next_token_logits
                )
                
                # Sample from the filtered distribution
                next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        return generated