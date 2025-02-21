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
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

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
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand_as(attention_score)
            attention_score = attention_score.masked_fill(mask == 0, -1e9) # Wherever mask == 0, we replace those positions with -1e9.
                                                                            # This is a very large negative number that makes those values effectively zero after applying softmax().
        
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

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiheadAttention(d_model, num_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    """src_mask is for masking source sequence. It ensures the model doesn't attend to padding tokens in the input sequence."""
    def forward(self, x, src_mask=None):
        # Multi-head attention layer with residual connection (original input x) and layer normalization
        attention_output = self.multi_head_attention(x, x, x, src_mask)
        attention_output = self.dropout(attention_output)
        x = self.norm1(x + attention_output)

        # Feed forward layer with residual connection (output of multi-head attention) and layer normalization
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.dropout(feed_forward_output)
        x = self.norm2(x + feed_forward_output)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        self.multi_head_attention = MultiheadAttention(d_model, num_heads)
        self.cross_attention = MultiheadAttention(d_model, num_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    """src_mask is for masking source sequence. It ensures the model doesn't attend to padding tokens in the input sequence.
        tgt_mask is for masking target sequence. It ensures the model doesn't peek ahead at the target sequence.
    """
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked multi-head attention layer with residual connection (original input x) and layer normalization
        attention_output = self.multi_head_attention(x, x, x, tgt_mask)
        attention_output = self.dropout(attention_output)
        x = self.norm1(x + attention_output)

        # Multi-head attention layer with residual connection (output of masked multi-head attention) and layer normalization
        cross_attention_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        cross_attention_output = self.dropout(cross_attention_output)
        x = self.norm2(x + cross_attention_output)

        # Feed forward layer with residual connection (output of multi-head attention) and layer normalization
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.dropout(feed_forward_output)
        x = self.norm3(x + feed_forward_output)

        return x
        
class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_len, dropout=0.1):
        super().__init__()

        # Embedding layers for input and target sequences
        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        # Positional encoding
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)

        # Encoder and decoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for i in range(num_layers)])
        
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for i in range(num_layers)])
        
        # Final linear layer with droupout
        self.final_layer = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_positional_encoding(self, max_seq_len, d_model):
        pos_encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)

    # Needed to prevent decorder from cheating -> looking ahead 
    def generate_square_subsequent_mask(self, size):
        # Create an upper-triangular matrix filled with ones and convert the mask to float, replacing 0s with -inf and 1s with 0
        mask = (torch.triu(torch.ones(size, size))== 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        if tgt is None:
            tgt = src
        
        if src_mask is None:
            src_mask = src.mask.unsqueeze(1).unsqueeze(2)
        if tgt_mask is None:
            tgt_mask = tgt.mask.unsqueeze(1).unsqueeze(2)

        src = self.encoder_embedding(src) * math.sqrt(self.encoder_embedding.embedding_dim)
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.decoder_embedding.embedding_dim)

        src = src + self.positional_encoding[:, :src.size(1), :].to(src.device)
        tgt = tgt + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        
        
        src = self.dropout(src)
        tgt = self.dropout(tgt)

        encoder_output = src
        for enc_layer in self.encoder_layers:
            encoder_output = enc_layer(encoder_output, src_mask)

        decoder_output = tgt
        for dec_layer in self.decoder_layers:
            decoder_output = dec_layer(decoder_output, encoder_output, src_mask, tgt_mask)

        output = self.final_layer(decoder_output)
        return output
    

    







        


        


    
        




