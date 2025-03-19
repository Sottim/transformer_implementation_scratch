import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from dataset import TextDataset
from transformer import Transformer
from logger import log
import os
import random
import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(transformer, train_loader, val_loader, optimizer, device, epochs, tokenizer):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    log("Starting training...")
    log(f"Training on: {device}")
    log(f"Number of training batches per epoch: {len(train_loader)}")
    log(f"Number of validation batches per epoch: {len(val_loader)}\n")

    for epoch in range(epochs):
        transformer.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = transformer(input_ids, mask=None)
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            predictions = shift_logits.argmax(dim=-1)
            mask = shift_labels != tokenizer.pad_token_id
            train_correct += (predictions[mask] == shift_labels[mask]).sum().item()
            train_total += mask.sum().item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
        log(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%")

        if (epoch + 1) % 5 == 0:
            transformer.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = transformer(input_ids, mask=None)
                    shift_logits = outputs[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                    predictions = shift_logits.argmax(dim=-1)
                    mask = shift_labels != tokenizer.pad_token_id
                    val_correct += (predictions[mask] == shift_labels[mask]).sum().item()
                    val_total += mask.sum().item()
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
            log(f"Epoch {epoch+1}/{epochs} - Validation Loss: {avg_val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(transformer.state_dict(), "./save_model/best_model.pth")
                log("Best model saved based on validation loss.")
                patience_counter = 0
            else:
                patience_counter += 1
                log(f"No improvement. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    log("Early stopping triggered.")
                    break
            
            log("==============================================================")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(transformer.state_dict(), checkpoint_path)
            log(f"Checkpoint saved at {checkpoint_path}")

    log("Training complete.")
    torch.save(transformer.state_dict(), "./save_model/decoder_only_model.pth")

def main():
    set_random_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = TextDataset('./processed_data/training_data_01.txt', tokenizer)
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)
    train_loader = DataLoader(dataset, batch_size=16, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=16, sampler=SubsetRandomSampler(val_indices))
    
    transformer = Transformer(
        num_layers=12,
        d_model=768,
        num_heads=12,
        d_ff=3072,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=512,
        dropout=0.1
    ).to(device)
    
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_state_dict = gpt2.state_dict()
    
    # Map GPT-2 weights to your custom Transformer architecture
    new_state_dict = {}
    for key, value in gpt2_state_dict.items():
        if key.startswith('transformer.h.'):
            layer_num = key.split('.')[2]
            base_key = f"decoder_layers.{layer_num}"
            if 'attn.c_attn.weight' in key:
                qkv_weight = value  # Shape: [768, 2304]
                q_weight, k_weight, v_weight = qkv_weight.split(768, dim=1)
                new_state_dict[f"{base_key}.self_attention.q_linear.weight"] = q_weight
                new_state_dict[f"{base_key}.self_attention.k_linear.weight"] = k_weight
                new_state_dict[f"{base_key}.self_attention.v_linear.weight"] = v_weight
            elif 'attn.c_attn.bias' in key:
                qkv_bias = value  # Shape: [2304]
                q_bias, k_bias, v_bias = qkv_bias.split(768)
                new_state_dict[f"{base_key}.self_attention.q_linear.bias"] = q_bias
                new_state_dict[f"{base_key}.self_attention.k_linear.bias"] = k_bias
                new_state_dict[f"{base_key}.self_attention.v_linear.bias"] = v_bias
            elif 'attn.c_proj.weight' in key:
                new_state_dict[f"{base_key}.self_attention.output_linear.weight"] = value
            elif 'attn.c_proj.bias' in key:
                new_state_dict[f"{base_key}.self_attention.output_linear.bias"] = value
            elif 'mlp.c_fc.weight' in key:
                new_state_dict[f"{base_key}.feed_forward.0.weight"] = value.t()  # Transpose to [3072, 768]
            elif 'mlp.c_fc.bias' in key:
                new_state_dict[f"{base_key}.feed_forward.0.bias"] = value
            elif 'mlp.c_proj.weight' in key:
                new_state_dict[f"{base_key}.feed_forward.2.weight"] = value.t()  # Transpose to [768, 3072]
            elif 'mlp.c_proj.bias' in key:
                new_state_dict[f"{base_key}.feed_forward.2.bias"] = value
            elif 'ln_1.weight' in key:
                new_state_dict[f"{base_key}.norm1.weight"] = value
            elif 'ln_1.bias' in key:
                new_state_dict[f"{base_key}.norm1.bias"] = value
            elif 'ln_2.weight' in key:
                new_state_dict[f"{base_key}.norm2.weight"] = value
            elif 'ln_2.bias' in key:
                new_state_dict[f"{base_key}.norm2.bias"] = value
        elif key == 'transformer.wte.weight':
            new_state_dict['embedding.weight'] = value
        elif key == 'transformer.wpe.weight':
            new_state_dict['positional_encoding'] = value
        elif key == 'lm_head.weight':
            new_state_dict['final_layer.weight'] = value

    # Load the mapped weights into the transformer
    transformer.load_state_dict(new_state_dict, strict=False)
    
    optimizer = torch.optim.Adam(transformer.parameters(), lr=3e-4)
    train_model(transformer, train_loader, val_loader, optimizer, device, epochs=40, tokenizer=tokenizer)

if __name__ == "__main__":
    main()