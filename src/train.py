import torch
from transformer import GPT2LMHeadModel
from transformer import GPT2Tokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

from dataset import TextDataset
from transformer import Transformer
from logger import log

def train_model(transformer, dataloader, optimizer, scheduler, device, epochs, tokenizer):
    transformer.train()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    best_loss = float('inf')

    log("Starting training...")
    log(f"Training on: {device}")
    log(f"Number of batches per epoch: {len(dataloader)}\n")

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        # Initialize progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            src_mask = batch['attention_mask'].to(device)
            tgt_mask = batch['target_attention_mask'].to(device)
            
            # Forward pass
            outputs = transformer(src=input_ids, tgt=target_ids, src_mask=src_mask, tgt_mask=tgt_mask)
            
            # Compute loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            
            # Compute accuracy
            predictions = outputs.argmax(dim=-1)  
            mask = target_ids != tokenizer.pad_token_id  
            correct_predictions += (predictions[mask] == target_ids[mask]).sum().item()
            total_predictions += mask.sum().item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}", acc=f"{accuracy:.4f}")

        # Epoch statistics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        log(f"\nEpoch {epoch+1} Summary:")
        log(f"Average Loss: {avg_loss:.4f}")
        log(f"Accuracy: {accuracy:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(transformer.state_dict(), "./save_model/transformer_best.pth")
            log("Best model saved!")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(transformer.state_dict(), f"./checkpoints/model_checkpoint_epoch_{epoch+1}.pth")
            log(f"Checkpoint saved at epoch {epoch+1}.\n")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} device")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TextDataset('./processed_data/training_data_01.txt', tokenizer)
    log(f"Number of question-answer pairs: {len(train_dataset)}")

    batch_size = 16
    num_layers = 6
    num_heads = 8
    d_model = 512
    d_ff = 2048
    dropout = 0.1
    max_seq_len = 512
    input_vocab_size = tokenizer.vocab_size
    target_vocab_size = tokenizer.vocab_size

    learning_rate = 3e-4
    epochs = 200

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    transformer = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_len, dropout).to(device)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader) * epochs)

    try:
        train_model(transformer, dataloader, optimizer, scheduler, device, epochs, tokenizer)
        log("\nTraining completed successfully!")

    except Exception as e:
        log(f"\nTraining interrupted: {str(e)}")

    finally:
        torch.save(transformer.state_dict(), "./save_model/final_model_weights.pth")
        log("Final model saved!")

if __name__ == "__main__":
    main()
