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

def train_model(transformer, train_loader, val_loader, optimizer, scheduler, device, epochs, tokenizer):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    best_val_loss = float('inf')
    early_stopping_patience = 3
    epochs_without_improvement = 0

    log("Starting training...")
    log(f"Training on: {device}")
    log(f"Number of training batches per epoch: {len(train_loader)}")
    log(f"Number of validation batches per epoch: {len(val_loader)}\n")

    for epoch in range(epochs):
        # Training phase
        transformer.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")
        for batch in train_progress:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            src_mask = batch['attention_mask'].to(device)
            tgt_mask = batch['target_attention_mask'].to(device)
            
            outputs = transformer(src=input_ids, tgt=target_ids, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            
            predictions = outputs.argmax(dim=-1)
            mask = target_ids != tokenizer.pad_token_id
            train_correct += (predictions[mask] == target_ids[mask]).sum().item()
            train_total += mask.sum().item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_avg_loss = train_loss / (train_progress.n + 1)
            train_accuracy = train_correct / train_total if train_total > 0 else 0
            train_progress.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{train_avg_loss:.4f}", acc=f"{train_accuracy:.4f}")

        train_avg_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0

        # Validation phase (every 5th epoch)
        if (epoch + 1) % 5 == 0:
            transformer.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="batch")
                for batch in val_progress:
                    input_ids = batch['input_ids'].to(device)
                    target_ids = batch['target_ids'].to(device)
                    src_mask = batch['attention_mask'].to(device)
                    tgt_mask = batch['target_attention_mask'].to(device)
                    
                    outputs = transformer(src=input_ids, tgt=target_ids, src_mask=src_mask, tgt_mask=tgt_mask)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                    
                    predictions = outputs.argmax(dim=-1)
                    mask = target_ids != tokenizer.pad_token_id
                    val_correct += (predictions[mask] == target_ids[mask]).sum().item()
                    val_total += mask.sum().item()
                    
                    val_loss += loss.item()
                    val_avg_loss = val_loss / (val_progress.n + 1)
                    val_accuracy = val_correct / val_total if val_total > 0 else 0
                    val_progress.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{val_avg_loss:.4f}", acc=f"{val_accuracy:.4f}")

            val_avg_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total if val_total > 0 else 0

            # Logging epoch stats (including validation)
            log(f"\nEpoch {epoch+1} Summary:")
            log(f"Train Loss: {train_avg_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
            log(f"Val Loss: {val_avg_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

            # Update scheduler based on validation loss
            scheduler.step(val_avg_loss)

            # Save best model based on validation loss
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                torch.save(transformer.state_dict(), "./save_model/model_02_best.pth")
                log("Best model saved (based on validation loss)!")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                log(f"No improvement in validation loss. Patience: {epochs_without_improvement}/{early_stopping_patience}")
                if epochs_without_improvement >= early_stopping_patience:
                    log("Early stopping triggered!")
                    break
        else:
            # Log training stats only when not validating
            log(f"\nEpoch {epoch+1} Summary:")
            log(f"Train Loss: {train_avg_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(transformer.state_dict(), f"./checkpoints/02_model_checkpoint_epoch_{epoch+1}.pth")
            log(f"Checkpoint saved at epoch {epoch+1}.\n")
            
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using {device} device")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    dataset = TextDataset('./processed_data/training_data_01.txt', tokenizer)
    log(f"Total Number of question-answer pairs: {len(dataset)}")

    batch_size = 16
    num_layers = 6
    num_heads = 8
    d_model = 512
    d_ff = 2048
    dropout = 0.3
    max_seq_len = 512
    input_vocab_size = tokenizer.vocab_size
    target_vocab_size = tokenizer.vocab_size
    epochs = 40

    # Split the dataset into 90% training and 10% validation
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)  # 90% train, 10% val
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)
    log(f"Training samples: {len(train_indices)} | Validation samples: {len(val_indices)}")
    
    # Initialize custom Transformer and load GPT-2 weights
    transformer = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_len, dropout).to(device)
    gpt2model = GPT2LMHeadModel.from_pretrained('gpt2')
    transformer.load_state_dict(gpt2model.state_dict(), strict=False)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    try:
        train_model(transformer, train_loader, val_loader, optimizer, scheduler, device, epochs, tokenizer)
        log("\nTraining completed successfully!")
    except Exception as e:
        log(f"\nTraining interrupted: {str(e)}")
    finally:
        torch.save(transformer.state_dict(), "./save_model/02_final_model_weights.pth")
        log("Final model saved!")

if __name__ == "__main__":
    main()