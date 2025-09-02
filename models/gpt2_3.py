import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from datasets import load_dataset
from torch.amp import autocast, GradScaler
import sys
import argparse
import random
import numpy as np
from datetime import datetime

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.stdout.reconfigure(line_buffering=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def setup_ddp():
    # Get SLURM environment variables with defaults
    rank = int(os.environ.get('SLURM_PROCID', '0'))
    local_rank = int(os.environ.get('SLURM_LOCALID', '0'))
    world_size = int(os.environ.get('SLURM_NTASKS', '1'))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')

    # Explicitly set environment variables for PyTorch
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # Initialize the process group
    dist.init_process_group(backend='gloo', init_method='env://')
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank}/{world_size} initialized")
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def create_model_config():
    config = GPT2Config(
        vocab_size=50257,  
        n_positions=512,   
        n_embd=384,        
        n_layer=6,         
        n_head=6,          
        n_inner=1536,      
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50256,
    )
    return config

def get_openwebtext_dataloader(rank, world_size, data_dir, cache_dir, batch_size, block_size=512, validation_split=0.1):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset(data_dir, split='train', streaming=False, trust_remote_code=True, 
                         cache_dir=cache_dir)

    def tokenize_function(examples):
        tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=block_size)
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    from torch.utils.data import IterableDataset
    
    class OpenWebTextDataset(IterableDataset):
        def __init__(self, dataset, rank, world_size, is_validation=False, validation_split=0.1, max_samples=None):
            self.dataset = dataset
            self.rank = rank
            self.world_size = world_size
            self.is_validation = is_validation
            self.validation_split = validation_split
            self.max_samples = max_samples

        def __len__(self):
            # For IterableDataset, we need to provide an estimated length
            if self.max_samples:
                return self.max_samples
            else:
                return 10000  # Arbitrary large number for estimation
            
        def __iter__(self):
            sample_count = 0
            for i, sample in enumerate(self.dataset):
                if i % self.world_size == self.rank:

                    is_val_sample = (i % 100) < int(100 * self.validation_split)
                    if (self.is_validation and is_val_sample) or (not self.is_validation and not is_val_sample):
                        yield {
                            'input_ids': torch.tensor(sample['input_ids']),
                            'attention_mask': torch.tensor(sample['attention_mask']),
                            'labels': torch.tensor(sample['labels'])
                        }
                        sample_count += 1

                        if self.max_samples and sample_count >= self.max_samples:
                            break
    
    train_dataset = OpenWebTextDataset(tokenized_dataset, rank, world_size, is_validation=False, validation_split=validation_split)
    val_dataset = OpenWebTextDataset(tokenized_dataset, rank, world_size, is_validation=True, validation_split=validation_split, max_samples=100)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader

def get_device(local_rank):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    else:
        return torch.device("cpu")

def initialize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            if 'ln' in name or 'layernorm' in name:
                # Layer norm weights
                nn.init.ones_(param)
            elif 'bias' in name:
                # Bias terms
                nn.init.zeros_(param)
            else:
                # Linear layer weights
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

def calculate_perplexity(loss):
    return torch.exp(loss).item()

def validate_model(model, val_loader, device, tokenizer, world_size, max_batches=100):
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, 
                              attention_mask=attention_mask, 
                              labels=labels)
                loss = outputs.loss

            val_loss += loss.item()
            val_batches += 1
    
    if val_batches > 0:
        avg_val_loss = val_loss / val_batches
        val_loss_tensor = torch.tensor(avg_val_loss, dtype=torch.float32).to(device)

        dist.barrier()
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        
        val_loss = (val_loss_tensor / world_size).item()

        return val_loss, calculate_perplexity(torch.tensor(val_loss))

    else:
        return float('inf'), float('inf')
    
def summarize_model(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print(f"Model Size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # Assuming float32
    print("="*80 + "\n")

def train(args):
    rank, local_rank, world_size = setup_ddp()

    device = get_device(local_rank)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id 
    
    config = create_model_config()
    model = GPT2LMHeadModel(config)
    
    initialize_weights(model)
    
    model = model.to(device)

    if rank == 0:
        summarize_model(model)

    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * 1000)
    
    scaler = GradScaler()

    batch_size = 8 // world_size
    train_loader, val_loader = get_openwebtext_dataloader(rank, world_size, args.data_dir, args.cache_dir, batch_size, block_size=512)

    num_epochs = args.epochs
    
    # Track best validation loss for early stopping
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, 
                              attention_mask=attention_mask, 
                              labels=labels)
                loss = outputs.loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability in from-scratch training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            batch_count += 1
            
            # Log progress more frequently for from-scratch training
            if batch_count % 100 == 0 and rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

            # Reduced batch count for faster training. The number here is arbitrary and could be adjusted.
            if batch_count >= 2000:
                break

        avg_train_loss = epoch_loss / batch_count
        train_perplexity = calculate_perplexity(torch.tensor(avg_train_loss))
        
        # Validation
        val_loss, val_perplexity = validate_model(model, val_loader, device, tokenizer, world_size, max_batches=100)
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
            
            # Track best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  New best validation loss: {best_val_loss:.4f}")

        # Generate sample text every few epochs
        # if rank == 0 and (epoch + 1) % 5 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         with autocast(device_type='cuda', dtype=torch.float16):
        #             prompts = [
        #                 "The future of technology",
        #                 "In a world where",
        #                 "Scientists have recently",
        #                 "The company announced",
        #                 "According to the report"
        #             ]
                    
        #             import random
        #             prompt = random.choice(prompts)
                    
        #             inputs = tokenizer(prompt, return_tensors='pt') 
        #             input_ids = inputs.input_ids.to(device)
        #             attention_mask = inputs.attention_mask.to(device)

        #             generated_ids = model.module.generate(
        #                 input_ids,
        #                 attention_mask=attention_mask,
        #                 max_length=150,
        #                 temperature=1.0,
        #                 top_k=50,
        #                 top_p=0.95,
        #                 do_sample=True,
        #                 pad_token_id=tokenizer.eos_token_id,
        #                 eos_token_id=tokenizer.eos_token_id,
        #                 no_repeat_ngram_size=2
        #             )

        #         generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        #         print(f"\n=== Sample Generated Text (Epoch {epoch+1}, Prompt: '{prompt}') ===\n{generated_text}\n")

    # Final validation
    if rank == 0:
        final_val_loss, final_val_perplexity = validate_model(model, val_loader, device, tokenizer, max_batches=200)
        print(f"\nFinal Validation Results:")
        print(f"  Final Val Loss: {final_val_loss:.4f}")
        print(f"  Final Val Perplexity: {final_val_perplexity:.2f}")
        print(f"  Best Val Loss: {best_val_loss:.4f}")

    # Save the model after training is complete
    if rank == 0:
        print("Saving model...")
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the model, config, and tokenizer
        model.module.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        # Save training checkpoint with validation metrics
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': num_epochs,
            'config': config,
            'best_val_loss': best_val_loss,
            'final_val_loss': final_val_loss,
            'final_val_perplexity': final_val_perplexity,
        }, os.path.join(save_dir, 'training_checkpoint.pt'))
        
        print(f"From-scratch trained model saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for the dataset')
    parser.add_argument('--cache_dir', type=str, default='./data', help='Directory for the dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='./saved_model_from_scratch', help='Directory to save the trained model')
    args = parser.parse_args()

    print("Date: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps(vars(args), indent=2))

    try:
        train(args)
    finally:
        cleanup()

if __name__ == '__main__':
    main()