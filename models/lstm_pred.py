import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import json
import datetime
import time
import random
import numpy as np
import sys
from collections import Counter
import glob

# ------------------------- Model Changes for Language Modeling -------------------------

class LSTMLanguageModel(nn.Module):
    """
    Key differences from classification:
    1. Output is vocab_size (predict any word) instead of num_classes (2-4)
    2. We output predictions at EVERY timestep, not just the last one
    3. Can optionally tie embedding and output weights (common trick)
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, 
                 dropout=0.5, tie_weights=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # LSTM processes sequences
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        
        # Output projection - predicts next word from hidden state
        if tie_weights and embed_dim == hidden_dim:
            # Share weights between embedding and output (reduces parameters)
            self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
            self.decoder.weight = self.embedding.weight
        else:
            self.decoder = nn.Linear(hidden_dim, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        if hasattr(self.decoder, 'bias') and self.decoder.bias is not None:
            self.decoder.bias.data.zero_()
    
    def forward(self, input, hidden=None):
        """
        Key difference: Returns predictions for ALL positions, not just last
        Input shape: (batch, seq_len)
        Output shape: (batch, seq_len, vocab_size)
        """
        emb = self.dropout(self.embedding(input))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)
        # Decode at every timestep (not just the last one like classification)
        decoded = self.decoder(output)
        return decoded, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state for stateful training (optional)"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

# ------------------------- Dataset for Local Files -------------------------

class LocalTextDataset(Dataset):
    """
    Load text data from local files for language modeling.
    Expects text files in a directory, processes them into token sequences.
    """
    def __init__(self, data_path, vocab=None, seq_len=256, stride=128, 
                 max_vocab=50000, split='train'):
        """
        Args:
            data_path: Path to directory containing .txt files OR single .txt file
            vocab: Pre-built vocabulary (build one if None)
            seq_len: Length of sequences to return
            stride: How much to shift between sequences (stride < seq_len = overlap)
            max_vocab: Maximum vocabulary size
            split: 'train', 'val', or 'test' (for splitting data)
        """
        self.seq_len = seq_len
        self.stride = stride
        self.split = split
        
        # Load text from files
        self.raw_text = self._load_text_files(data_path)
        
        # Build or use provided vocabulary
        if vocab is None:
            self.vocab, self.inverse_vocab = self._build_vocab(self.raw_text, max_vocab)
        else:
            self.vocab = vocab
            self.inverse_vocab = {v: k for k, v in vocab.items()}
        
        # Tokenize all text
        self.tokens = self._tokenize_text(self.raw_text)
        
        # Split data (80/10/10 by default)
        self._split_data()
        
        print(f"[{split}] Loaded {len(self.tokens):,} tokens, "
              f"{len(self):,} sequences (seq_len={seq_len}, stride={stride})")
    
    def _load_text_files(self, data_path):
        """Load all text from files"""
        text = []
        
        if os.path.isdir(data_path):
            # Load all .txt files from directory
            file_paths = glob.glob(os.path.join(data_path, "*.txt"))
            print(f"Found {len(file_paths)} text files in {data_path}")
        elif os.path.isfile(data_path):
            # Single file
            file_paths = [data_path]
        else:
            raise ValueError(f"Path {data_path} not found")
        
        for file_path in sorted(file_paths):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                text.append(content)
                print(f"  Loaded {file_path}: {len(content):,} characters")
        
        return ' '.join(text)
    
    def _build_vocab(self, text, max_vocab):
        """Build vocabulary from text"""
        print("Building vocabulary...")
        words = text.lower().split()
        counter = Counter(words)
        
        # Special tokens
        vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        
        # Add most common words
        for word, _ in counter.most_common(max_vocab - len(vocab)):
            vocab[word] = len(vocab)
        
        inverse_vocab = {v: k for k, v in vocab.items()}
        print(f"Vocabulary size: {len(vocab):,}")
        return vocab, inverse_vocab
    
    def _tokenize_text(self, text):
        """Convert text to token IDs"""
        words = text.lower().split()
        tokens = [self.vocab.get(word, self.vocab["<unk>"]) for word in words]
        return tokens
    
    def _split_data(self):
        """Split tokens into train/val/test"""
        n = len(self.tokens)
        if self.split == 'train':
            self.tokens = self.tokens[:int(0.8 * n)]
        elif self.split == 'val':
            self.tokens = self.tokens[int(0.8 * n):int(0.9 * n)]
        elif self.split == 'test':
            self.tokens = self.tokens[int(0.9 * n):]
    
    def __len__(self):
        # Number of sequences we can extract
        return max(0, (len(self.tokens) - self.seq_len) // self.stride + 1)
    
    def __getitem__(self, idx):
        """
        Key difference from classification:
        - Returns (input, target) where target is input shifted by 1
        - Example: input=[the, cat, sat], target=[cat, sat, on]
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len
        
        # Input sequence
        input_seq = self.tokens[start_idx:end_idx]
        
        # Target sequence (shifted by 1 for next-word prediction)
        target_seq = self.tokens[start_idx + 1:end_idx + 1]
        
        # Pad if necessary (for last sequences)
        if len(input_seq) < self.seq_len:
            input_seq = input_seq + [0] * (self.seq_len - len(input_seq))
            target_seq = target_seq + [0] * (self.seq_len - len(target_seq))
        
        return (torch.tensor(input_seq, dtype=torch.long),
                torch.tensor(target_seq, dtype=torch.long))

# ------------------------- Training Loop Changes -------------------------

def train_epoch_lm(model, dataloader, criterion, optimizer, device, scaler, args):
    """
    Key differences from classification training:
    1. Loss is computed over ALL positions, not just final output
    2. Metric is perplexity (exp(loss)) not accuracy
    3. Target is next word at each position
    """
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                # Get predictions for all positions
                output, _ = model(inputs)
                
                # Reshape for loss computation
                # output: (batch, seq_len, vocab) -> (batch * seq_len, vocab)
                # targets: (batch, seq_len) -> (batch * seq_len)
                loss = criterion(output.reshape(-1, output.size(-1)), 
                               targets.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output, _ = model(inputs)
            loss = criterion(output.reshape(-1, output.size(-1)), 
                           targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        
        # Count actual tokens (not padding)
        num_tokens = (targets != 0).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        
        # Print progress
        if batch_idx % 100 == 0 and args.rank == 0:
            curr_loss = loss.item()
            curr_ppl = np.exp(curr_loss)
            print(f"    Batch {batch_idx}/{len(dataloader)}: "
                  f"loss={curr_loss:.4f}, ppl={curr_ppl:.2f}")
    
    # Calculate epoch metrics
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity

@torch.no_grad()
def evaluate_lm(model, dataloader, criterion, device, args):
    """Evaluation for language modeling"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if args.amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                output, _ = model(inputs)
                loss = criterion(output.reshape(-1, output.size(-1)),
                               targets.reshape(-1))
        else:
            output, _ = model(inputs)
            loss = criterion(output.reshape(-1, output.size(-1)),
                           targets.reshape(-1))
        
        num_tokens = (targets != 0).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity

# ------------------------- Main Training Function -------------------------

def train(args):
    device = torch.device(args.device)
    
    # Build vocabulary from training data
    print(f"Loading data from {args.data_path}")
    train_dataset = LocalTextDataset(
        args.data_path, vocab=None, seq_len=args.seq_len, 
        stride=args.stride, max_vocab=args.max_vocab, split='train'
    )
    
    # Use same vocab for validation
    val_dataset = LocalTextDataset(
        args.data_path, vocab=train_dataset.vocab, seq_len=args.seq_len,
        stride=args.seq_len, max_vocab=args.max_vocab, split='val'  # No overlap in val
    )
    
    # Create data loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, 
                                       rank=args.rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size,
                                     rank=args.rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             sampler=train_sampler, num_workers=args.workers,
                             pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           sampler=val_sampler, num_workers=args.workers,
                           pin_memory=True, persistent_workers=True)
    
    # Model
    vocab_size = len(train_dataset.vocab)
    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tie_weights=args.tie_weights
    ).to(device)
    
    model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None)
    
    # Loss - ignore padding tokens (index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, 
                                          gamma=args.gamma)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if device.type == "cuda" else None
    
    if args.rank == 0:
        print(f"\nModel: LSTM Language Model")
        print(f"  Vocabulary size: {vocab_size:,}")
        print(f"  Embedding dim: {args.embed_dim}")
        print(f"  Hidden dim: {args.hidden_dim}")
        print(f"  Layers: {args.num_layers}")
        print(f"  Tie weights: {args.tie_weights}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"\nTraining:")
        print(f"  Sequence length: {args.seq_len}")
        print(f"  Stride: {args.stride}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print("=" * 60)
    
    best_val_ppl = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_ppl = train_epoch_lm(model, train_loader, criterion, 
                                               optimizer, device, scaler, args)
        
        # Evaluate
        val_loss, val_ppl = evaluate_lm(model, val_loader, criterion, device, args)
        
        epoch_time = time.time() - epoch_start
        
        if args.rank == 0:
            print(f"[Epoch {epoch:03d}] "
                  f"train_loss={train_loss:.4f}, train_ppl={train_ppl:.2f}, "
                  f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}, "
                  f"lr={scheduler.get_last_lr()[0]:.6f}, time={epoch_time:.1f}s")
            
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                print(f"  New best perplexity: {best_val_ppl:.2f}")
        
        scheduler.step()
    
    if args.rank == 0:
        print(f"\nTraining completed! Best validation perplexity: {best_val_ppl:.2f}")

# ------------------------- Setup and Entry -------------------------

def setup_ddp(args):
    # Standard DDP setup (same as classification)
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    
    if args.world_size > 1:
        dist.init_process_group(backend=args.backend, init_method="env://")

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='LSTM Language Model Training')

    parser.add_argument('--rank', type=int, default=0, help='Rank of current process')
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes')
    parser.add_argument('--iface', type=str, default="ens4f0", help='Network interface')
    parser.add_argument('--master_addr', type=str, default="42.0.0.1", help='Master node address')
    parser.add_argument("--master_port", type=int, default=29500, help='Master node port')
    parser.add_argument("--backend", type=str, default="gloo", choices=['gloo', 'nccl'], help="DDP backend")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cuda', help='Device to use for training')
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--workers", type=int, default=4)

    # Data arguments
    parser.add_argument('--data', type=str, required=True, help='Path to directory with .txt files or single .txt file')
    parser.add_argument('--max_vocab', type=int, default=50000, help='Maximum vocabulary size')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length for training')
    parser.add_argument('--stride', type=int, default=128, help='Stride for creating sequences (overlap if < seq_len)')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--tie_weights', action='store_true', help='Tie embedding and output weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument("--drop_last_train", action='store_true', help="Drop last batch from train dataset")
    parser.add_argument("--drop_last_val", action='store_true', help="Drop last batch from val dataset")
    parser.add_argument("--prefetch_factor", type=int, default=2, help='Prefetch factor for data loader')
    
    # System arguments
    # parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--backend', type=str, default='gloo')
    # parser.add_argument('--workers', type=int, default=4)
    # parser.add_argument('--amp', action='store_true')
    
    args = parser.parse_args()
    args.local_rank = 0
    args.rank = 0
    args.world_size = 1
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print("CUDA not available, using CPU")
    
    sys.stdout.reconfigure(line_buffering=True)
    setup_ddp(args)
    
    try:
        train(args)
    finally:
        cleanup()

if __name__ == '__main__':
    main()