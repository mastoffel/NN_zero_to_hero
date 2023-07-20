import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ---------------

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique characters in the file
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encoder = lambda s: [stoi[c] for c in s] # take a string, output list of ints
decoder = lambda i: ''.join([itos[n] for n in i]) # take list of ints, output string

# train and test splits
data = torch.tensor(encoder(text), dtype = torch.long)
n = int(0.9 * len(data)) # rounds down
train_data = data[:n]
val_data = data[n:]

idx = torch.randint(0, len(data)-block_size, (batch_size,))
x=[data[i:i+block_size] for i in idx]
print(torch.stack(x[:3]).shape)

# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(0, len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key() = nn.Linear(n_embd, head_size, bias=False)
        self.query() = nn.Linear(n_embd, head_size, bias=False)
        self.value() = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # input of size (B,T,C)
        # output of size (B,T,head size)
        B, T, C = x.shape
        k = self.key(x) # B, T, hs
        q = self.key(x) # B, T, hs
        # attention
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) @ (B, T, T) ---> (B, T, T)
        wei = F.softmax(wei, dim=-1) # B, T, T
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # B, T, hs
        out = wei @ v # B, T, hs
        return out
    
    class MultiHeadAttention(nn.Module):
        """ multiple heads of self attention """
    
        def __init__(self, num_heads, head_size):
            super().__init__()        
            self.heads = nn.ModuleList(Head(head_size) for _ in num_heads)
            self.proj = nn.Linear(head_size * num_heads, n_embd)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out
        
    