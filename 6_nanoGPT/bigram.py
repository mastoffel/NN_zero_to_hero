import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 6000
eval_interval = 500
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ---------------

torch.manual_seed(0)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# make encoder / decoder    
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # take a string, output list of ints
decode = lambda i: ''.join([itos[n] for n in i]) # take list of ints, output string
print(decode(encode('hello world!')))

# train and test splits
data = torch.tensor(encode(text), dtype = torch.long) # 64bit integers
n = int(0.9 * len(data)) # rounds down
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x.to(device)
    y.to(device)
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
        out[split] = losses.mean().item()
    model.train()
    return out

# bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        
        # idx and target are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx) # (B, T, C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # (B*T, C)
            targets = targets.view(B*T) # (B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) tensor of integers
        for _ in range(max_new_tokens):
            # get the logits for the next token
            logits, loss = self.forward(idx)
            # focus on the last time step
            logits = logits[:, -1, :] # (B, C)
            # apply softmax to get a probability distribution
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution to get a new token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append to the sequence
            idx = torch.cat([idx, idx_next], dim=-1) # (B, T+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'iter {iter} | train loss {losses["train"]} | val loss {losses["val"]}')
        
    # sample a batch of training data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# generate some text
context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
        
