import torch
import torch.nn as nn
import math
torch.manual_seed(69)

vocab_size = 14
chars = "0123456789+=. " # '.':EOS, ' ':NULL
stoi = {c:i for i, c in enumerate(chars)}
encode = lambda str: [stoi[c] for c in str]
decode = lambda idxs: ''.join([chars[i] for i in idxs])

max_len = 10
rhs_max_len = 4
lhs_max = 1000
B = 64              # batch size
T = max_len + 2     # context length, including '+' and '='
embd_dim = 16          # embedding dim
# head_size = 8       # attention head size

def get_batch():
    batch = []
    for _ in range(B):
        a = torch.randint(1, lhs_max, (2,))
        sum = a.sum()
        padding = " " * (max_len - int(math.log10(sum))-1 - int(math.log10(a[0]))-1 - int(math.log10(a[1]))-1)
        s = f"{a[0]}+{a[1]}={padding}.{sum}"
        batch.append(s)
    return get_examples(batch)

# given batch of strings, encodes and converts to training examples
def get_examples(batch):
    Xb, Yb = [], []
    for x in batch:
        i = x.index('=')
        xb = x[:i+1]+x[:i+1:-1]
        yb = x[:T-(rhs_max_len+1):-1]
        Xb.append(encode(xb))
        Yb.append(encode(yb))
    # Yb = nn.functional.one_hot(torch.tensor(Yb).to('cuda'), vocab_size)
    return torch.tensor(Xb).to('cuda'), torch.tensor(Yb).to('cuda')

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(embd_dim, head_size, bias=False)
        self.key = nn.Linear(embd_dim, head_size, bias=False)
        self.value = nn.Linear(embd_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(T, T), diagonal=T-(rhs_max_len+1)))


    def forward(self, x):
        context_len = x.shape[1]
        if context_len >= T and self.training:
            queries = self.query(x[:, T-(rhs_max_len+1):, :]) # only predict answer  
        else:
            queries = self.query(x)
        keys = self.key(x)
        wei = queries @ keys.transpose(1, 2) / x.shape[2]**-0.5 # (B, rhs_max_len+1, H) @ (B, H, T) = (B, rhs_max_len+1, T)
        if context_len >= T and self.training:
            wei = wei.masked_fill(self.tril[:rhs_max_len+1, :context_len]==0, -torch.inf)
        else:
            wei = wei.masked_fill(self.tril[:context_len, :context_len]==0, -torch.inf)
        wei = torch.softmax(wei, dim=-1)

        values = self.value(x)
        out = wei @ values # (B, rhs_max_len+1, T) @ (B, T, H) = (B, rhs_max_len+1, H)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embd_dim)

    def forward(self, embd):
        x = torch.cat([head(embd) for head in self.heads], dim=-1)
        out = self.proj(x)
        return out # (B, rhs_max_len+1, embd_dim)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embd_dim, 4*embd_dim),
            nn.ReLU(),
            nn.Linear(4*embd_dim, embd_dim),
        )

    def forward(self, x):
        out = self.layers(x)
        return out

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.sa_head = MultiHeadAttention(num_heads=4, head_size=4) # multiply to embd_dim?
        self.ln2 = nn.LayerNorm(embd_dim)
        self.mlp = FeedForward()

    def forward(self, x):
        if x.shape[1] >= T and self.training:
            x = x[:, T-(rhs_max_len+1):, :] + self.sa_head(self.ln1(x)) # skip connections
        else:
            x = x + self.sa_head(self.ln1(x)) # skip connections
        out = x + self.mlp(self.ln2(x))
        return out

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, embd_dim, padding_idx=13) # padding works?
        self.pos_embedding = nn.Embedding(T, embd_dim)
        self.layers = nn.Sequential(
            Block(),
            nn.LayerNorm(embd_dim),
            nn.Linear(embd_dim, vocab_size)
        )
    
    def forward(self, x, y=None):
        context_len = x.shape[1]
        tok_embd = self.tok_embedding(x)
        pos_embd = self.pos_embedding(torch.arange(context_len, device='cuda'))
        embd = tok_embd + pos_embd
        logits = self.layers(embd) # (B, rhs_max_len+1, vocab_size)
        
        if y is None:
            loss = None
        else:
            logits = logits.view(-1, vocab_size)
            y = y.view(-1)
            loss = torch.nn.functional.cross_entropy(logits, y)
        return logits, loss
    
    def generate(self, context):
        next = -1
        ans = []
        while next != 12 and len(ans) < 10: # generate until EOS token
            cropped = context[:, -T:]
            logits, _ = self(cropped)
            logits = logits[:,-1,:]
            probs = nn.functional.softmax(logits, dim=-1)
            next = torch.multinomial(probs, num_samples=1)
            ans.insert(0, next.item())
            context = torch.cat((context, next), dim=1)
        return ans

if __name__ == "__main__":
    model = GPT().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(10000):
        
        if i % 100 == 0 and i > 0:
            print(f'Batch {i}, Loss: {loss}')
            # model.eval()
            # x = model.generate(torch.tensor([encode("123+456=")], device='cuda'))
            # print(f'123+456={decode(x)}')
            # model.train()
        
        Xb, Yb = get_batch()
        logits, loss = model(Xb, Yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "model.pt")
    