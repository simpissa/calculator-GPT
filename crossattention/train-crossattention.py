import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.manual_seed(69)

vocab_size = 17
chars = "0123456789+_-=., " # '_':subtract      '-':minus      '.':EOS      ',':start token      ' ':padding
stoi = {c:i for i, c in enumerate(chars)}
encode = lambda str: [stoi[c] for c in str]
decode = lambda idxs: ''.join([chars[i] for i in idxs])

embd_dim = 16
batch_size = 64
lhs_max = 999
max_left = 10
max_right = 7

def get_batch():
    batch = [] 
    qs = [] # questions for encoder
    padding_mask_qs = []
    padding_mask_batch = []
    for _ in range(batch_size):
        a = torch.randint(-lhs_max, lhs_max+1, (2,))
        if torch.randint(0,2, (1,)).item():
            op = "_"
            result = (a[0]-a[1]).item()
        else:
            op = "+"
            result = a.sum().item()
        sign = "-" if result < 0 else "" # add negative sign to end   (add positive sign too?)

        q = f"{a[0]}{op}{a[1]}="
        ans = f".{abs(result)}{sign},"

        # padding
        padding_mask_qs.append([1]*len(q) + [0]*(max_left-len(q)))
        padding_mask_batch.append([1]*(len(ans)-1) + [0]*(max_right-len(ans)))
        q += " " * (max_left-len(q))
        ans = " " * (max_right-len(ans)) + ans
        qs.append(encode(q))
        batch.append(ans)
        
    return torch.tensor(qs).to('cuda'), get_examples(batch), torch.tensor(padding_mask_qs).to('cuda'), torch.tensor(padding_mask_batch).to('cuda')

def get_examples(batch):
    Xb, Yb = [], []
    for x in batch:
        xb = x[:0:-1]
        yb = x[len(x)-2::-1]
        Xb.append(encode(xb))
        Yb.append(encode(yb))
    return torch.tensor(Xb).to('cuda'), torch.tensor(Yb).to('cuda')

class Head(nn.Module):
    def __init__(self, head_size, context_len, encoder):
        super().__init__()
        self.query = nn.Linear(embd_dim, head_size, bias=False)
        self.key = nn.Linear(embd_dim, head_size, bias=False)
        self.value = nn.Linear(embd_dim, head_size, bias=False)
        self.encoder = encoder
        self.register_buffer('tril', torch.tril(torch.ones(context_len, context_len)))

    def forward(self, x, ca_x=None):
        B, T, C = x.shape

        if ca_x is not None: # cross-attention head
            # truncate to match decoder
            ca_x = ca_x[:,:T]
            keys = self.key(ca_x)
            queries = self.key(ca_x)
            
        else:
            keys = self.key(x)
            queries = self.query(x)
        values = self.value(x)

        wei = queries @ keys.transpose(1, 2) / C**-0.5
        if not self.encoder: # don't apply mask to encoder block
            wei = wei.masked_fill(self.tril[:T, :T]==0, -torch.inf)
        wei = torch.softmax(wei, dim=-1)
        out = wei @ values
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, context_len, encoder):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, context_len, encoder) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, embd_dim)

    def forward(self, embd, ca_x=None):
        if ca_x is not None:
            x = torch.cat([head(embd, ca_x) for head in self.heads], dim=-1)    
        else:
            x = torch.cat([head(embd) for head in self.heads], dim=-1)
        out = self.proj(x)
        return out

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
    
class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.sa_head = MultiHeadAttention(4, 4, max_left, True)
        self.ln2 = nn.LayerNorm(embd_dim)
        self.mlp = FeedForward()

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        out = x + self.mlp(self.ln2(x))
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.sa_head = MultiHeadAttention(4, 4, max_right, False)
        self.ln2 = nn.LayerNorm(embd_dim)
        self.ln22 = nn.LayerNorm(embd_dim) # for ca_x
        self.ca_head = MultiHeadAttention(4, 4, max_left, False)
        self.ln3 = nn.LayerNorm(embd_dim)
        self.mlp = FeedForward()
    
    def forward(self, x, ca_x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ca_head(self.ln2(x), self.ln22(ca_x))
        out = x + self.mlp(self.ln3(x))
        return out
    
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_tok_embedding = nn.Embedding(vocab_size, embd_dim, padding_idx=16)
        self.encoder_pos_embedding = nn.Embedding(max_left, embd_dim)
        self.decoder_tok_embedding = nn.Embedding(vocab_size, embd_dim, padding_idx=16)
        self.decoder_pos_embedding = nn.Embedding(max_right, embd_dim)
        self.encoder_layers = nn.Sequential(
            EncoderBlock(),
            EncoderBlock()
            )
        self.decoder = DecoderBlock()
        self.decoder2 = DecoderBlock()
        self.decoder_layers = nn.Sequential(
            nn.LayerNorm(embd_dim),
            nn.Linear(embd_dim, vocab_size)
        )
    
    def forward(self, x, ca_x, y=None, padding_q=None, padding_ans=None):
        B, T = x.shape
        Be, Te = ca_x.shape

        encoder_tok_embd = self.encoder_tok_embedding(ca_x)
        encoder_pos_embd = self.encoder_pos_embedding(torch.arange(Te, device='cuda'))
        encoder_embd = encoder_tok_embd + encoder_pos_embd

        decoder_tok_embd = self.decoder_tok_embedding(x)
        decoder_pos_embd = self.decoder_pos_embedding(torch.arange(T, device='cuda'))
        decoder_embd = decoder_tok_embd + decoder_pos_embd

        if padding_q is not None and padding_ans is not None:
            padding_q = padding_q.view(padding_q.shape[0], padding_q.shape[1], 1) # broadcasting
            padding_ans = padding_ans.view(padding_ans.shape[0], padding_ans.shape[1], 1)
            # attention mask for padding
            encoder_embd = encoder_embd * padding_q
            decoder_embd = decoder_embd * padding_ans

        encode_x = self.encoder_layers(encoder_embd)
        temp = self.decoder(decoder_embd, encode_x)
        temp2 = self.decoder2(temp, encode_x)
        logits = self.decoder_layers(temp2)

        if y is None:
            loss = None
        else:
            logits = logits.view(-1, vocab_size)
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)
        return logits, loss
    
    def generate(self, q):
        next = -1
        ans = []
        context = torch.tensor([encode(["."])], device='cuda')
        while next != encode(["."])[0] and len(ans) < 10: # generate until EOS token
            cropped = context[:, -max_right:]
            logits, _ = self(cropped, q)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            next = torch.multinomial(probs, num_samples=1)
            ans.insert(0, next.item())
            context = torch.cat((context, next), dim=1)
        return ans
    
if __name__ == "__main__":
    model = GPT().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    cumulative_loss = 0
    for i in range(50000):
        
        # expected random loss is 2.7725
        if i % 1000 == 0 and i > 0:
            print(f'Batch {i}, Loss: {cumulative_loss / 1000}')
            cumulative_loss = 0
            model.eval()
            x = model.generate(torch.tensor([encode("123+456=")], device='cuda'))
            print(f'123+456={decode(x)}')
            model.train()
        
        qs, (Xb, Yb), padding_mask_qs, padding_mask_batch = get_batch()
        logits, loss = model(Xb, qs, y=Yb, padding_q=padding_mask_qs, padding_ans=padding_mask_batch)
        cumulative_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "ca_model.pt")