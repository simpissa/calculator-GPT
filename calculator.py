import torch

from train import GPT, encode, decode

model = GPT().to('cuda')
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()

# supposed to pad <3 digit numbers with space(s) after when generating, but didn't give enough 1 and 2 digits examples during training so can only do 3 digit addition anyways :(

while True:
    x = input("Enter problem in form 'x'+'y'=: ")
    ans = decode(model.generate(torch.tensor([encode(x)], device='cuda')))
    print(f'Answer: {ans}')

