import os
import torch
import torch.nn as nn
from torch.nn import functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_training_data(strings):
    # open all the files in the training_data folder
    files = os.listdir("training_data")

    # read the files into a list of strings
    for file in files:
        with open("training_data/" + file, "r", encoding="utf8") as f:
            strings += f.read()

    # then split the strings into a list of chars
    return sorted(list(set("".join(strings))))

def construct_training_data(strings, encode):
    # Encode Data
    data = torch.tensor(encode(strings), dtype=torch.long)
    print("Data Loaded:\t\t", data.dtype, data.shape)

    # Split Data
    training_split = int(len(data) * 0.9)
    training_data = data[:training_split]
    validation_data = data[training_split:]

    return training_data, validation_data


class Head(nn.Module):
    def __init__(self, block_size, head_size, n_embeds=16):
        super().__init__()
        self.key = nn.Linear(n_embeds, head_size, bias=False)
        self.query = nn.Linear(n_embeds, head_size, bias=False)
        self.value = nn.Linear(n_embeds, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        Bx, Tx, Cx = x.shape
        k = self.key(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1) * (1.0 / (Cx ** 0.5))
        weight = weight.masked_fill(self.tril[:Tx, :Tx] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        v = self.value(x)
        out = weight @ v
        return out


class MultiHead(nn.Module):
    def __init__(self, block_size, head_size, n_embeds=16, n_heads=8):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, head_size, n_embeds) for _ in range(n_heads)])
        #self.linear = nn.Linear(n_heads * head_size, n_embeds)
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        #out = self.linear(out)
        return out


class BigramModel(nn.Module):
    def __init__(self, block_size, vocab_size, n_embeds=16):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, n_embeds)
        self.position_embed = nn.Embedding(block_size, n_embeds)
        self.head = MultiHead(block_size, n_embeds // 8, n_embeds, 8)
        self.linear_head = nn.Linear(n_embeds, vocab_size)

    def forward(self, x, targets=None):
        Bx, Tx = x.shape
        
        token_embed = self.embedding(x)
        position_embed = self.position_embed(torch.arange(Tx, device=device))
        tx = token_embed + position_embed
        tx = self.head(tx)
        logits = self.linear_head(tx)

        if targets is None:
            return logits, None
        else :
            Bl, Tl, Cl = logits.shape
            logits = logits.view(Bl*Tl, Cl)
            targets = targets.view(Bl*Tl)
            loss = F.cross_entropy(logits, targets)

            return logits, loss

    def generate(self, x, n):
        for _ in range(n):
            x_conditional = x[:, -self.block_size:]
            logits, loss = self(x_conditional)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, idx_next], dim=1)
        return x


def main():
    eval_iterations = 250
    max_iterations = 16725
    learning_rate = 0.00000725
    batch_size = 32
    block_size = 16
    n_embeds = 32
    strings = []
    # TODO: Turn this into WORD Matching instead of CHAR Matching
    # TODO: Implement Sub-Word Schema
    chars = load_training_data(strings)
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda x: [stoi[ch] for ch in x]
    decode = lambda x: ''.join([itos[i] for i in x])

    print(''.join(chars[3:]))
    print("Vocab size: " + str(vocab_size) + "\n")

    encoded = encode("Never Outshine the Master\n")
    print(encoded)
    print(decode(encoded))

    # Construct Training Data
    training_data, validation_data = construct_training_data(strings, encode)
    print("Training Data:\t\t", training_data.dtype, training_data.shape)
    print("Validation Data:\t", validation_data.dtype, validation_data.shape)

    def get_batch(training_split):
        data = training_data if training_split == "train" else validation_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        
        for split in ["train", "valid"]:
            losses = torch.zeros(eval_iterations)
            for k in range(eval_iterations):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        
        return out
    
    # xb, yb = get_batch("train")
    # print("\nInput Batch:", xb.shape, "\n", xb, "\n")
    # print("Target Batch:", yb.shape, "\n", yb, "\n")

    # for b in range(batch_size):
    #     for t in range(block_size):
    #         context = xb[b, :t+1]
    #         target = yb[b, t]
    #         print(f"Context: {context.tolist()} \t Target: {target}")

    model = BigramModel(block_size, vocab_size, n_embeds)
    m = model.to(device)
    # logits, loss = m(xb, yb)
    # print("Logits:", logits.shape)
    # print("Loss:", loss)

    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

    for steps in range(max_iterations):
        if steps % eval_iterations == 0:
            losses = estimate_loss()
            print(f"Step: {steps} \t Train Loss: {losses['train']:.4f} \t Valid Loss: {losses['valid']:.4f}")
            
        xb, yb = get_batch("train")
        logits, loss = m(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Generated:", decode(m.generate(x = torch.zeros((1, 1), dtype=torch.long), n=125)[0].tolist()))


if __name__ == "__main__":
    main()
