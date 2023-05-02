import os
import torch
import torch.nn as nn
from torch.nn import functional as F

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

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        logits = self.embedding(x)

        if targets is None:
            return logits, None
        else :
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

            return logits, loss

    def generate(self, x, n):
        for _ in range(n):
            logits, loss = self(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, idx_next], dim=1)
        return x


def main():
    batch_size = 32
    block_size = 8
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
        return x, y

    # xb, yb = get_batch("train")
    # print("\nInput Batch:", xb.shape, "\n", xb, "\n")
    # print("Target Batch:", yb.shape, "\n", yb, "\n")

    # for b in range(batch_size):
    #     for t in range(block_size):
    #         context = xb[b, :t+1]
    #         target = yb[b, t]
    #         print(f"Context: {context.tolist()} \t Target: {target}")

    m = BigramModel(vocab_size)
    # logits, loss = m(xb, yb)
    # print("Logits:", logits.shape)
    # print("Loss:", loss)

    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

    for steps in range(100):
        xb, yb = get_batch("train")
        logits, loss = m(xb, yb)
        print("Step:", steps, "\tLoss:", loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print("Generated:", decode(m.generate(xb, n=10)))


if __name__ == "__main__":
    main()
