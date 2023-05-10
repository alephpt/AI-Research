import os, re, torch, time
import os, re, torch, time
import torch.nn as nn
from torch.nn import functional as F

dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def gather_input_data():
    texts = []

    # see if we can open the pre-made text file
    if os.path.isfile('training_data.txt'):
        # if so, open it and read it
        with open('training_data.txt', 'r', encoding='utf-8') as f:
            texts = f.read()
    else:
        # if not, gather all the text files in the training_data folder
        files = os.listdir('training_data')

        for file in files:
            with open('training_data/' + file, 'r', encoding='utf-8') as f:
                texts.append(f.read())

        texts = ''.join(texts)

        # save the text file for future use
        with open('training_data.txt', 'w', encoding='utf-8') as f:
            f.write(texts)

    return texts

def sort_tokens(strings):
    # then split the strings into a list of chars
    return sorted(list(set("".join(strings))))

def construct_training_data(strings, encode):
    # Encode Data
    data = torch.tensor(encode(strings), dtype=torch.long)
    print("\tData Loaded:\t\t", data.dtype, data.shape)

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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Bx, Tx, Cx = x.shape
        k = self.key(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1) * (1.0 / (Cx ** 0.5))
        weight = weight.masked_fill(self.tril[:Tx, :Tx] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        v = self.value(x)
        out = weight @ v
        return out


class MultiHead(nn.Module):
    def __init__(self, block_size, head_size, n_embeds=16, n_heads=8):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, head_size, n_embeds) for _ in range(n_heads)])
        self.linear = nn.Linear(n_heads * head_size, n_embeds)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.linear(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embeds=16, n_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeds, 4 * n_hidden),
            nn.ReLU(),
            nn.Linear(4 * n_hidden, n_embeds),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, block_size, n_embeds=16, n_heads=8):
        super().__init__()
        head_size = n_embeds // n_heads
        self.head = MultiHead(block_size, head_size, n_embeds, n_heads)
        self.ff = FeedForward(n_embeds, n_embeds)
        self.ln = nn.LayerNorm(n_embeds)
        self.ln2 = nn.LayerNorm(n_embeds)

    def forward(self, x):
        x = x + self.head(self.ln(x))
        x = x + self.ff(self.ln2(x))
        return x


class BigramModel(nn.Module):
    def __init__(self, block_size, vocab_size, n_embeds=16):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, n_embeds)
        self.position_embed = nn.Embedding(block_size, n_embeds)
        self.block = nn.Sequential(
            Block(block_size, n_embeds),
            Block(block_size, n_embeds),
            Block(block_size, n_embeds),
            #Block(block_size, n_embeds),
            nn.LayerNorm(n_embeds)
        )
        self.linear_head = nn.Linear(n_embeds, vocab_size)

    def forward(self, x, targets=None):
        Bx, Tx = x.shape

        token_embed = self.embedding(x)
        position_embed = self.position_embed(torch.arange(Tx, device=device))
        tx = token_embed + position_embed
        tx = self.block(tx)
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
    continue_training = True
    model_loaded = False
    eval_iterations = 50
    max_iterations = 500
    learning_rate = 0.0005
    batch_size = 188
    block_size = 1024
    n_embeds = 256
    strings = gather_input_data()
    chars = sort_tokens(strings)
    vocab_size = len(chars)
    model_path = "large_gpt_model.pt"
    save_path = "generated_lg.txt"
    model = BigramModel(block_size, vocab_size, n_embeds)
    m = model.to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for i, s in enumerate(chars)}
    encode = lambda x: [stoi[s] for s in x]
    decode = lambda x: ''.join([itos[i] for i in x])
    encoded = encode("Never Outshine the Master\n")

    # basic encoder/decoder test
    print(encoded)
    print(decode(encoded))

    # load model if it exists
    if os.path.isfile(model_path):
        print(f'Loading model {model_path}...')
        m.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model_loaded = True
        print("Loaded:", model_path)

    # train model if it doesn't exist or if we want to continue training
    if continue_training or not model_loaded:
        import gc
        print("Vocab size: " + str(vocab_size))
        print("Grammar:\n", ''.join(chars[3:]), "\n")

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

        print("\tMax Iterations: \t" + str(max_iterations))
        print("\tEval Iterations: \t" + str(eval_iterations))
        print("\tLearning Rate: \t\t" + str(learning_rate))
        print("\tBatch Size: \t\t" + str(batch_size))
        print("\tBlock Size: \t\t" + str(block_size))
        print("\tEmbedding Size: \t" + str(n_embeds))

        # Construct Training Data
        training_data, validation_data = construct_training_data(strings, encode)
        print("\tTraining Data:\t\t", training_data.shape, training_data.dtype)
        print("\tValidation Data:\t", validation_data.shape, validation_data.dtype)
        start_loss = estimate_loss()
        print("\tCurrent Loss: \t\t" + str(start_loss) + "\n")

        elapsed_t = time.time()
        start_time = elapsed_t
        print_t = time.strftime("%H:%M:%S", time.gmtime(elapsed_t))
        print("\nTraining... starting at " + str(print_t) + "\n")

        # Training Loop
        for steps in range(max_iterations):
            torch.cuda.empty_cache()
            gc.collect()

            if steps % eval_iterations == 0:
                new_t = time.time()
                elapsed_t = new_t - elapsed_t

                losses = estimate_loss()

                print(f"Step: {steps} \t Time: {elapsed_t:09.4f} \t", end="")
                print(f" Train Loss: {losses['train']:.4f} \t Valid Loss: {losses['valid']:.4f}")
                elapsed_t = new_t

            xb, yb = get_batch("train")
            _, loss = m(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        end_loss = estimate_loss()
        training_loss = end_loss['train'] - start_loss['train']
        validation_loss = end_loss['valid'] - start_loss['valid']
        print("\nTraining Completed at " + str(time.strftime("%H:%M:%S", time.gmtime(time.time()))) + "\n")
        print("\tTotal Training Loss: " + str(training_loss))
        print("\tTotal Validation Loss: " + str(validation_loss))
        print("\tTotal Training Time: " + str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))) + "\n")



        #save model
        print(f'Saving model {model_path}...')
        torch.save(m.state_dict(), model_path)

    print("Generating text...")
    context = m.generate(x = torch.zeros((1, 1), dtype=torch.long).to(device), n=600)
    generated = decode(context[0].tolist())

    print("Saving generated text...")
    with open(save_path, 'w') as f:
      f.write(generated)

    print("\nGenerated:\n", generated)

if __name__ == "__main__":
    main()
