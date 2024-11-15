import torch
import torch.nn as nn
from torch.nn import functional as F

import argparse
import os
import sys

import wave
from piper.voice import PiperVoice

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# print(f"Using {device}")

torch.manual_seed(1337)

# Load data
with open("./datasets/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Instantiate model
model = GPTLanguageModel().to(device)

# Command-line arguments
parser = argparse.ArgumentParser(
    description="Train or Generate Text with GPT Model and TTS"
)
parser.add_argument("--train", action="store_true", help="Train the model")
parser.add_argument(
    "--generate", action="store_true", help="Generate text from the model"
)
parser.add_argument(
    "--tts", action="store_true", help="Convert generated text to speech"
)
parser.add_argument(
    "--stream",
    action="store_true",
    help="Stream audio in real-time while generating text",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="store/attention.pth",
    help="Path to save/load the model weights",
)
parser.add_argument(
    "--starter_text",
    type=str,
    default="",
    help="Starting text for text generation (leave empty for random start)",
)
parser.add_argument(
    "--output_size",
    type=int,
    default=-1,
    help="Number of tokens to generate (default: -1 for infinite generation)",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="./output/generated_text.txt",
    help="Path to save the generated text output",
)
parser.add_argument(
    "--voice_model",
    type=str,
    default="./voices/en_US-ljspeech-medium.onnx",
    help="Path to the Piper voice model",
)
parser.add_argument(
    "--audio_path",
    type=str,
    default="./output/generated_audio.wav",
    help="Path to save the generated audio file",
)


args = parser.parse_args()

if args.train:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train loop
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")


# Load TTS model
def synthesize_audio(text, voice_model, audio_path):
    print("Loading TTS model...")
    voice = PiperVoice.load(voice_model)
    print("Synthesizing audio...")
    with wave.open(audio_path, "w") as wav_file:
        voice.synthesize(text, wav_file)
    print(f"Audio saved to {audio_path}")


if args.generate:
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
        print(f"Model loaded from {args.model_path}")
    else:
        print(f"Model file {args.model_path} not found. Exiting.")
        exit()

    # Use starter text or default to random
    start_text = args.starter_text
    if not start_text:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Random starter
    else:
        try:
            context = torch.tensor(
                [[stoi[ch] for ch in start_text]], dtype=torch.long, device=device
            )
        except KeyError as e:
            print(f"Error: Character '{e.args[0]}' not in vocabulary. Exiting.")
            exit()

    # Output size
    output_size = args.output_size

    # Output file path
    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Generated text (press Ctrl+C to stop if generating infinitely):")
    with open(output_path, "w") as f:
        token_count = 0
        generated_text = ""
        try:
            while True:
                idx_cond = context[:, -block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, idx_next), dim=1)

                # Decode and output the new token
                next_char = decode([idx_next.item()])
                sys.stdout.write(next_char)
                sys.stdout.flush()
                f.write(next_char)
                generated_text += next_char

                token_count += 1
                if output_size > 0 and token_count >= output_size:
                    break

        except KeyboardInterrupt:
            print("\nGeneration stopped by user.")

    print(f"\n\nGenerated text saved to {output_path}")

    # Text-to-Speech (TTS)
    if args.tts:
        synthesize_audio(generated_text, args.voice_model, args.audio_path)
