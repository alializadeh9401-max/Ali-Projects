import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------
# Vocabulary
# -----------------------------

vocab = [
    "<PAD>",
    "I", "am", "Ali", "Hi", "Who", "is", "Smart",
    "Are", "you", "Yes", "No", "What", "your", "name", "My",
    "Nice", "to", "meet", "too", "are",
    "<SEP>", "<EoS>"
]

word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

pad_id = word_to_id["<PAD>"]
vocab_size = len(vocab)


# -----------------------------
# Hyperparameters
# -----------------------------

d_model = 32
max_len = 12
lr = 0.003
epochs = 5000
batch_size = 4


# -----------------------------
# One-hot token helper
# -----------------------------

def one_hot_token(token, word_to_id=word_to_id, vocab_size=vocab_size):
    vec = torch.zeros(vocab_size)
    vec[word_to_id[token]] = 1.0
    return vec


def encode_tokens(tokens, word_to_id=word_to_id, vocab_size=vocab_size):
    return torch.stack([
        one_hot_token(token, word_to_id, vocab_size)
        for token in tokens
    ])


def decode_ids(ids, id_to_word=id_to_word):
    return [id_to_word[int(i)] for i in ids]


def pad_tokens(tokens, max_len=max_len, pad_token="<PAD>"):
    tokens = tokens.copy()

    while len(tokens) < max_len:
        tokens.append(pad_token)

    return tokens[:max_len]


# -----------------------------
# Word embedding using one-hot input
# -----------------------------

class WordEmbedding(nn.Module):

    def __init__(self, vocab_size=vocab_size, d_model=d_model):
        super().__init__()

        self.tokenvec_to_hidden = nn.Linear(
            in_features=vocab_size,
            out_features=d_model,
            bias=True
        )

    def forward(self, x):
        hidden = self.tokenvec_to_hidden(x)
        return hidden


# -----------------------------
# Sinusoidal positional encoding
# -----------------------------

class PositionalEncoding(nn.Module):

    def __init__(self, d_model=d_model, max_len=max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            start=0,
            end=max_len,
            step=1
        ).float().unsqueeze(1)

        idx = torch.arange(
            start=0,
            end=d_model,
            step=2
        ).float()

        den = torch.tensor(10000.0) ** (idx / d_model)

        pe[:, 0::2] = torch.sin(position / den)
        pe[:, 1::2] = torch.cos(position / den)

        self.register_buffer("pe", pe)

    def forward(self, embeddings):
        seq_len = embeddings.shape[-2]
        return embeddings + self.pe[:seq_len]


# -----------------------------
# Masked self-attention block
# -----------------------------

class MaskedSelfAttentionBlock(nn.Module):

    def __init__(self, d_model=d_model):
        super().__init__()

        self.d_model = d_model

        self.q = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False
        )

        self.k = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False
        )

        self.v = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False
        )

        self.softmax = nn.Softmax(dim=-1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):

        seq_len = x.shape[-2]

        mask = torch.triu(
            torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=x.device,
                dtype=x.dtype
            ),
            diagonal=1
        )

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attention_scores = (Q @ K.transpose(-2, -1)) / (self.d_model ** 0.5)
        masked_attention_scores = attention_scores + mask

        attention_weights = self.softmax(masked_attention_scores)
        attention_output = attention_weights @ V

        # First residual connection and layer norm
        x = self.norm1(x + attention_output)

        # Feed-forward block
        ff_output = self.feed_forward(x)

        # Second residual connection and layer norm
        x = self.norm2(x + ff_output)

        return x


# -----------------------------
# Tiny decoder-only transformer
# -----------------------------

class TinyDecoderTransformer(nn.Module):

    def __init__(self, d_model=d_model, vocab_size=vocab_size, max_len=max_len):
        super().__init__()

        self.embedding = WordEmbedding(
            vocab_size=vocab_size,
            d_model=d_model
        )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len
        )

        self.transformer_block = MaskedSelfAttentionBlock(
            d_model=d_model
        )

        self.output_layer = nn.Linear(
            in_features=d_model,
            out_features=vocab_size,
            bias=True
        )

        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=pad_id
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_block(x)
        logits = self.output_layer(x)

        return logits

    def configure_optimizer(self):
        return Adam(self.parameters(), lr=lr)

    def training_step(self, batch):
        input_i, label_i = batch

        logits = self.forward(input_i)

        loss = self.loss_function(
            logits.transpose(1, 2),
            label_i
        )

        return loss


# -----------------------------
# Training dataset
# -----------------------------

training_sentences = [
    ["Hi", "<SEP>", "Hi", "<EoS>"],
    ["Who", "is", "Ali", "<SEP>", "Ali", "is", "Smart", "<EoS>"],
    ["What", "is", "your", "name", "<SEP>", "My", "name", "is", "Ali", "<EoS>"],
    ["Are", "you", "Smart", "<SEP>", "Yes", "I", "am", "Smart", "<EoS>"],
    ["Nice", "to", "meet", "you", "<SEP>", "Nice", "to", "meet", "you", "too", "<EoS>"],
]


input_tensors = []
label_tensors = []

for sentence in training_sentences:

    sentence = sentence.copy()

    while len(sentence) < max_len + 1:
        sentence.append("<PAD>")

    sentence = sentence[:max_len + 1]

    input_tokens = sentence[:-1]
    label_tokens = sentence[1:]

    input_one_hot = torch.stack([
        one_hot_token(token)
        for token in input_tokens
    ])

    label_ids = torch.tensor([
        word_to_id[token]
        for token in label_tokens
    ], dtype=torch.long)

    input_tensors.append(input_one_hot)
    label_tensors.append(label_ids)


inputs = torch.stack(input_tensors)   # [batch, seq_len, vocab_size]
labels = torch.stack(label_tensors)   # [batch, seq_len]

dataset = TensorDataset(inputs, labels)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# -----------------------------
# Training loop
# -----------------------------

def train(dataloader, epochs, model, optimizer):

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for batch in dataloader:

            optimizer.zero_grad()

            loss = model.training_step(batch)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss = {avg_loss:.6f}")


# -----------------------------
# Prediction over full sequence
# -----------------------------

def predict_next_tokens(model, tokens, word_to_id, id_to_word, vocab_size, max_len):

    model.eval()

    tokens = pad_tokens(
        tokens,
        max_len=max_len,
        pad_token="<PAD>"
    )

    x = encode_tokens(
        tokens,
        word_to_id,
        vocab_size
    )

    x = x.unsqueeze(0)

    with torch.no_grad():

        logits = model(x)
        pred_ids = torch.argmax(logits, dim=-1)[0]

    return tokens, decode_ids(pred_ids, id_to_word)


# -----------------------------
# Autoregressive generation
# -----------------------------

def generate(
    model,
    prompt_tokens,
    word_to_id,
    id_to_word,
    vocab_size,
    max_len,
    max_new_tokens=8
):

    model.eval()

    generated = prompt_tokens.copy()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            if len(generated) >= max_len:
                break

            x_tokens = pad_tokens(
                generated,
                max_len=max_len,
                pad_token="<PAD>"
            )

            x = encode_tokens(
                x_tokens,
                word_to_id,
                vocab_size
            )

            x = x.unsqueeze(0)

            logits = model(x)

            # Use the last real token position
            last_pos = len(generated) - 1

            next_logits = logits[0, last_pos]

            next_id = torch.argmax(next_logits).item()

            next_word = id_to_word[next_id]

            generated.append(next_word)

            if next_word == "<EoS>":
                break

    return generated


# -----------------------------
# Main script
# -----------------------------

if __name__ == "__main__":

    model = TinyDecoderTransformer(
        d_model=d_model,
        vocab_size=vocab_size,
        max_len=max_len
    )

    optimizer = model.configure_optimizer()

    train(
        dataloader=dataloader,
        epochs=epochs,
        model=model,
        optimizer=optimizer
    )

    print("\n--- Next-token prediction test ---")

    test_prompt = [
        "Who", "is", "Ali", "<SEP>",
        "Ali", "is", "Smart", "<EoS>"
    ]

    padded_input, predicted = predict_next_tokens(
        model,
        test_prompt,
        word_to_id,
        id_to_word,
        vocab_size,
        max_len=max_len
    )

    print("Input:    ", padded_input)
    print("Predicted:", predicted)

    print("\n--- Generation test ---")

    prompt = ["What", "is", "your", "name", "<SEP>"]

    generated = generate(
        model,
        prompt_tokens=prompt,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        vocab_size=vocab_size,
        max_len=max_len,
        max_new_tokens=8
    )

    response = generated[len(prompt):]

    print("Prompt:  ", " ".join(prompt))
    print("Full:    ", " ".join(generated))
    print("Response:", " ".join(response))

    print("\n--- Another generation test ---")

    prompt = ["Are", "you", "Smart", "<SEP>"]

    generated = generate(
        model,
        prompt_tokens=prompt,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        vocab_size=vocab_size,
        max_len=max_len,
        max_new_tokens=8
    )

    response = generated[len(prompt):]

    print("Prompt:  ", " ".join(prompt))
    print("Full:    ", " ".join(generated))
    print("Response:", " ".join(response))