import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# =========================
# 1. Data Preparation
# =========================

# File path to your dataset
file_path = "data/wiki_train_tokens.txt"

# Read the file
with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# Create a sorted list of unique characters
chars = sorted(list(set(content)))
vocab_size = len(chars)
print(f"Unique characters: {vocab_size}")

# Create mappings from characters to indices and vice versa
char_to_index = {char: idx for idx, char in enumerate(chars)}
index_to_char = {idx: char for idx, char in enumerate(chars)}

# Tokenizer function: Convert text to sequence of integers
def encoder(text):
    return [char_to_index[char] for char in text if char in char_to_index]

# Detokenizer function: Convert sequence of integers back to text
def decoder(indices):
    return ''.join([index_to_char[idx] for idx in indices])

# Test the encoder and decoder
sample_text = "hello world"
encoded_sample = encoder(sample_text)
decoded_sample = decoder(encoded_sample)
print(f"Encoded '{sample_text}': {encoded_sample}")
print(f"Decoded back: '{decoded_sample}'")

# Convert the entire dataset to a PyTorch tensor
data = torch.tensor(encoder(content), dtype=torch.long)
print(f"Dataset size: {data.shape[0]} characters")

# Define device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# 2. Hyperparameters
# =========================

block_size = 128       # Context length
batch_size = 64        # Number of sequences per batch
embedding_dim = 256    # Embedding size
hidden_dim = 512       # LSTM hidden size
num_layers = 2         # Number of LSTM layers
num_epochs = 10        # Number of training epochs
learning_rate = 1e-3   # Learning rate

# =========================
# 3. Data Splitting
# =========================

# Split the data into training and validation sets (90% train, 10% val)
n = len(data)
train_size = int(n * 0.9)
train_data = data[:train_size]
val_data = data[train_size:]
print(f"Training data size: {train_data.shape[0]} characters")
print(f"Validation data size: {val_data.shape[0]} characters")

# =========================
# 4. Batch Generation Function
# =========================

def get_batch(split):
    """
    Generates a batch of input-target pairs for training or validation.

    Args:
        split (str): 'train' or 'val' to specify the dataset split.

    Returns:
        x (torch.Tensor): Input tensor of shape [batch_size, block_size]
        y (torch.Tensor): Target tensor of shape [batch_size, block_size]
    """
    data_ = train_data if split == 'train' else val_data
    # Randomly choose batch_size starting indices
    ix = torch.randint(0, len(data_) - block_size - 1, (batch_size,))
    # Slice the data to get inputs and targets
    x = torch.stack([data_[i : i + block_size] for i in ix])
    y = torch.stack([data_[i + 1 : i + block_size + 1] for i in ix])
    # Move to the specified device
    x, y = x.to(device), y.to(device)
    return x, y

# =========================
# 5. Model Definition
# =========================

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        """
        A simple character-level RNN model using LSTM.

        Args:
            vocab_size (int): Number of unique characters.
            embedding_dim (int): Dimension of the embedding vectors.
            hidden_dim (int): Number of features in the hidden state of LSTM.
            num_layers (int): Number of recurrent layers in LSTM.
        """
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, block_size]
            hidden (tuple): Hidden state for LSTM (optional)

        Returns:
            out (torch.Tensor): Output logits of shape [batch_size, block_size, vocab_size]
            hidden (tuple): Hidden state from LSTM
        """
        embeds = self.embedding(x)  # [batch_size, block_size, embedding_dim]
        if hidden is None:
            out, hidden = self.lstm(embeds)  # out: [batch, seq, hidden_dim]
        else:
            out, hidden = self.lstm(embeds, hidden)
        logits = self.fc(out)  # [batch_size, block_size, vocab_size]
        return logits, hidden

# Initialize the model
model = CharRNN(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
print(model)

# =========================
# 6. Loss and Optimizer
# =========================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# =========================
# 7. Training Loop
# =========================

for epoch in range(1, num_epochs + 1):
    model.train()  # Set model to training mode
    epoch_loss = 0.0
    num_batches = 100  # Number of batches per epoch (adjust as needed)

    for batch in range(num_batches):
        x_batch, y_batch = get_batch('train')  # Get a batch of data
        optimizer.zero_grad()                  # Clear gradients

        logits, _ = model(x_batch)             # Forward pass
        # Reshape logits and targets for loss computation
        logits = logits.view(-1, vocab_size)   # [batch_size * block_size, vocab_size]
        y_batch = y_batch.view(-1)             # [batch_size * block_size]

        loss = criterion(logits, y_batch)      # Compute loss
        loss.backward()                         # Backpropagate
        optimizer.step()                        # Update parameters

        epoch_loss += loss.item()

        if (batch + 1) % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch+1}/{num_batches}], Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / num_batches
    print(f"--- Epoch [{epoch}/{num_epochs}] completed. Average Loss: {avg_loss:.4f} ---\n")

    # Optionally, evaluate on validation set
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        val_loss = 0.0
        val_batches = 10  # Number of validation batches
        for _ in range(val_batches):
            x_val, y_val = get_batch('val')
            logits, _ = model(x_val)
            logits = logits.view(-1, vocab_size)
            y_val = y_val.view(-1)
            loss = criterion(logits, y_val)
            val_loss += loss.item()
        avg_val_loss = val_loss / val_batches
        print(f"Validation Loss after Epoch {epoch}: {avg_val_loss:.4f}\n")

print("Training completed!")

# =========================
# 8. Saving the Model
# =========================

torch.save(model.state_dict(), "char_rnn_model.png")
print("Model saved to 'char_rnn_model.pth'")

# =========================
# 9. Example of Text Generation (Optional)
# =========================

def generate_text(model, start_text, length=100):
    """
    Generates text using the trained model.

    Args:
        model (nn.Module): Trained character-level RNN model.
        start_text (str): Initial string to start the generation.
        length (int): Number of characters to generate.

    Returns:
        generated_text (str): The generated text string.
    """
    model.eval()
    generated = encoder(start_text)
    input_seq = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
    hidden = None

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_seq, hidden)
            # Get the logits for the last character in the sequence
            last_logits = logits[:, -1, :]  # [1, vocab_size]
            # Apply softmax to get probabilities
            probs = F.softmax(last_logits, dim=-1)
            # Sample from the distribution
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            # Append to the generated sequence
            generated.append(next_char_idx)
            # Update the input sequence
            input_seq = torch.tensor([next_char_idx], dtype=torch.long).unsqueeze(0).to(device)

    return decoder(generated)

# Example usage:
start = "The "
generated = generate_text(model, start, length=200)
print(f"--- Generated Text ---\n{generated}\n")
