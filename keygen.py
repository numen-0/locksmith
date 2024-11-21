"""
NOTE: the objective of this is to generate a PERFECT encoder decoder so in the
      after the train the acc has to be 100.0%.

      for now we have a seed set, but the seed would be random and the program
      sould get a well trained model in a reasonable time.

      Im thinking of changing the network structure to get better and faster
      results.
"""
import torch
import torch.nn as nn
import torch.optim as optim

import time
import random


# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Hyperparameters
LATENT_DIM = 8        # Dimension of encoded representation
BATCH_SIZE = 2 ** 10  # Ensure this is a power of 2
EPOCHS = 2 ** 64      # NOTE: train loops will be EPOCHS * ERAS
ERAS = 2 ** 10
LEARNING_RATE = 0.0001

# Ensure BATCH_SIZE is a power of 2
if BATCH_SIZE & (BATCH_SIZE - 1) != 0:
    raise ValueError("BATCH_SIZE must be a power of 2.")

# Input type and value range
"""
NOTE: we can't use uint32 beacuse float32 tremples with small values, so we need
a space to represent our data without loosing data back. The IEEE 754 float:
 31   30-23    22-0
|sing|exponent|fraction|
so a sort can fit inside the 23bit
"""
dtype = torch.uint16  # Change to uint16 for 16-bit unsigned integers
bytes_n = 2  # 2 bytes (16-bit integers)
N = (1 << (8 * bytes_n))  # Number of unique byte values
max_int = N - 1  # max_int is now 65535 for uint16


# net #########################################################################
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, latent_dim),
        )

    def forward(self, x):
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, 128)
        self.decoder = Decoder(128, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# fun #########################################################################
def sample_random_dataset(batch_size, key_length):
    """
    Generate a random batch of `batch_size` keys with `key_length` elements.
    The byte data will be reinterpreted as float32.
    """
    dataset = [
        [random.randint(0, max_int) for _ in range(key_length)]
        for _ in range(batch_size)
    ]

    dataset_tensor = torch.tensor(dataset, dtype=dtype)

    return dataset_tensor


def train_model(key_length):
    batch_size = BATCH_SIZE
    latent_dim = LATENT_DIM
    epochs = EPOCHS
    eras = ERAS
    print("training model:")
    print(f"    key_length: {key_length} (input vect dim (decoded))")
    print(f"    key_size:   {bytes_n}B   (each vect value size)")
    print(f"    latent_dim: {LATENT_DIM} (out vect dim (encoded))")
    print(f"    batch_size: {BATCH_SIZE}")
    print(f"    epochs:     {EPOCHS}")

    model = Autoencoder(key_length, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Record overall training start time
    overall_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        for _ in range(eras):
            train_batch = sample_random_dataset(batch_size, key_length)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(train_batch.float())
            loss = criterion(outputs, train_batch.float())

            # Backward pass and optimization
            loss.backward()
            #  Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Record epoch duration
        epoch_duration = time.time() - epoch_start_time
        print(f"    epoch [{epoch}/{epochs}], loss: {loss.item():.4f}, "
              f"time: {epoch_duration:.2f}s")

    # Record overall training duration
    overall_duration = time.time() - overall_start_time
    print(f"Training completed in {overall_duration:.2f}s")

    return model


# test ########################################################################
def test_autoencoder(model, key_length, batch_size=10):
    """
    Tests the autoencoder model on random samples and prints the original and
    decoded keys.

    :param model: The trained autoencoder model.
    :param batch_size: The number of test samples to use.
    :param key_length: The length of each key in the dataset.
    """
    # Generate a random test batch
    test_keys = sample_random_dataset(batch_size, key_length)

    print("=" * 50)
    print(f"Testing the model on {batch_size} random samples...\n")

    with torch.no_grad():  # Disable gradient computation for inference
        for idx, key in enumerate(test_keys):
            # Encode and decode the key
            encoded = model.encoder(key.unsqueeze(
                0).float())  # Add batch dimension
            decoded = model.decoder(encoded)

            # Print original and decoded keys
            print(f"Test Sample {idx + 1}:")
            print(f"Original Key:  {key.int().tolist()}")
            print(f"Decoded Key:   {decoded}")
            print("-" * 50)


# main ########################################################################
if __name__ == "__main__":
    key_length = 1
    model = train_model(key_length)
    test_autoencoder(model, key_length)
