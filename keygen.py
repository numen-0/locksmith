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
import numpy as np

import time
from math import log2
import random


# Set random seed for reproducibility
torch.manual_seed(0xabad_1dea)
random.seed(0xabad_1dea)

# Hyperparameters # Ensure all are power of 2 for bigger placebo effect
LATENT_DIM = 8        # Dimension of encoded representation
BATCH_SIZE = 2 ** 14  # big batch_size makes big changes o_o
EPOCHS = 2 ** 3       # train loops will be EPOCHS * ERAS adjust them to print
ERAS = 2 ** 10        # more or less text
LEARNING_RATE = 0.00001  # I tryed to move this, I think this is the sweet spot

HIDEN_NEURONS_PER_LAYER = 32

# Input type and value range
"""
NOTE: we can't use uint32 beacuse float32 tremples with small values, so we need
a space to represent our data without loosing data back. The IEEE 754 float:
 31   30-23    22-0
|sing|exponent|fraction|
so a sort can fit inside the 23bit
"""
dtype = torch.uint16  # 16-bit unsigned integers
bytes_n = 2           # 2 bytes (16-bit integers)
N = (1 << 16)         # Number of unique byte values
max_int = N - 1       # max_int is now 65535 for uint16


# net #########################################################################
class RoundActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Pass-through gradient (non-differentiable)


round_activation = RoundActivation.apply


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(input_dim, input_dim),
            # nn.ReLU(),
            # nn.Linear(input_dim, latent_dim),
            # nn.Linear(input_dim, HIDEN_NEURONS_PER_LAYER),
            # nn.ReLU(),
            # nn.Linear(HIDEN_NEURONS_PER_LAYER, HIDEN_NEURONS_PER_LAYER),
            # nn.ReLU(),
            # nn.Linear(HIDEN_NEURONS_PER_LAYER, HIDEN_NEURONS_PER_LAYER),
            # nn.ReLU(),
            # nn.Linear(HIDEN_NEURONS_PER_LAYER, HIDEN_NEURONS_PER_LAYER),
            # nn.ReLU(),
            # nn.Linear(HIDEN_NEURONS_PER_LAYER, latent_dim),
            # nn.Linear(input_dim, HIDEN_NEURONS_PER_LAYER), # really bad
            # nn.BatchNorm1d(HIDEN_NEURONS_PER_LAYER),
            # nn.LeakyReLU(0.01),
            # nn.Linear(HIDEN_NEURONS_PER_LAYER, HIDEN_NEURONS_PER_LAYER),
            # nn.BatchNorm1d(HIDEN_NEURONS_PER_LAYER),
            # nn.LeakyReLU(0.01),
            # nn.Linear(HIDEN_NEURONS_PER_LAYER, latent_dim),
            # nn.GELU(),
            # nn.Linear(HIDEN_NEURONS_PER_LAYER, HIDEN_NEURONS_PER_LAYER),
            nn.Linear(input_dim, HIDEN_NEURONS_PER_LAYER),
            torch.nn.SiLU(),
            nn.Linear(HIDEN_NEURONS_PER_LAYER, HIDEN_NEURONS_PER_LAYER),
            torch.nn.SiLU(),
            nn.Linear(HIDEN_NEURONS_PER_LAYER, latent_dim),
        )

    def forward(self, x):
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, HIDEN_NEURONS_PER_LAYER),
            torch.nn.SiLU(),
            nn.Linear(HIDEN_NEURONS_PER_LAYER, HIDEN_NEURONS_PER_LAYER),
            torch.nn.SiLU(),
            nn.Linear(HIDEN_NEURONS_PER_LAYER, output_dim),
        )

    def forward(self, x):
        x = self.fc(x)
        return round_activation(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

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

    dataset_tensor = torch.tensor(dataset, dtype=torch.float32)

    return dataset_tensor


# TODO: chage this function to just split the offset num, and then join the
#       tables (I think is going to be faster)
def sample_slice_dataset_py(offset, batch_size, key_length):
    """
    Generate a batch of vectors for large N-bit numbers using Python's int.

    :param offset: The starting value.
    :param batch_size: The number of vectors to generate.
    :param key_length: The number of 16-bit chunks in each vector.
    :return: A numpy array (or torch tensor).
    """
    chunks = []
    for num in range(offset, offset + batch_size):
        # Split the number into 16-bit chunks
        vect = []
        for _ in range(key_length):
            vect.append(num & 0xffff)
            num >>= 16
        chunks.append(vect[::-1])  # Reverse to match high-to-low order

    # Convert to numpy array or torch tensor
    chunks_np = np.array(chunks, dtype=np.uint16)
    return torch.tensor(chunks_np, dtype=torch.float32)


def sample_slice_dataset_np(offset, batch_size, key_length):
    """
    Generate a batch with the vectors starting from offset.

    :param offset: The starting value to generate the dataset.
    :param batch_size: The number of vectors to generate.
    :param key_length: The number of 16-bit chunks each vector will have.
    :return: A tensor containing the dataset.
    """
    numbers = np.arange(offset, offset + batch_size, dtype=np.ulonglong)
    chunks = np.zeros((batch_size, key_length), dtype=np.uint16)

    for i in range(key_length):
        chunks[:, i] = numbers & 0xffff
        numbers >>= 16

    chunks = np.flip(chunks, axis=1).copy()

    dataset_tensor = torch.tensor(chunks, dtype=torch.float32)

    return dataset_tensor


def train_model(key_length):
    batch_size = BATCH_SIZE
    latent_dim = LATENT_DIM
    epochs = EPOCHS
    eras = ERAS
    P = N ** key_length  # total_permutations
    M = eras * epochs * batch_size
    sorted_batches = [(offset, min(batch_size, P - offset))
                      for offset in range(0, P, batch_size)]
    # batches = sorted_batches.copy() # for full range

    print("training model:")
    print(f"    key_length:     {key_length} (input vect dim (decoded))")
    print(f"    key_size:       {bytes_n}B (each vect value size)")
    print(f"    latent_dim:     {LATENT_DIM}x32 (out vect dim (encoded))")
    print(f"    learning_rate:  {LEARNING_RATE}")
    print(f"    epochs:         {log2(EPOCHS):02d} = log2({EPOCHS})")
    print(f"    eras:           {log2(ERAS):02d} = log2({ERAS})")
    print(f"    batch_size:     {log2(BATCH_SIZE):02d} = log2({BATCH_SIZE})")
    print(f"    N (full range): {log2(P):02d} = log2({P})")
    print(f"    M (per cycle):  {log2(M):02d} = log2({M})")

    if key_length <= 4:
        sample_slice_dataset = sample_slice_dataset_np
    else:
        sample_slice_dataset = sample_slice_dataset_py

    model = Autoencoder(key_length, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    overall_start_time = time.time()
    while True:
        print("starting a new training cycle...")
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            epoch_start_time = time.time()
            '''
            random and full range work:
              key_length: 1
                random: 310.53s
                full:   365.95
            '''
            for _ in range(eras):
                train_batch = sample_random_dataset(batch_size, key_length)
            # np.random.shuffle(batches)  # Shuffle batch order
            # for offset, b_size in batches:
            #     train_batch = sample_slice_dataset(offset,
            #                                        b_size,
            #                                        key_length)
            #
            #     random.shuffle(train_batch)
                # Forward pass
                optimizer.zero_grad()
                outputs = model(train_batch)
                loss = criterion(outputs, train_batch)
                # total_loss += loss.item() * b_size  # Scale by batch size
                total_loss += loss.item() * batch_size  # Scale by batch size

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Record epoch duration
            epoch_duration = time.time() - epoch_start_time
            avg_loss = total_loss / P
            print(f"    epoch [{epoch}/{epochs}], avg_loss: {avg_loss:.6f}, "
                  f"time: {epoch_duration:.2f}s")
            if total_loss == 0.0:
                print("    -> total_loss == 0.0f, testing")
                break

        correct = 0
        print(
            "|- testing full range -----------------------------------------|"
            "\n ", end="")

        with torch.no_grad():
            step = P // 64
            next = step
            for offset, b_size in sorted_batches:
                if offset >= next:
                    print("=", end="", flush=True)
                    next += step
                test_batch = sample_slice_dataset(offset,
                                                  b_size,
                                                  key_length)
                # predicts
                outputs = model(test_batch)

                c = torch.sum(torch.all(
                    torch.round(outputs) == test_batch, dim=1
                )).item()
                correct += c

                if c != b_size:
                    print(f"\n    need more trainning; total: {
                        P}; correct: {correct}/{
                        offset + b_size};")
                    break

        if correct == P:
            print("\nPerfect accuracy achieved. Stopping training.")
            break

        # epochs = 4 if epochs <= 4 else epochs // 2
        # eras = 64 if eras <= 64 else eras // 2
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


def test_autoencoder_full_range(model, key_length, batch_size=10):
    """
    Tests the autoencoder model on all possible values from 0x0000 to 0xFFFF.
    Prints the original and decoded keys.

    :param model: The trained autoencoder model.
    :param batch_size: The number of test samples to process at once.
    :param key_length: The length of each key in the dataset.
    """
    print("=" * 50)
    print("Testing the model on all values from 0x0000 to 0xFFFF...")

    total_permutations = N ** key_length
    batches = [(offset, min(batch_size, total_permutations - offset))
               for offset in range(0, total_permutations, batch_size)]

    if key_length <= 4:
        sample_slice_dataset = sample_slice_dataset_np
    else:
        sample_slice_dataset = sample_slice_dataset_py
    total_permutations = N ** key_length

    with torch.no_grad():  # Disable gradient computation for inference
        correct = 0
        for offset, b_size in batches:
            test_batch = sample_slice_dataset(offset, b_size, key_length)

            # Encode and decode the batch
            encoded = model.encoder(test_batch)  # Add batch dimension
            decoded = model.decoder(encoded)
            correct += torch.sum(torch.all(
                torch.round(decoded) == test_batch, dim=1
            )).item()
        accuracy = correct / total_permutations * 100
        print(f"    validation acc: {accuracy:.2f}%")
    print("=" * 50)


# main ########################################################################
if __name__ == "__main__":
    key_length = 2
    model = train_model(key_length)
    test_autoencoder(model, key_length)
    test_autoencoder_full_range(model, key_length)

