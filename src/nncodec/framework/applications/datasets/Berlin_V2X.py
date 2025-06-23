'''
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or
other Intellectual Property Rights other than the copyrights concerning
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2019-2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''
import os
import glob
import random
import torch
import numpy as np

SEED_RANDOM = 909
SEED_TORCH = 808

def seed_worker(worker_id):
    seed = SEED_RANDOM + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

g = torch.Generator()
g.manual_seed(SEED_TORCH)


class PretokDatasetTelko(torch.utils.data.Dataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, max_seq_len, bin_path, split='train', shuffle=False):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.bin_path = bin_path
        self.split = split
        self.shuffle = shuffle

        m = np.memmap(self.bin_path, dtype=np.uint8, mode="r")

        start_indices = np.where(m == 17)[0]
        if start_indices[0] != 0:
            start_indices = np.insert(start_indices, 0, 0)
        end_indices = np.append(start_indices[1:], len(m))
        self.sequences = [m[start:end] for start, end in zip(start_indices, end_indices)]

        self.num_samples = len(self.sequences)

        # Optional shuffling for training
        if self.shuffle:
            np.random.seed(42)  # or pass a seed argument
            np.random.shuffle(self.sequences)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Pad or truncate
        if len(seq) >= self.max_seq_len:
            padded = seq[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(seq)
            padding = np.zeros(pad_len, dtype=seq.dtype)
            padded = np.concatenate((seq, padding))

        chunk = torch.from_numpy(padded.astype(np.int32))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class PretokDatasetTelkoTest(torch.utils.data.Dataset):
    """Loads pretokenized examples from a .npz file and returns them as PyTorch tensors."""

    def __init__(self, max_seq_len, bin_path, split='train', shuffle=False):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.bin_path = bin_path
        self.split = split
        self.shuffle = shuffle

        m = np.memmap(self.bin_path, dtype=np.uint8, mode="r")

        start_indices = np.where(m == 17)[0]
        if start_indices[0] != 0:
            start_indices = np.insert(start_indices, 0, 0)
        end_indices = np.append(start_indices[1:], len(m))
        self.sequences = [m[start:end] for start, end in zip(start_indices, end_indices)]
        max_length = max(len(seq) for seq in self.sequences)
        print(f"Maximum sequence length: {max_length}")

        self.num_samples = len(self.sequences)

        # Optional shuffling for training
        if self.shuffle:
            np.random.seed(42)  # or pass a seed argument
            np.random.shuffle(self.sequences)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Pad or truncate
        if len(seq) >= self.max_seq_len:
            padded = seq[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(seq)
            padding = np.zeros(pad_len, dtype=seq.dtype)
            padded = np.concatenate((seq, padding))

        chunk = torch.from_numpy(padded.astype(np.int32))
        return chunk


def V2X(args, test_only=False, shuffle=False):

    if not test_only:

        print(f'searching: {os.path.join(args.dataset_path, "train")} and {os.path.join(args.dataset_path, "test")}')
        train_shard_filenames = sorted(glob.glob(os.path.join(os.path.join(args.dataset_path, 'train'), "*.bin")))

        test_shard_filename = sorted(glob.glob(os.path.join(os.path.join(args.dataset_path, 'test'), "*.bin")))

        if not len(train_shard_filenames) == args.num_clients:
            print(f"train_shard_filenames: {train_shard_filenames}")
            print(f"test_shard_filename: {test_shard_filename}")
            assert 0, f'number of pretokenized train files {len(train_shard_filenames)} does not match number of clients {args.num_clients}'

        train_loaders, val_loaders = [], []
        for shard in train_shard_filenames:
            ds = PretokDatasetTelko(args.max_seq_len, shard)
            dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers,
                                             worker_init_fn=seed_worker, generator=g)
            train_loaders.append(dl)

            dlt = torch.utils.data.DataLoader(PretokDatasetTelko(args.max_seq_len, test_shard_filename[0], split='test'),
                                              batch_size=args.batch_size, pin_memory=True, num_workers=args.workers,
                                              worker_init_fn=seed_worker, generator=g)
            val_loaders.append(dlt)

        test_loader = torch.utils.data.DataLoader(PretokDatasetTelko(args.max_seq_len, test_shard_filename[0], split='test'),
                                                  batch_size=args.batch_size, pin_memory=True, num_workers=args.workers,
                                                  worker_init_fn=seed_worker, generator=g)
        return train_loaders, val_loaders, test_loader  ## Note: curently test loader and client val loaders are identical

    else:

        print(f'searching: {os.path.join(args.dataset_path, "test")}')
        test_shard_filename = sorted(glob.glob(os.path.join(os.path.join(args.dataset_path, 'test'), "*.bin")))

        test_loader = torch.utils.data.DataLoader(PretokDatasetTelkoTest(args.max_seq_len, test_shard_filename[0], split='test', shuffle=shuffle),
                                                  batch_size=args.batch_size, pin_memory=True, num_workers=0,
                                                  worker_init_fn=seed_worker, generator=g)
        return test_loader