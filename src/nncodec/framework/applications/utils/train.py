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
import logging
import torch
import time
import random
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext
from nncodec.nnc_core import nnr_model
import math

LOGGER = logging.getLogger(__name__)

def freeze_batch_norm_layers(model):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()

def train_classification_model(model, optimizer=None, criterion=None, trainloader=None, device=None, verbose=True,
                               max_batches=None, freeze_batch_norm=False, return_model=False, args=None, round=0):
    """
    Parameters
    ----------
    device: torch.device
        Choose between cuda or cpu.
    model: torch.nn.Module
        A pytorch network model.
    optimizer: torch.optim.Optimizer
        A pytorch optimizer like Adam.
    criterion: torch.nn.Loss
        A pytorch criterion that defines the loss.
    trainloader: torch.utils.data.DataLoader
        Loader of train data.
    max_batches: int
        How many batches the model should train for.
    verbose: bool
        If True, print text - verbose mode.
    freeze_batch_norm: bool
        If True set batch norm layers to eval. Default: False

    Returns
    -------
    success: bool
        Returns False is nans encountered in the loss else True.
    """

    if isinstance(model, nnr_model.ModelExecute):
        optimizer = model.optimizer
        criterion = model.handle.criterion
        trainloader = model.test_loader
        device = model.device
        model = model.model

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    model.to(device)
    model.train()
    if freeze_batch_norm:
        freeze_batch_norm_layers(model)

    train_loss = []
    correct = 0
    total = 0

    if args is not None and args.max_batches == None:
        max_batches = len(trainloader)
    elif args is not None:
        max_batches = args.max_batches


    total_iterations = max_batches or len(trainloader)
    iterator = tqdm(enumerate(trainloader), total=total_iterations, position=0, leave=True, desc='train_classification') \
        if verbose else enumerate(trainloader)

    DeepLab_condition = model.__class__.__name__ == "DeepLabV3" if not (isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel)) \
                            else model.module.__class__.__name__ == "DeepLabV3"

    if DeepLab_condition:
        from torchmetrics import JaccardIndex
        num_of_classes = model.classifier[len(model.classifier) - 1].weight.shape[0]
        jaccard = JaccardIndex(task="multiclass", num_classes=num_of_classes, average="macro").to(device)

    for batch_idx, (inputs, targets) in iterator:

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        if DeepLab_condition:
            outputs = outputs['out']
            targets = targets * (targets != 1) * 255 # for Pascal VOC (different for, e.g., RailSem19)
            targets = targets.squeeze(1).long() # for Pascal VOC (different for, e.g., RailSem19 (use "decode_mask_RS()")

        loss = criterion(outputs, targets)

        if torch.isnan(loss):
            LOGGER.warning('--> Loss is Nan.')
            break

        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        _, predicted = outputs.max(1)

        total += targets.size(0) if not DeepLab_condition else targets.numel()
        correct += predicted.eq(targets).sum().item()

        if DeepLab_condition:
            jaccard.update(predicted, targets)

        if batch_idx == max_batches:
            break

    acc = correct * 100.0 / total
    mean_train_loss = np.mean(train_loss)

    if return_model:
        if DeepLab_condition:
            final_mIoU = jaccard.compute() * 100
            return {'acc': acc, 'final_mIoU': final_mIoU.item()}, model
        else:
            return {'acc': acc, 'mean_train_loss': mean_train_loss}, model
    else:
        return {'acc': acc, 'mean_train_loss': mean_train_loss}


def train_language_model(model, optimizer=None, criterion=None, trainloader=None, device=None, verbose=False, args=None,
                         round=0):

    torch.manual_seed(808 + round)
    np.random.seed(303 + round)
    random.seed(909 + round)

    try:
        len_trainloader = len(trainloader)
    except TypeError:
        len_trainloader = trainloader.dataset.num_samples

    if args is not None and args.max_batches is None:
        max_batches = len_trainloader
    else:
        max_batches = args.max_batches if args is not None else int(1e6)

    model.to(device)
    model.train()

    class Iterator:
        @staticmethod
        def iter_batches(dataloader, device):
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                yield x, y

    dtype = "float16"

    train_batches = []
    batch_iter = Iterator.iter_batches(trainloader, device)
    for idx, (x, y) in enumerate(batch_iter):
        if idx >= max_batches:
            break
        train_batches.append((x, y))

    print(f"[INFO] Loaded {len(train_batches)} batches for round {round}")

    iter_num = round * len(train_batches)
    log_interval = 100
    decay_lr = True
    warmup_iters = 1000
    lr_decay_iters = args.epochs * max_batches if args is not None else 10000
    min_lr = 0.0
    grad_clip = 1.0

    # Learning rate scheduler
    def get_lr(it):
        if it < warmup_iters:
            return args.lr * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.lr - min_lr)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(enabled=(dtype == "float16" and torch.cuda.is_available()))

    train_loss = []
    train_acc = []
    t0 = time.time()
    ctx = nullcontext()

    for X, Y in train_batches:
        with ctx:
            lr = get_lr(iter_num) if decay_lr else args.lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            logits = model(X, Y)
            loss = model.last_loss
            scaler.scale(loss).backward()
            train_loss.append(loss.item())

            predictions = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_length)
            correct = (predictions == Y).float()
            train_acc.append(correct.mean().item() * 100)

            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

        if iter_num % log_interval == 0:
            dt = time.time() - t0
            print(f"[Round {round}] {iter_num} | "
                  f"loss_cls {loss.item():.4f} | "
                  f"lr {lr:.6f} | {dt * 1000:.2f}ms")
            t0 = time.time()

        iter_num += 1


    return {'mean_train_acc': np.mean(train_acc), 'mean_train_loss': np.mean(train_loss)}
