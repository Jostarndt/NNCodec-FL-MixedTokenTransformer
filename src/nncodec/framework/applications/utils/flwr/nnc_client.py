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
import os.path
import random
import torch
import numpy as np
from collections import OrderedDict
import flwr as fl
from typing import Dict, Tuple
from flwr.common import (Scalar, ndarrays_to_parameters)

def model_diff(state_dict_a, state_dict_b): # new, old
    state_dict_diff = OrderedDict()
    for param_name in state_dict_a:
        state_dict_diff[param_name] = state_dict_a[param_name] - state_dict_b[param_name]
    return state_dict_diff

def model_add(state_dict_a, state_dict_b): # full, partial
    state_dict_sum = OrderedDict()
    for param_name in state_dict_a:
        if param_name in state_dict_b:
            state_dict_sum[param_name] = state_dict_a[param_name] + state_dict_b[param_name]
        else:
            state_dict_sum[param_name] = state_dict_a[param_name]
    return state_dict_sum
class NNClient(fl.client.NumPyClient):

    def __init__(self, context, trainloader, valloader, id, criterion, device, c_model, args, mdl_info,
                 encode_fn, decode_fn, train_fn) -> None:

        super().__init__()
        self.context = context
        self.trainloader = trainloader
        self.testloader = valloader
        self.model = c_model
        self.mdl_info = mdl_info
        self.criterion = criterion
        self.device = device
        try:
            self.optimizer = c_model.configure_optimizers(args.weight_decay, args.lr, (0.9, 0.95), device.type)
        except:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.accumulated_bs_sizes_per_round = 0
        self.id = id
        self.args = args
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.train_fn = train_fn
        self.internal_states = {"approx_param_base": {"parameters": {},
                                                      "put_node_depth": {},
                                                      "device_id": self.id,
                                                      "parameter_id": {},
                                                      },
                                "comm_round": 0}
    def get_parameters(self, config
                       ) -> bytearray:
        """Extract model parameters and return them as a list of numpy arrays."""

        if self.args.compress_upstream:
            print("UP-STREAM compression:")
            param_dict = {k: np.float32(v.cpu().detach().numpy()) for k, v in self.model.state_dict().items()
                          if v.shape != torch.Size([])}

            if self.args.compress_differences: ## TODO: replace w/ checking mps_parent_signalling_enabled_flag
                if self.args.bnf:
                    param_dict = self.fold_bn(param_dict)
                param_dict = model_diff(param_dict, self.internal_states["prev_mdl"])

            if self.args.err_accumulation and "residuals" in self.internal_states:
                param_dict = model_add(param_dict, self.internal_states["residuals"])

            blkid_ptypes = {"block_identifier": self.mdl_info["block_identifier"],
                            "parameter_type": self.mdl_info["parameter_type"]} \
                             if self.args.bnf and not self.args.compress_differences else None

            params = [self.encode_fn(param_dict, vars(self.args), blkid_ptypes=blkid_ptypes,
                                     approx_param_base=self.internal_states["approx_param_base"],
                                     epoch=self.internal_states["comm_round"], device_id=self.id)]
            self.accumulated_bs_sizes_per_round += len(params[0])

            if self.args.err_accumulation:
                self.update_residual(compressed_update=self.decode_fn(params[0], approx_param_base=self.internal_states["approx_param_base"]),
                                     uncompressed_update=param_dict)
        else:
            params = [v.cpu().numpy() for _, v in self.model.state_dict().items() if v.shape != torch.Size([])]

        self.internal_states["comm_round"] += 1
        self.save_internal_states()

        return params

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""

        if self.args.compress_downstream and len(parameters) == 1: # decoding
            parameters = ndarrays_to_parameters(parameters).tensors[0]
            decompressed_weights = self.decode_fn(bytearray(parameters[parameters.index(b'\n') + 1:]),
                                                  internal_states_path=f"{self.args.results}")

            if self.args.compress_differences and os.path.exists(f"{self.args.results}/client_ID{self.id}_internal_states.npz"):
                self.load_internal_states()
                decompressed_weights = model_add(self.internal_states["prev_mdl"], decompressed_weights)

            if self.args.bnf:
                decompressed_weights = self.unfold_bn(decompressed_weights)

            state_dict = OrderedDict({k: torch.tensor(v) for k, v in decompressed_weights.items()})

        else:
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(self.mdl_info["parameter_index"], parameters)})

        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config
            ) -> Tuple[bytearray, int, Dict[str, Scalar]]:
        """Train model received by the server (parameters) using the data.
           that belongs to this client. Then, send it back to the server.
        """

        if os.path.exists(f"{self.args.results}/client_ID{self.id}_internal_states.npz"):
            self.load_internal_states()


        self.set_parameters(parameters)
        self.internal_states["prev_mdl"] = {k: np.float32(v.cpu().detach().numpy())
                                            for k, v in self.model.state_dict().items()
                                            if v.shape != torch.Size([])}

        torch.manual_seed(808 + self.id + self.internal_states["comm_round"])
        np.random.seed(303 + self.id + self.internal_states["comm_round"])
        random.seed(909 + self.id + self.internal_states["comm_round"])

        train_res_dict = self.train_fn(self.model, optimizer=self.optimizer, criterion=self.criterion,
                                                    trainloader=self.trainloader, device=self.device, verbose=False,
                                                    args=self.args, round=self.internal_states["comm_round"])

        if self.args.bnf:
            self.internal_states["local_bn_params"] = {n: v.detach().cpu().numpy() for n, v in
                                                       self.model.state_dict().items() if
                                                       v.shape != torch.Size([]) and self.mdl_info["parameter_type"][n]
                                                       in ["bn.beta", "bn.gamma", "bn.mean", "bn.var"]}
            self.internal_states["prev_mdl"] = self.fold_bn(self.internal_states["prev_mdl"])

        print(f"Client ID: {self.id} {[str(k) + ': ' + str(v) for k, v in train_res_dict.items()]}")

        try:
            num_samples = len(self.trainloader)
        except:
            num_samples = self.trainloader.dataset.num_samples

        return self.get_parameters({}), num_samples, {**train_res_dict, "bs_size": self.accumulated_bs_sizes_per_round}

    def save_internal_states(self):
        np.savez(f"{self.args.results}/client_ID{self.id}_internal_states.npz", **self.internal_states)

    def load_internal_states(self):
        loaded_states = np.load(f"{self.args.results}/client_ID{self.id}_internal_states.npz",
                                allow_pickle=True)  # TODO get rid of allow_pickle
        self.internal_states = {k: loaded_states[k].item() for k in loaded_states.files}
        loaded_states.close()

    def update_residual(self, compressed_update, uncompressed_update):
        if not "residuals" in self.internal_states:
            self.internal_states["residuals"] = {}
        for k in uncompressed_update:
            ignored_param = False # k.endswith(".weight_scaling")
            if k in compressed_update and not ignored_param:
                self.internal_states["residuals"][k] = uncompressed_update[k] - compressed_update[k]
            else:
                self.internal_states["residuals"][k] = np.zeros_like(uncompressed_update[k]) \
                                                        if ignored_param else uncompressed_update[k]

    def fold_bn(self, param_dict):
        eps = 1e-3 if self.mdl_info['topology_storage_format'] == 4 else 1e-5

        for blk in self.mdl_info["mdl_blocks"]:
            block_access = self.mdl_info["mdl_blocks"][blk]
            if block_access["block_id"] is None:
                continue

            if "bn_mean" in block_access:
                bn_shape = param_dict[block_access["bn_mean"]].shape

                delta = block_access["bi"]
                if delta not in param_dict:
                    param_dict[delta] = np.zeros(bn_shape, dtype=np.float32)

                alpha = block_access["ls"]
                if alpha not in param_dict:
                    param_dict[alpha] = np.ones(bn_shape, dtype=np.float32)

                g = param_dict[block_access["bn_gamma"]] / np.sqrt(param_dict[block_access["bn_var"]] + eps)
                param_dict[alpha] *= g
                param_dict[delta] = (param_dict[delta] - param_dict[block_access["bn_mean"]]) * g + param_dict[block_access["bn_beta"]]

                del param_dict[block_access["bn_gamma"]], param_dict[block_access["bn_var"]],\
                    param_dict[block_access["bn_mean"]], param_dict[block_access["bn_beta"]]

        return param_dict

    def unfold_bn(self, folded_state_dict, momentum=0.3):
        eps = 1e-3 if self.mdl_info['topology_storage_format'] == 4 else 1e-5
        unfolded_state_dict = {}

        self.load_internal_states()
        local_last_round = self.internal_states["local_bn_params"]

        for k in self.mdl_info["parameter_index"]:

            if k in folded_state_dict: # non-BN params
                unfolded_state_dict[k] = folded_state_dict[k]

            else: # BN params
                block_id = self.mdl_info["block_identifier"][k]
                if block_id is None:
                    continue
                block_access = self.mdl_info["mdl_blocks"][block_id]

                if block_access["bn_beta"] not in unfolded_state_dict: # FedBNF
                    # BN beta update
                    unfolded_state_dict[block_access["bn_beta"]] = ((1 - momentum) * local_last_round[block_access["bn_beta"]]) +\
                                                                   (momentum * (folded_state_dict[block_access["bi"]] +
                                                                                (folded_state_dict[block_access["ls"]] *
                                                                                 local_last_round[block_access["bn_mean"]])))
                    # BN gamma update
                    unfolded_state_dict[block_access["bn_gamma"]] = ((1 - momentum) * local_last_round[block_access["bn_gamma"]]) +\
                                                                    (momentum * folded_state_dict[block_access["ls"]] *
                                                                     np.sqrt(local_last_round[block_access["bn_var"]] + eps))
                    # BN running stats
                    unfolded_state_dict[block_access["bn_mean"]] = local_last_round[block_access["bn_mean"]]
                    unfolded_state_dict[block_access["bn_var"]] = local_last_round[block_access["bn_var"]]

        return unfolded_state_dict
