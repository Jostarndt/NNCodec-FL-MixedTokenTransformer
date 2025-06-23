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
import copy
import os
import torch
import torchvision
import numpy as np
import wandb
from collections import OrderedDict
import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

def model_add(state_dict_a, state_dict_b): # full, partial
    state_dict_sum = OrderedDict()
    for param_name in state_dict_a:
        if param_name in state_dict_b:
            state_dict_sum[param_name] = state_dict_a[param_name] + state_dict_b[param_name]
        else:
            state_dict_sum[param_name] = state_dict_a[param_name]
    return state_dict_sum


class NNCFedAvg(fl.server.strategy.FedAvg):

    def __init__(self, min_fit_clients, min_available_clients, evaluate_fn, accept_failures,
                 fit_metrics_aggregation_fn, initial_parameters, model_arch, mdl_info,  args, encode_fn, decode_fn):

        super().__init__() # Call the constructor of the original class
        self.id = 0
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.accumulated_bs_size = 0
        self.accumulated_uncompressed = 0
        self.current_bs_size = 0
        self.model_arch = model_arch
        self.mdl_info = mdl_info
        self.bytes_mdl_full_prec = sum([v.numel( ) * 4 for v in self.model_arch.state_dict().values() if v.shape != torch.Size([])])
        self.args = args
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.base_mdl = OrderedDict({k: np.float32(v.cpu().detach().numpy()) for k, v in self.model_arch.state_dict().items()
                                    if v.shape != torch.Size([])})
        if self.args.bnf:
            self.base_mdl = self.fold_bn(self.base_mdl)
        self.expected_keys = self.mdl_info["parameter_index"] if not self.args.bnf else mdl_info["bnf_map"]
        self.internal_states = {}
        self.internal_states = {"approx_param_base": {"parameters": {},
                                                      "put_node_depth": {},
                                                      "device_id": self.id,
                                                      "parameter_id": {},
                                                      },
                                "best_loss": 1e9}

        self.previously_encoded_params = None

        print("SERVER INITIALIZED") ## server only once, clients every time they're called

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""

        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        if self.args.compress_downstream and len(parameters.tensors) == 1: # NNC-encoded
            parameters_ndarrays = self.decode_fn(bytearray(parameters.tensors[0][parameters.tensors[0].index(b'\n') + 1:]),
                                                 internal_states_path=f"{self.args.results}")
            self.save_internal_states()

            if self.args.compress_differences: ## TODO: replace w/ checking mps_parent_signalling_enabled_flag
                parameters_ndarrays = model_add(self.base_mdl, parameters_ndarrays)
                self.base_mdl = copy.deepcopy(parameters_ndarrays)

            if self.args.bnf:
                parameters_ndarrays = self.unfold_bn(parameters_ndarrays)

            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in parameters_ndarrays.items()})

        else: # not NNC-encoded
            parameters_ndarrays = parameters_to_ndarrays(parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in zip(self.mdl_info["parameter_index"], parameters_ndarrays)})

        test_results_dict = self.evaluate_fn(server_round, state_dict)

        if self.args.wandb:
            print("wandb: log accumulated_bs_size and global test performance")
            is_seg_mdl = self.args.model in torchvision.models.list_models(torchvision.models.segmentation)
            wandb_dict = {"accumulated_bs_size": self.accumulated_bs_size,
                          "accumulated_uncompressed_size": self.accumulated_uncompressed,
                          "running_CR": self.accumulated_bs_size / self.accumulated_uncompressed if self.accumulated_uncompressed else 0,
                          "current_bs_size": self.current_bs_size,
                          "current_CR": self.current_bs_size / (self.bytes_mdl_full_prec * self.min_available_clients * 2),
                          }
            wandb_dict.update(test_results_dict)
            wandb.log(wandb_dict)

        mean_loss = torch.mean(torch.tensor(list(test_results_dict.values())))
        if mean_loss <= self.internal_states["best_loss"]:
           self.internal_states["best_loss"] = mean_loss
           print("save new best model")
           torch.save(state_dict, os.path.join(self.args.results, f'best_{self.args.model}_{self.args.job_id}.pt'))

        return mean_loss, test_results_dict


    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if os.path.exists(f"{self.args.results}/client_ID{self.id}_internal_states.npz"):
            self.load_internal_states()

        if self.previously_encoded_params:

            _ = self.decode_fn(self.previously_encoded_params,
                                           approx_param_base=self.internal_states["approx_param_base"],
                                           update_base_param=True)

        # Convert results
        if self.args.compress_upstream:
            decompressed_weights = [(self.decode_fn(bytearray(bs[bs.index(b'\n') + 1:]),
                                                    internal_states_path=f"{self.args.results}",
                                                    update_base_param=True), ne)
                                                    for bs, ne in [(fitres.parameters.tensors[0], fitres.num_examples)
                                                        for _, fitres in results]]

            # If any key of the expected keys is missing, it is inserted with a zero valued ndarray
            for tensors, num_samples in decompressed_weights:
                if len(tensors) != len(self.expected_keys):
                    for k in self.expected_keys.keys():
                        tensors.setdefault(k, np.zeros_like(self.base_mdl[k]))

            # sort all client updates in the same order (as in self.expected_keys)
            decompressed_weights = [(dict(sorted(tensors.items(), key=lambda x: list(self.expected_keys.keys()).index(x[0]))), num_samples)
                                    for tensors, num_samples in decompressed_weights]

            weights_results = [([v for _, v in w.items() if v.shape != ()], ne) for w, ne in decompressed_weights]

        else:
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]

        # Aggregate
        parameters_aggregated = aggregate(weights_results)

        if self.args.compress_downstream:
            print("DOWN-STREAM compression:")
            agg_mdl_state_dict = OrderedDict({k: np.float32(v) for k, v in zip(self.expected_keys, parameters_aggregated)})
            
            if self.args.err_accumulation and "residuals" in self.internal_states:
                agg_mdl_state_dict = model_add(agg_mdl_state_dict, self.internal_states["residuals"])

            parameters_aggregated = [self.encode_fn(agg_mdl_state_dict, vars(self.args), approx_param_base=self.internal_states["approx_param_base"],
                                                    device_id=self.id)]
            self.accumulated_bs_size += (self.min_fit_clients * len(parameters_aggregated[0]))
            current_bs_size_server = self.min_fit_clients * len(parameters_aggregated[0])

            self.previously_encoded_params = parameters_aggregated[0]
            
            if self.args.err_accumulation:
                self.update_residual(compressed_update=self.decode_fn(parameters_aggregated[0], approx_param_base=self.internal_states["approx_param_base"]),
                                     uncompressed_update=agg_mdl_state_dict)
        else:
            self.accumulated_bs_size += (self.min_fit_clients * self.bytes_mdl_full_prec)
            current_bs_size_server = self.min_fit_clients * self.bytes_mdl_full_prec

        self.save_internal_states()

        parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)

        self.accumulated_uncompressed += self.bytes_mdl_full_prec * self.min_available_clients * 2 # up and down link

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            self.accumulated_bs_size += metrics_aggregated["accumulated_bs_sizes"]
            self.current_bs_size = current_bs_size_server + metrics_aggregated["accumulated_bs_sizes"]
            print(f"accumulated_bs_size: {self.accumulated_bs_size}")
        return parameters_aggregated, metrics_aggregated
    
    def update_residual(self, compressed_update, uncompressed_update):
        if not "residuals" in self.internal_states:
            self.internal_states["residuals"] = {}
        for k in uncompressed_update:
            ignored_param = False #k.endswith(".weight_scaling")
            if k in compressed_update and not ignored_param:
                self.internal_states["residuals"][k] = uncompressed_update[k] - compressed_update[k]
            else:
                self.internal_states["residuals"][k] = np.zeros_like(uncompressed_update[k]) \
                                                        if ignored_param else uncompressed_update[k]

    def fold_bn(self, param_dict):
        eps = 1e-3 if self.mdl_info['topology_storage_format'] == 4 else 1e-5

        for blk in self.mdl_info["mdl_blocks"]:
            block_access = self.mdl_info["mdl_blocks"][blk]
            block_id = block_access["block_id"]
            if block_id is None:
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
    def unfold_bn(self, folded_state_dict):
        mdl_arch = {n: v.detach().cpu().numpy() for n, v in self.model_arch.state_dict().items()}
        unfolded_state_dict = {}

        for blk in self.mdl_info["mdl_blocks"]:
            block_access = self.mdl_info["mdl_blocks"][blk]
            if block_access["block_id"] is None:
                continue

            if "w" in block_access and block_access["w"] in mdl_arch:
                unfolded_state_dict[block_access["w"]] = folded_state_dict[block_access["w"]]
            if "bi" in block_access and block_access["bi"] in mdl_arch:
                unfolded_state_dict[block_access["bi"]] = folded_state_dict[block_access["bi"]]
            if "bn_mean" in block_access and block_access["bn_mean"] in mdl_arch:
                bn_shape = mdl_arch[block_access["bn_mean"]].shape
                unfolded_state_dict[block_access["bn_mean"]] = np.zeros(bn_shape, dtype=np.float32)
                unfolded_state_dict[block_access["bn_var"]] = np.ones(bn_shape, dtype=np.float32)
                unfolded_state_dict[block_access["bn_beta"]] = folded_state_dict[block_access["bi"]]
                unfolded_state_dict[block_access["bn_gamma"]] = folded_state_dict[block_access["ls"]]
        return unfolded_state_dict

    def save_internal_states(self):
        np.savez(f"{self.args.results}/client_ID{self.id}_internal_states.npz", **self.internal_states)


    def load_internal_states(self):
        loaded_states = np.load(f"{self.args.results}/client_ID{self.id}_internal_states.npz",
                                allow_pickle=True)  # TODO get rid of allow_pickle
        self.internal_states = {k: loaded_states[k].item() for k in loaded_states.files}
        loaded_states.close()
