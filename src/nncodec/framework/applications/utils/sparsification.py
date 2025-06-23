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
from collections import OrderedDict
from nncodec.nnc_core import common
import numpy as np

def get_sparsity(net):
    num_zeros = 0
    num_params = 0
    for param in net:
        num_zeros += net[param][net[param] == 0].size
        num_params += net[param].size
    return num_zeros / num_params

def filter_sparsific_v2(model_param_diff, p):
    diffs = copy.deepcopy(model_param_diff)
    sparsified_channels = OrderedDict()

    for param in diffs:
        W = diffs[param]
        if len(W.shape) < 2 or not W.any() or '.weight_scaling' in param:
            sparsified_channels[param] = W
        else:
            filter_mean = []
            for filter in range(len(W)):
                filter_mean.append(np.mean(abs(W[filter])))

            filter_mean = np.array(filter_mean) / abs(np.array(filter_mean)).max()
            sort = np.argsort(np.array(filter_mean))

            if isinstance(p, OrderedDict) or isinstance(p, dict):
                magnitude_poor_filters = np.sort(sort[:int(p[param] * len(sort))])
            else:
                magnitude_poor_filters = np.sort(sort[:int(p * len(sort))])

            for filter_idx in range(len(W)):
                if filter_idx in magnitude_poor_filters:
                    W[filter_idx] *= 0

            sparsified_channels[param] = W
    return sparsified_channels

def get_filter_percentages(model_param_diff, gain=0.9):
    diffs = copy.deepcopy(model_param_diff)
    percentage_below_mean = OrderedDict()
    for param in diffs:
        W = diffs[param]
        if len(W.shape) > 1 and W.any() and '.weight_scaling' not in param:
            filter_mean = []
            for filter in range(len(W)):
                filter_mean.append(np.mean(abs(W[filter])))
            percentage_below_mean[param] = len(np.array(filter_mean)[np.array(filter_mean) < np.mean(filter_mean) * gain]) / len(W)
    return percentage_below_mean

def apply_struct_spars_v2(model_diff, gain=0.9, filter_sparsity=0.0):
    model_diff_cp = copy.deepcopy(model_diff)
    if filter_sparsity > 0: # for exact pruning rate per layer
        sparse_filters_model_diffs, _, _, _ = filter_sparsific_v2(model_diff_cp, p=filter_sparsity)
        return sparse_filters_model_diffs

    percentage_below_mean = get_filter_percentages(model_diff_cp, gain=gain)
    print(f"Structured sparsification: {percentage_below_mean}")
    return filter_sparsific_v2(model_diff_cp, p=percentage_below_mean)

def stats_based_sparsific_v2(model_param_diff, delta=1.0, step=None, num_sparser_params=0, qp_induced_sparsity=False):
    diffs = copy.deepcopy(model_param_diff)
    sparse_diffs = OrderedDict()
    sparsity_log = OrderedDict()
    for param in diffs:
        W = diffs[param]
        if len(W.shape) < 2 or not W.any() or '.weight_scaling' in param:
            sparse_diffs[param] = W
        else:
            mu = np.mean(W)
            std = np.std(W)
            threshold = np.max([abs(mu - (delta * std)), abs(mu + (delta * std))])

            if isinstance(step, dict):
                if qp_induced_sparsity or threshold < step[param] / 2:
                    threshold = step[param] / 2
                else:
                    num_sparser_params += 1
            else:
                if qp_induced_sparsity or threshold < step / 2:
                    threshold = step / 2
                else:
                    num_sparser_params += 1

            sparse_diffs[param] = (W * (W >= threshold)) + (W * (W <= -threshold))
            sparsity_log[param] = sparse_diffs[param][sparse_diffs[param] == 0].size / W.size
    return sparse_diffs, sparsity_log, num_sparser_params

def achieve_target_sparsity(target_sparsity, model_param_diff, step_size, d):
    sparse_model_diffs = copy.deepcopy(model_param_diff)
    inc = 0.2
    for i in range(501):  # 500 iterations max
        sparsity = get_sparsity(sparse_model_diffs)
        if sparsity >= target_sparsity:
            break
        if target_sparsity - sparsity < 0.05:
            inc = 0.01
        elif target_sparsity - sparsity < 0.10:
            inc = 0.05
        elif target_sparsity - sparsity < 0.30:
            inc = 0.10
        d += inc
        sparse_model_diffs, spars_log, _ = stats_based_sparsific_v2(model_param_diff, delta=d, step=step_size)
    return sparse_model_diffs

def apply_unstruct_spars_v2(qp, model_param_diff, target_sparsity=0.0, qp_density=2):
    step_size = common.get_stepsize_from_qp(qp, qp_density)
    sparse_model_diffs, spars_log, _ = stats_based_sparsific_v2(model_param_diff, step=step_size, qp_induced_sparsity=True)
    qp_sparsity = get_sparsity(sparse_model_diffs)
    print(f"QP-induced Sparsity: {qp_sparsity*100:.2f}%")
    if target_sparsity > 0.0 and qp_sparsity < target_sparsity:
        sparse_model_diffs = achieve_target_sparsity(target_sparsity, sparse_model_diffs, step_size, d=0.25)
    return sparse_model_diffs