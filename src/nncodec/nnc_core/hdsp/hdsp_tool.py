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

from collections import OrderedDict
import numpy as np

from nncodec.extensions.deepCABAC import HdspMode as HdspMode
def HDSP_OPTS_OFF():
    return [HdspMode.AlwaysOff, np.array((0, 0), dtype=np.int8)]


class HdspTool( ):
    def __init__(self, hdsp_enabled ):
        self.hdsp_enabled   = hdsp_enabled
        self.best_mode_idx = dict()
        self.data_hist     = OrderedDict()
        self.empty_pos     = np.array((0, 0), dtype=np.int8)

    def enabled(self):
        return self.hdsp_enabled

    def has_hist(self, param_name ):
        return param_name in self.data_hist.keys()

    def get_num_modes( self, params ):
        all_have_hist = all( [ self.has_hist( param_name )  for param_name in params ]  )
        return 2 if ( self.enabled() and all_have_hist ) else 1

    def get_opts(self, param_name, inst, mode= None ):
        if not self.enabled() or inst == 'a':
            opts = [ HdspMode.AlwaysOff, self.empty_pos ]
        else:
            if inst in [ 'e', 'd', 'de']:
                has_hist = self.has_hist(param_name)
                if mode is None:
                    if has_hist:
                        if inst == "d": # For the decoder, best_mode will only be checked vs. the value in the bitstream
                            opts = [ HdspMode.TensorOn, self.data_hist[param_name] ]
                        else:
                            assert self.best_mode_idx[param_name] in [ 0, 1 ], "Invalid best mode"
                            best_mode = HdspMode.TensorOff if self.best_mode_idx[param_name] == 0 else HdspMode.TensorOn
                            opts = [ best_mode , self.data_hist[param_name] ]
                    else:
                        opts = [ HdspMode.TensorOff, self.empty_pos ]
                elif mode == 1 and has_hist:
                    opts =  [ HdspMode.TensorOn, self.data_hist[param_name] ]
                elif mode == 0:
                    opts = [ HdspMode.TensorOff, self.empty_pos ]
                else:
                    assert False, "Unknown mode" + str(mode)
            else:
                assert False, "Unknown inst" + inst
        return opts

    def get_best_bit_stream_and_set_mode(self, param_names, bit_streams):
        lens = [ len(x) for x in bit_streams ]

        if False:
            print( "Bytes of Modes: " + str( lens ) )

        # First is off
        best_idx = np.argmin( lens )
        for param_name in param_names:
            self.best_mode_idx[param_name] = best_idx
        return bit_streams[best_idx]


    def add_data_to_hist(self, enc_diff_rec_approx_data):
        if self.enabled():
            for k, v in enc_diff_rec_approx_data.items():
                if not self.has_hist( k ):
                    self.data_hist[k] = np.zeros( v.shape, dtype=np.int8  )
                self.data_hist[k][ v != 0 ] = 1

    def print_max_diff(self, name, dict1, dict2, asssert_on_mismatch):
        max_diff = 0.0
        for k in dict1.keys():
            cur_diff = np.abs(dict1[k] - dict2[k]).max()
            if cur_diff != 0:
                print(k + ": " + str(cur_diff))
            max_diff = max(max_diff, cur_diff )

        if not asssert_on_mismatch:
            print(name + ": " + str(max_diff))
        else:
            assert max_diff == 0.0, name + " mismatch"
