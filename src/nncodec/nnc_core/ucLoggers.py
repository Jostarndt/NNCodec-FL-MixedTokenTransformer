#!/usr/bin/env python3
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

import h5py
import numpy as np
import os

class ResultLogger():
    def __init__(self, result_dir):
        self.result_dir = result_dir

    def log( self, file_name, value):
        with open( os.path.join( self.result_dir, file_name + ".txt" ), 'a+' ) as fh:
            fh.write('{}\n'.format(value))

class DataExporter():
    def __init__(self, enabled, base_name, add_names, num_iterations ):
        self.base_name = base_name
        self.add_names = add_names
        self.enabled   = enabled

        if self.enabled:
            with self.xOpenFile( base_name + "_dataInfo.hdf5" ) as h:
                h.create_dataset( "base_name"     , data=np.string_(base_name), dtype= h5py.string_dtype() )
                h.create_dataset( "num_iterations", data=num_iterations )

                a_list = [n.encode("ascii", "ignore") for n in add_names]
                h.create_dataset( "add_names",(len(a_list),1), h5py.string_dtype(), a_list )


    def export(self, add_name, epoch, data ):
        if self.enabled:
            if not add_name in self.add_names:
                print( add_name + " " + " is not a valid add_name" )
                raise

            file_name =  self.base_name + "_" + add_name + "_" + str(epoch).zfill(3) + ".hdf5"
            with self.xOpenFile(file_name) as h:
                for key, val in data.items():
                    h.create_dataset(key, data=np.array(val, dtype=val.dtype ), compression="gzip", compression_opts=9 )

    def xOpenFile( self, name ):
        return h5py.File( name ,mode="w", libver=('earliest', 'v108') )