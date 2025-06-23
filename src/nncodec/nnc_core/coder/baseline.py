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

from nncodec.nnc_core import hls
import numpy as np
from nncodec.extensions.deepCABAC import HdspMode

from nncodec.nnc_core.hdsp.hdsp_tool import HDSP_OPTS_OFF

def encode(encoder, approx_data, approx_param_base, param, ndu, mps, general_profile_idc, param_opt_flag, rowSkipFlag, tool_if, mode, lps):
    if (
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_FLOAT) or
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK)
    ):
        quantization_parameter =  lps["lps_quantization_parameter"] if lps is not None else mps["mps_quantization_parameter"]
        qp_density             =  lps["lps_qp_density"] if lps is not None else mps["mps_qp_density"]
        encoder.iae_v( 6 + qp_density, approx_data["qp"][param] - quantization_parameter)
    
    encoder.initCtxModels( ndu["cabac_unary_length_minus1"]+1, param_opt_flag )
    if param in approx_data["scan_order"]:
        assert ndu["scan_order"] == approx_data["scan_order"][param], "All parameters of a block must use the same scan_order."
    scan_order = ndu.get("scan_order", 0)
    if approx_data["parameters"][param].ndim <= 1:
        scan_order = 0

    chan_skip_list = np.zeros( approx_data["parameters"][param].shape[0], dtype=np.int32 )

    if general_profile_idc == 1:
        if len(approx_data["parameters"][param].shape) >= 2:
            for ch in range(approx_data["parameters"][param].shape[0]):
                chan_skip_list[ch] =  1 if not np.any( approx_data["parameters"][param][ch] ) else 0
        else:
            for ch in range(approx_data["parameters"][param].shape[0]):
                chan_skip_list[ch] = 0
    else:
        rowSkipFlag = 0

    if general_profile_idc == 1:
        if approx_data["approx_method"][param] == "codebook":
            codebook_size = len(approx_data["codebooks"][param])
            codebook_zero_offset = approx_data["codebook_zero_offsets"][param]
        else:
            codebook_size = 0
            codebook_zero_offset = 0
    else:
        codebook_size = 0
        codebook_zero_offset = 0

    if tool_if:
        hdsp_opts = tool_if.get_opts(param, "e", mode)
    else:
        hdsp_opts = HDSP_OPTS_OFF()

    if ndu.get("temporal_context_modeling_flag", 0) and general_profile_idc == 1 and approx_param_base:
        assert ndu["device_id"] == approx_param_base["device_id"], "device_id of the current NDU and of the reference NDU shall be equal!"
        assert ndu["parameter_id"] == approx_param_base["parameter_id"][param], "parameter_id of the current NDU and of the reference NDU shall be equal!"
        assert ndu["put_node_depth"]-1 == approx_param_base["put_node_depth"][param], "put_node_depth-1 of the current NDU shall be equal to the put_node_depth of the reference NDU!"
        encoder.encodeLayer2(approx_data["parameters"][param], approx_param_base["parameters"][param], approx_data["dq_flag"][param], scan_order, general_profile_idc, ndu.get('parent_node_id_present_flag', 0), rowSkipFlag, chan_skip_list, *hdsp_opts, codebook_size, codebook_zero_offset)
    else:
        encoder.encodeLayer(approx_data["parameters"][param], approx_data["dq_flag"][param], scan_order, general_profile_idc, ndu.get('parent_node_id_present_flag', 0), rowSkipFlag, chan_skip_list, *hdsp_opts , codebook_size, codebook_zero_offset)


def decode( decoder, approx_data, approx_param_base, param, ndu, mps, ndu_start, tool_if, lps ):
    if (
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_FLOAT) or
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK)
    ):
        quantization_parameter        =  lps["lps_quantization_parameter"] if lps is not None else mps["mps_quantization_parameter"]
        qp_density                    =  lps["lps_qp_density"] if lps is not None else mps["mps_qp_density"]
        approx_data["qp"][param]      = np.int32(decoder.iae_v( 6 + qp_density ) + quantization_parameter)
        approx_data["dq_flag"][param] = ndu["dq_flag"]

    else:
        approx_data["dq_flag"][param] = 0

    general_profile_idc = ndu_start['general_profile_idc']
        
    decoder.initCtxModels( ndu["cabac_unary_length_minus1"]+1) ##TODO: +1 or not?
    scan_order = ndu.get("scan_order", 0)
    if approx_data["parameters"][param].ndim <= 1:
        scan_order = 0

    if general_profile_idc == 1:
        if approx_data["approx_method"][param] == "codebook":
            codebook_size = len(approx_data["codebooks"][param])
            codebook_zero_offset = approx_data["codebook_zero_offsets"][param]
        else:
            codebook_size = 0
            codebook_zero_offset = 0
    else:
        codebook_size = 0
        codebook_zero_offset = 0

    if tool_if:
        hdsp_opts = tool_if.get_opts(param, "d", None)
    else:
        hdsp_opts = HDSP_OPTS_OFF()
    if ndu.get("temporal_context_modeling_flag", 0) and general_profile_idc == 1 and approx_param_base:
        assert ndu["device_id"] == approx_param_base["device_id"], "device_id of the current NDU and of the reference NDU shall be equal!"
        assert ndu["parameter_id"] == approx_param_base["parameter_id"][param], "parameter_id of the current NDU and of the reference NDU shall be equal!"
        assert ndu["put_node_depth"]-1 == approx_param_base["put_node_depth"][param], "put_node_depth-1 of the current NDU shall be equal to the put_node_depth of the reference NDU!"
        decoder.decodeLayer2(approx_data["parameters"][param], approx_param_base["parameters"][param], approx_data["dq_flag"][param], scan_order, general_profile_idc, ndu.get('parent_node_id_present_flag', 0), *hdsp_opts, codebook_size, codebook_zero_offset)
    else:
        decoder.decodeLayer(approx_data["parameters"][param], approx_data["dq_flag"][param], scan_order, general_profile_idc, ndu.get('parent_node_id_present_flag', 0), *hdsp_opts, codebook_size, codebook_zero_offset)


def decodeAndCreateEPs( decoder, approx_data, approx_param_base, param, ndu, mps, ndu_start, tool_if, lps ):
    if (
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_FLOAT) or
        (ndu["nnr_compressed_data_unit_payload_type"] == hls.CompressedDataUnitPayloadType.NNR_PT_BLOCK)
    ):
        quantization_parameter        = lps["lps_quantization_parameter"] if lps is not None else mps["mps_quantization_parameter"]
        qp_density                    = lps["lps_qp_density"] if lps is not None else mps["mps_qp_density"]
        approx_data["qp"][param]      = np.int32(decoder.iae_v( 6 + qp_density ) + quantization_parameter)
        approx_data["dq_flag"][param] = ndu["dq_flag"]
        
    general_profile_idc = ndu_start['general_profile_idc']

    decoder.initCtxModels( ndu["cabac_unary_length_minus1"]+1 )
    scan_order = ndu.get("scan_order", 0)
    if approx_data["parameters"][param].ndim <= 1:
        scan_order = 0

    if general_profile_idc == 1:
        if approx_data["approx_method"][param] == "codebook":
            codebook_size = len(approx_data["codebooks"][param])
            codebook_zero_offset = approx_data["codebook_zero_offsets"][param]
        else:
            codebook_size = 0
            codebook_zero_offset = 0
    else:
        codebook_size = 0
        codebook_zero_offset = 0

    if tool_if:
        hdsp_opts = tool_if.get_opts(param, "de", None )
    else:
        hdsp_opts = HDSP_OPTS_OFF()

    if ndu.get("temporal_context_modeling_flag", 0) and general_profile_idc == 1 and approx_param_base:
        assert ndu["device_id"] == approx_param_base["device_id"], "device_id of the current NDU and of the reference NDU shall be equal!"
        assert ndu["parameter_id"] == approx_param_base["parameter_id"][param], "parameter_id of the current NDU and of the reference NDU shall be equal!"
        assert ndu["put_node_depth"]-1 == approx_param_base["put_node_depth"][param], "put_node_depth-1 of the current NDU shall be equal to the put_node_depth of the reference NDU!"
        entryPointArray = decoder.decodeLayerAndCreateEPs2(approx_data["parameters"][param], approx_param_base["parameters"][param], approx_data.get("dq_flag", {}).get(param, 0), scan_order, general_profile_idc, ndu.get('parent_node_id_present_flag', 0), *hdsp_opts, codebook_size, codebook_zero_offset)
    else:
        entryPointArray = decoder.decodeLayerAndCreateEPs(approx_data["parameters"][param], approx_data.get("dq_flag", {}).get(param, 0), scan_order, general_profile_idc, ndu.get('parent_node_id_present_flag', 0), *hdsp_opts, codebook_size, codebook_zero_offset)


    return entryPointArray


