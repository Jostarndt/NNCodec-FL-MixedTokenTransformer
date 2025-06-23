/* -----------------------------------------------------------------------------
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


------------------------------------------------------------------------------------------- */
#ifndef __COMMON__
#define __COMMON__

#include "TypeDef.h"
namespace py = pybind11;
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using HdspDataType  = int8_t;
using HdspPyAryType = py::array_t<HdspDataType, py::array::c_style>;

class HdspOpts
{
private: 
  HdspMode       m_mode;
  HdspDataType*  m_histData;
  uint32_t       m_numWeights;
public:
  HdspOpts( HdspMode mode, const HdspPyAryType& pyAry )
  {
    m_mode       = mode;
    m_histData   = (HdspDataType*) pyAry.request().ptr;
    m_numWeights = pyAry.request().size;
  }

  bool hdspFlagPresent() const
  {
    return m_mode != HdspMode::AlwaysOff;
  }

  bool hdspEnabled() const
  {
    return (m_mode != HdspMode::TensorOff && m_mode != HdspMode::AlwaysOff );
  }

  bool getEnabledAt(uint32_t pos) const
  {
    return m_mode == HdspMode::TensorOn && (m_histData[pos] == 1);
  }
};
#endif





