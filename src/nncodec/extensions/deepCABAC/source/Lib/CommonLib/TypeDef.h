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
#pragma once

#include <vector>
#include <utility>
#include <sstream>
#include <cstddef>
#include <cstring>
#include <assert.h>
#include <cassert>
#include <pybind11/pybind11.h>

#define CFG_FIX_TC    1

enum class HdspMode
{  
   TensorOff  =  0,
   TensorOn   =  1,
   AlwaysOff  =  99,
};

enum class ctxIds
{
  // TBD: Use also other indices
  sigfBaEquZroNbEquZro = 0,
  sigfBaEquZroNbGrtZro = 1,
  sigfBaEquZroNbLesZro = 2,
  sigfHdsp             = 3,
  sigfBaEquOne         = 4,
  sigfBaGrtOne         = 5,
  signBaEquZroNbEquZro = 0 + 8 * 6,
  signBaEquZroNbGrtZro = 1 + 8 * 6,
  signBaEquZroNbLesZro = 2 + 8 * 6,
  signBaEquOne         = 3 + 8 * 6,
  signBaGrtOne         = 4 + 8 * 6,
  gtx0BaEquZroCvGrt0   = 0 + 8 * 6 + 6,
  gtx0BaEquZroCvLeq0   = 1 + 8 * 6 + 6,
  gtx1BaEquZroCvGrt0   = 2 + 8 * 6 + 6,
  gtx1BaEquZroCvLeq0   = 3 + 8 * 6 + 6,
  gtx2BaEquZroCvGrt0   = 4 + 8 * 6 + 6,
  gtx2BaEquZroCvLeq0   = 5 + 8 * 6 + 6,
  gtx3BaEquZroCvGrt0   = 6 + 8 * 6 + 6,
  gtx3BaEquZroCvLeq0   = 7 + 8 * 6 + 6,
  gtx4BaEquZroCvGrt0   = 8 + 8 * 6 + 6,
  gtx4BaEquZroCvLeq0   = 9 + 8 * 6 + 6,
};

using namespace pybind11::literals;

class Exception : public std::exception
{
public:
  Exception( const std::string& _s ) : m_str( _s ) { }
  Exception( const Exception& _e ) : std::exception( _e ), m_str( _e.m_str ) { }
  virtual ~Exception() noexcept { };
  virtual const char* what() const noexcept { return m_str.c_str(); }
  Exception& operator=( const Exception& _e ) { std::exception::operator=( _e ); m_str = _e.m_str; return *this; }
  template<typename T> Exception& operator<<( T t ) { std::ostringstream oss; oss << t; m_str += oss.str(); return *this; }
private:
  std::string m_str;
};

#define THROW(x)            throw( Exception( "\nERROR: In function \"" ) << __FUNCTION__ << "\" in " << __FILE__ << ":" << __LINE__ << ": " << x )
#define CHECK(c,x)          if(c){ THROW(x); }

typedef float float32_t;

