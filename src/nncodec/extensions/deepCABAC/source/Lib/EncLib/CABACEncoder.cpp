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
#include "CABACEncoder.h"
#include <iostream>
#include <cstdlib>
#include <cmath>


void CABACEncoder::startCabacEncoding( std::vector<uint8_t>* pBytestream )
{
    m_BinEncoder.setByteStreamBuf(pBytestream);
    m_BinEncoder.startBinEncoder();
}

void CABACEncoder::initCtxMdls(uint32_t numGtxFlags, uint8_t param_opt_flag)
{
  TCABACEncoder<BinEnc>::xInitCtxModels(numGtxFlags);
  initOptimizerCtxMdls(numGtxFlags);

  m_ParamOptFlag = param_opt_flag;
}

void CABACEncoder::resetCtxMdls()
{
  TCABACEncoder<BinEnc>::xResetCtxModels();
}

void CABACEncoder::initOptimizerCtxMdls(uint32_t numGtxFlags)
{
  m_CtxStoreOpt.resize(8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 5);

  for (uint32_t ctxId = 0; ctxId < m_CtxStoreOpt.size(); ctxId++)
  {
    m_CtxStoreOpt[ctxId].initStates();
  }
}

void CABACEncoder::resetOptimizerMdls()
{
  for (uint32_t ctxId = 0; ctxId < m_CtxStoreOpt.size(); ctxId++)
  {
    m_CtxStoreOpt[ctxId].resetStates();
  }
}

void CABACEncoder::iae_v( uint8_t v, int32_t value )
{
    uint32_t pattern = uint32_t(value) & (uint32_t(0xFFFFFFFF) >> (32-v));
    m_BinEncoder.encodeBinsEP( pattern, v );
}

void CABACEncoder::uae_v( uint8_t v, uint32_t value )
{
    m_BinEncoder.encodeBinsEP( value, v );
}

void CABACEncoder::terminateCabacEncoding()
{
    m_BinEncoder.encodeBinTrm(1);
    m_BinEncoder.finish();
}

void CABACEncoder::pseudoEncodeRemAbsLevelNew(uint32_t value, uint32_t remMaxAbsVal )
{
  int32_t  remAbsBaseLevel = 0;
  uint32_t log2NumElemNextGroup = 0;
  uint32_t ctxIdx = (8 * 6 + 5 + m_NumGtxFlags * 4);
  if (remMaxAbsVal > remAbsBaseLevel)
  {
    if (value > 0)
    {
      m_BinEncoder.pseudoEncodeBin(1, m_CtxStoreOpt[ctxIdx]);
      remAbsBaseLevel += (1 << log2NumElemNextGroup);
      ctxIdx++;
      log2NumElemNextGroup++;
    }
    else
    {
      m_BinEncoder.pseudoEncodeBin(0, m_CtxStoreOpt[ctxIdx]);
      return;
    }
    while( value > (remAbsBaseLevel + (1 << log2NumElemNextGroup) - 1) && (remMaxAbsVal == -1 || remMaxAbsVal >= (remAbsBaseLevel + (1 << log2NumElemNextGroup)  )) )
    {
      m_BinEncoder.pseudoEncodeBin(1, m_CtxStoreOpt[ctxIdx]);
      remAbsBaseLevel += (1 << log2NumElemNextGroup);
      ctxIdx++;
      log2NumElemNextGroup++;
    }

    m_BinEncoder.pseudoEncodeBin(0, m_CtxStoreOpt[ctxIdx]);
    //no pseudoEncode of EP bins
  }
}

void  CABACEncoder::pseudoEncodeWeightVal( int32_t value, int32_t stateId, uint8_t general_profile_idc, uint32_t codebook_size, uint32_t codebook_zero_offset )
{
  if(codebook_size == 1 && general_profile_idc == 1)
  {
    return;
  }

  uint32_t sigFlag = value != 0 ? 1 : 0;
  int32_t sigctx = m_CtxModeler.getSigCtxId(stateId);

  m_BinEncoder.pseudoEncodeBin(sigFlag, m_CtxStoreOpt[sigctx]);

  if (sigFlag)
  {
    uint32_t signFlag = value < 0 ? 1 : 0;

    int64_t maxAbsVal = 0;
    if( codebook_size > 0 && general_profile_idc == 1)
    {
      maxAbsVal = signFlag ? codebook_zero_offset : ( codebook_size - codebook_zero_offset -1 );
    }
    else
    {
      maxAbsVal = -1;
    }

    if( !(codebook_size > 0 && ( codebook_zero_offset == 0 || codebook_zero_offset == codebook_size-1 )) || general_profile_idc == 0)
    {
      int32_t signCtx = m_CtxModeler.getSignFlagCtxId();
      m_BinEncoder.pseudoEncodeBin(signFlag, m_CtxStoreOpt[signCtx]);
    }

    if(maxAbsVal == 1)
    {
      return;
    }

    uint32_t remAbsLevel = abs(value) - 1;
    uint32_t grXFlag = remAbsLevel ? 1 : 0; //greater1
    int32_t ctxIdx = m_CtxModeler.getGtxCtxId(value, 0, stateId);

    if(maxAbsVal == -1 || maxAbsVal > 1)
    {
      m_BinEncoder.pseudoEncodeBin(grXFlag, m_CtxStoreOpt[ctxIdx]);

      uint32_t numGreaterFlagsCoded = 1;

      while (grXFlag && (numGreaterFlagsCoded < m_NumGtxFlags) && ( maxAbsVal == -1 || maxAbsVal > numGreaterFlagsCoded +1 ) )
      {
        remAbsLevel--;
        grXFlag = remAbsLevel ? 1 : 0;
        ctxIdx = m_CtxModeler.getGtxCtxId(value, numGreaterFlagsCoded, stateId);
        m_BinEncoder.pseudoEncodeBin(grXFlag, m_CtxStoreOpt[ctxIdx]);
        numGreaterFlagsCoded++;
      }

      if (grXFlag && ( maxAbsVal == -1 || maxAbsVal > numGreaterFlagsCoded +1 ) )
      {
        remAbsLevel--;
        pseudoEncodeRemAbsLevelNew(remAbsLevel, maxAbsVal == -1 ? -1 : maxAbsVal - (numGreaterFlagsCoded+1));
      }
    }
  }
}

void CABACEncoder::xShiftParameterIds( uint8_t dq_flag, bool useTca, bool useHdsp, uint32_t codebook_size, uint32_t codebook_zero_offset )
{
  int32_t offset_sign       = 48;
  int32_t offset_grX        = 53;
  int32_t offset_grX2       = 53 + 4* m_NumGtxFlags;
  uint8_t bestEcoIdx = 0;
  
  for(int i = 0; i < (dq_flag ? 24 : 3); i++)
  {
    bestEcoIdx = 0;
    if ( codebook_size != 1 )
    {
      bestEcoIdx = m_CtxStoreOpt[i].getBestIdx();

      m_BinEncoder.encodeBin(bestEcoIdx ? 1 : 0, m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]); //second last ctx model
      if (bestEcoIdx != 0)
      {
        m_BinEncoder.encodeBinsEP(bestEcoIdx - 1, 3);
      }
    }
    m_CtxStore[i].initState(bestEcoIdx);
  }

  if(useTca)
  {
    for(int i = 24; i < (dq_flag ? 40 : 26); i++)
    {
      bestEcoIdx = 0;
      if( codebook_size != 1 )
      {
        bestEcoIdx = m_CtxStoreOpt[i].getBestIdx();
        m_BinEncoder.encodeBin(bestEcoIdx ? 1 : 0, m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]); //second last ctx model
        if (bestEcoIdx != 0)
        {
          m_BinEncoder.encodeBinsEP(bestEcoIdx - 1, 3);
        }
      }

      m_CtxStore[i].initState(bestEcoIdx);
    }
  }

  if(useHdsp)
  {
    for(int i = 40; i < (dq_flag ? 48 : 41); i++)
    {
      bestEcoIdx = 0;
      if( codebook_size != 1 )
      {
        bestEcoIdx = m_CtxStoreOpt[i].getBestIdx();
        m_BinEncoder.encodeBin(bestEcoIdx ? 1 : 0, m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]); //second last ctx model
        if (bestEcoIdx != 0)
        {
          m_BinEncoder.encodeBinsEP(bestEcoIdx - 1, 3);
        }
      }

      m_CtxStore[i].initState(bestEcoIdx);
    }
  }

  int i = 0;
  for( int a = 0; a < (useTca ? 5 : 3) ; a++)
  {
    bestEcoIdx = 0; 
    i = offset_sign + a;
    if( !(codebook_size > 0 && (codebook_zero_offset == 0 || codebook_zero_offset == codebook_size - 1)) && codebook_size != 1 )
    {
      bestEcoIdx = m_CtxStoreOpt[i].getBestIdx();
      m_BinEncoder.encodeBin(bestEcoIdx ? 1 : 0, m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]); //second last ctx model
      if (bestEcoIdx != 0)
      {
        m_BinEncoder.encodeBinsEP(bestEcoIdx - 1, 3);
      }
    }
    m_CtxStore[i].initState(bestEcoIdx);
  }
  
  int64_t maxAbsVal = codebook_size > 0 ? std::max( codebook_zero_offset, codebook_size -1 -codebook_zero_offset ) : - 1;
  for( int a = 0; a < 2*m_NumGtxFlags;  a++)
  {
    bestEcoIdx = 0;
    i = offset_grX + a;
    if( maxAbsVal == -1 || a < (maxAbsVal * 2) )
    {
      bestEcoIdx = m_CtxStoreOpt[i].getBestIdx();
      m_BinEncoder.encodeBin(bestEcoIdx ? 1 : 0, m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]); //second last ctx model
      if (bestEcoIdx != 0)
      {
        m_BinEncoder.encodeBinsEP(bestEcoIdx - 1, 3);
      }
    }
    m_CtxStore[i].initState(bestEcoIdx);
  }

  if(useTca)
  {
    for( int a = 2*m_NumGtxFlags; a < 4*m_NumGtxFlags;  a++)
    {
      bestEcoIdx = 0;
      i = offset_grX + a;
      if( maxAbsVal == -1 || a < ( (maxAbsVal-1)*2 + (2*m_NumGtxFlags) ) )
      {
        bestEcoIdx = m_CtxStoreOpt[i].getBestIdx();
      m_BinEncoder.encodeBin(bestEcoIdx ? 1 : 0, m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]); //second last ctx model
        if (bestEcoIdx != 0)
        {
          m_BinEncoder.encodeBinsEP(bestEcoIdx - 1, 3);
        }
      }
      m_CtxStore[i].initState(bestEcoIdx);
    }
  }

  int64_t currX2Level = m_NumGtxFlags - 1;
  for( int a = 0; a < 31; a++ )
  {
    bestEcoIdx = 0;
    i = offset_grX2 + a;
    currX2Level += (1 << a);
    if( maxAbsVal == -1 || currX2Level < maxAbsVal )
    {
      bestEcoIdx = m_CtxStoreOpt[i].getBestIdx();
      m_BinEncoder.encodeBin(bestEcoIdx ? 1 : 0, m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]); //second last ctx model
      if (bestEcoIdx != 0)
      {
        m_BinEncoder.encodeBinsEP(bestEcoIdx - 1, 3);
      }
    }
     m_CtxStore[i].initState(bestEcoIdx);
  }
}


void CABACEncoder::xEncRowSkip(uint8_t general_profile_idc, uint8_t rowSkipFlag,uint32_t layerWidth,uint32_t numWeights,int32_t* pChanZeroList, uint32_t codebook_size)
{
  if(general_profile_idc == 1 && layerWidth > 1 && numWeights > layerWidth && codebook_size != 1)
  {
    m_BinEncoder.encodeBinEP( rowSkipFlag ? 1 : 0 );
    
    if(rowSkipFlag)
    {
      int32_t numRows = numWeights / layerWidth;
      for(int row = 0; row < numRows; row++)
      {
        m_BinEncoder.encodeBin(pChanZeroList[row],m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 4]);
    }
    }
  }
}

int32_t CABACEncoder::encodeWeights(int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, const uint8_t dq_flag, const int32_t scan_order, uint8_t general_profile_idc, uint8_t parent_node_id_present_flag, uint8_t rowSkipFlag, int32_t* pChanZeroList, uint32_t codebook_size, uint32_t codebook_zero_offset, const HdspOpts& hdspOpts  )
{
  const QuantType qtype = QuantType( dq_flag );

  if ( qtype == URQ || qtype == TCQ8States)
  {
    return xEncodeWeights<Trellis8States,false>( pWeights, nullptr, layerWidth, numWeights, dq_flag, scan_order, general_profile_idc, parent_node_id_present_flag, rowSkipFlag, pChanZeroList, codebook_size, codebook_zero_offset , hdspOpts  );
  }
  assert( !"Unsupported TCQType" );
}


int32_t CABACEncoder::encodeWeights2(int32_t *pWeights, int32_t *pWeightsBase, uint32_t layerWidth, uint32_t numWeights, const uint8_t dq_flag, const int32_t scan_order, uint8_t general_profile_idc, uint8_t parent_node_id_present_flag, uint8_t rowSkipFlag, int32_t* pChanZeroList, uint32_t codebook_size, uint32_t codebook_zero_offset, const HdspOpts& hdspOpts  )
{
  const QuantType qtype = QuantType( dq_flag );

  if ( qtype == URQ || qtype == TCQ8States)
  {
    return xEncodeWeights<Trellis8States,true>( pWeights, pWeightsBase, layerWidth, numWeights, dq_flag, scan_order, general_profile_idc, general_profile_idc, rowSkipFlag, pChanZeroList, codebook_size, codebook_zero_offset, hdspOpts );
  }
  assert( !"Unsupported TCQType" );
}
