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
#include "CABACDecoder.h"
#include <iostream>

void CABACDecoder::startCabacDecoding(uint8_t *pBytestream)
{
  m_BinDecoder.setByteStreamBuf(pBytestream);
  m_BinDecoder.startBinDecoder();
}

void CABACDecoder::initCtxMdls(uint32_t cabac_unary_length)
{
  m_NumGtxFlags = cabac_unary_length;

  m_CtxStore.resize(8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 5);

  for (uint32_t ctxId = 0; ctxId < m_CtxStore.size(); ctxId++)
  {
    m_CtxStore[ctxId].initState();
  }
  m_CtxModeler.init(cabac_unary_length);
}

void CABACDecoder::resetCtxMdls()
{
  for (uint32_t ctxId = 0; ctxId < m_CtxStore.size(); ctxId++)
  {
    m_CtxStore[ctxId].resetState();
  }
}

int32_t CABACDecoder::iae_v(uint8_t v)
{
  uint32_t pattern = m_BinDecoder.decodeBinsEP(v);
  return int32_t(pattern << (32 - v)) >> (32 - v);
}

uint32_t CABACDecoder::uae_v(uint8_t v)
{
  return m_BinDecoder.decodeBinsEP(v);
}

void CABACDecoder::xShiftParameterIds( uint8_t dq_flag, bool useTca, bool useHdsp, uint32_t codebook_size, uint32_t codebook_zero_offset )
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
      if (m_BinDecoder.decodeBin(m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]))
      {
        bestEcoIdx += (1 + m_BinDecoder.decodeBinsEP(3));
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
        if (m_BinDecoder.decodeBin(m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]))
        {
          bestEcoIdx += (1 + m_BinDecoder.decodeBinsEP(3));
        }
      }
      m_CtxStore[i].initState(bestEcoIdx);
    }
  }

  if( useHdsp )
  {
    for(int i = 40; i < (dq_flag ? 48 : 41); i++)
    {
      bestEcoIdx = 0;

      if( codebook_size != 1 )
      {
        if (m_BinDecoder.decodeBin(m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]))
        {
          bestEcoIdx += (1 + m_BinDecoder.decodeBinsEP(3));
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
      if (m_BinDecoder.decodeBin(m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]))
      {
        bestEcoIdx += (1 + m_BinDecoder.decodeBinsEP(3));
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
      if (m_BinDecoder.decodeBin(m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]))
      {
        bestEcoIdx += (1 + m_BinDecoder.decodeBinsEP(3));
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
      if (m_BinDecoder.decodeBin(m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]))
        {
          bestEcoIdx += (1 + m_BinDecoder.decodeBinsEP(3));
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
      if (m_BinDecoder.decodeBin(m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 2]))
      {
        bestEcoIdx += (1 + m_BinDecoder.decodeBinsEP(3));
      }
    }
     m_CtxStore[i].initState(bestEcoIdx);
  }
}

template <class trellisDef,bool bCreateEntryPoints,bool bPrevCtx >
void CABACDecoder::decodeWeightsBase(int32_t* pWeights, int32_t* pWeightsBase, uint32_t layerWidth,uint32_t numWeights,uint8_t dq_flag,const int32_t scan_order,uint8_t general_profile_idc,uint8_t parent_node_id_present_flag,std::vector<uint64_t>& entryPoints, uint32_t codebook_size, uint32_t codebook_zero_offset, const HdspOpts& hdspOpts)
{
  typename trellisDef::stateTransTab sttab = trellisDef::getStateTransTab();

  std::vector<int32_t> chanSkip;
  uint8_t rowSkipFlag = 0;
  uint8_t hist_dep_sig_prob_enabled_flag = 0;

  if(general_profile_idc == 1 && layerWidth > 1 && numWeights > layerWidth && codebook_size != 1)
  {
    if(parent_node_id_present_flag)
    {
       hist_dep_sig_prob_enabled_flag = m_BinDecoder.decodeBinEP();
    }
    rowSkipFlag = m_BinDecoder.decodeBinEP();

    if( rowSkipFlag )
    {
      int32_t numRows = numWeights / layerWidth;
      int32_t skipRow = 0;

      for(int row = 0; row < numRows; row++)
      {
        skipRow = m_BinDecoder.decodeBin(m_CtxStore[8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 4]);
        chanSkip.push_back(skipRow);
      }
    }
  }

  xShiftParameterIds(dq_flag, bPrevCtx, hist_dep_sig_prob_enabled_flag, codebook_size, codebook_zero_offset);

  m_CtxModeler.resetNeighborCtx();
  int32_t stateId = 0;

  Scan scanIterator(ScanType(scan_order),numWeights,layerWidth);

  uint64_t lastBitOffset = 0;
  if(!bCreateEntryPoints && scan_order != 0 && !entryPoints.empty())
  {
    uint32_t bytesReadBefore = m_BinDecoder.getBytesRead();
    uint8_t* byteStreamPtrBefore = m_BinDecoder.getByteStreamPtr();
    uint8_t* byteStreamPtrAfter = nullptr;
    EntryPoint firstEp = m_BinDecoder.getEntryPoint();
    firstEp.dqState = stateId;

    lastBitOffset = firstEp.totalBitOffset;
    // convert from differential entry points to absolute entry points
    for(int epIdx = 0; epIdx < entryPoints.size(); epIdx++)
    {
      EntryPoint ep;
      ep.setEntryPointInt(entryPoints[epIdx]);
      ep.totalBitOffset += lastBitOffset;
      lastBitOffset = ep.totalBitOffset;
      entryPoints[epIdx] = ep.getEntryPointInt();
    }
    entryPoints.insert(entryPoints.begin(),firstEp.getEntryPointInt());

    EntryPoint finalEntryPoint;

    for(int epIdx = (int)entryPoints.size() - 1; epIdx >= 0; epIdx--)
    {
      //int epIdx2 = epIdx == -1 ? m_EntryPoints.size() - 1 : epIdx;
      scanIterator.seekBlockRow(epIdx);
      EntryPoint ep;
      ep.setEntryPointInt(entryPoints[epIdx]);
      m_BinDecoder.setEntryPoint(ep);
      stateId = ep.dqState;
      resetCtxMdls();
      m_CtxModeler.resetNeighborCtx(); //TODO HAASE: WHAT ABPUT THIS?

      int32_t skipRow = 0;

      while(true)
      {
        if(general_profile_idc == 1 && rowSkipFlag && layerWidth > 1 && numWeights > layerWidth&& scanIterator.isFirstPositionOfRowInBlock() && codebook_size != 1)
        {
          uint32_t currRow = scanIterator.getRow();

          skipRow = chanSkip.at(currRow);
          if(skipRow)
          {
            scanIterator.seekRowEndOfCurrBlockAndReturnInc();
            if(dq_flag)
            {
              for(int a = 0; a <= (int)layerWidth - 1; a++)
              {
                stateId = sttab[stateId][0];
              }
            }
          }
        }
        if(!skipRow || general_profile_idc == 0)
        {
          pWeights[scanIterator.posInMat()] = 0;
          if(bPrevCtx && general_profile_idc == 1)
          {
            m_CtxModeler.updateBaseMdlCtx(pWeightsBase[scanIterator.posInMat()]);
          }
          if(general_profile_idc == 1)
          {
            m_CtxModeler.updateHdspEnabled( hist_dep_sig_prob_enabled_flag == 1 && hdspOpts.getEnabledAt( scanIterator.posInMat()   )  );
          }
          decodeWeightVal(pWeights[scanIterator.posInMat()],stateId, general_profile_idc, codebook_size, codebook_zero_offset);
          m_CtxModeler.updateNeighborCtx(pWeights[scanIterator.posInMat()],scanIterator.posInMat(),layerWidth);
          if(dq_flag)
          {
            int32_t newState = sttab[stateId][pWeights[scanIterator.posInMat()] & 1];

            if(pWeights[scanIterator.posInMat()] != 0)
            {
              pWeights[scanIterator.posInMat()] <<= 1;
              pWeights[scanIterator.posInMat()] += pWeights[scanIterator.posInMat()] < 0 ? (stateId & 1) : -(stateId & 1);
            }

            stateId = newState;
          }
        }
        if(scanIterator.isLastPosOfBlockRow())
        {
          if(epIdx == entryPoints.size() - 1) //last Entry Point
          {
            finalEntryPoint = m_BinDecoder.getEntryPoint();
            byteStreamPtrAfter = m_BinDecoder.getByteStreamPtr();
          }
          break;
        }
        scanIterator++;
      }
    }

    m_BinDecoder.setByteStreamPtr(byteStreamPtrAfter);
    m_BinDecoder.setBytesRead(uint32_t(bytesReadBefore + (byteStreamPtrAfter - byteStreamPtrBefore)));
    m_BinDecoder.setEntryPointWithRange(finalEntryPoint);
  }
  else
  {
    if(bCreateEntryPoints && scan_order != 0)
    {
      m_BinDecoder.entryPointStart();
      EntryPoint ep = m_BinDecoder.getEntryPoint();
      ep.dqState = stateId;
      lastBitOffset = ep.totalBitOffset;
    }

    int32_t skipRow = 0;

    for(int i = 0; i < (int)numWeights;)
    {
      if(general_profile_idc == 1 && rowSkipFlag && layerWidth > 1 && numWeights > layerWidth&& scanIterator.isFirstPositionOfRowInBlock() && codebook_size != 1)
      {
        uint32_t currRow = scanIterator.getRow();

        skipRow = chanSkip.at(currRow);
        if(skipRow)
        {
          i += scanIterator.seekRowEndOfCurrBlockAndReturnInc();
          if(dq_flag)
          {
            for(int a = 0; a <= (int)layerWidth - 1; a++)
            {
              stateId = sttab[stateId][0];
            }
          }
        }
      }

      if(!skipRow || general_profile_idc == 0) 
      {
        pWeights[scanIterator.posInMat()] = 0;
        if(bPrevCtx && general_profile_idc == 1)
        { 
          m_CtxModeler.updateBaseMdlCtx(pWeightsBase[scanIterator.posInMat()]);
        }
        if(general_profile_idc == 1)
        {
          m_CtxModeler.updateHdspEnabled( hist_dep_sig_prob_enabled_flag == 1 && hdspOpts.getEnabledAt( scanIterator.posInMat()   )  );
        }
        decodeWeightVal(pWeights[scanIterator.posInMat()],stateId, general_profile_idc, codebook_size, codebook_zero_offset);

        m_CtxModeler.updateNeighborCtx(pWeights[scanIterator.posInMat()],scanIterator.posInMat(),layerWidth);

        if(dq_flag)
        {
          int32_t newState = sttab[stateId][pWeights[scanIterator.posInMat()] & 1];

          if(pWeights[scanIterator.posInMat()] != 0)
          {
            pWeights[scanIterator.posInMat()] <<= 1;
            pWeights[scanIterator.posInMat()] += pWeights[scanIterator.posInMat()] < 0 ? (stateId & 1) : -(stateId & 1);
          }

          stateId = newState;
        }
      }

      if(bCreateEntryPoints && scanIterator.isLastPosOfBlockRowButNotLastPosOfBlock())
      {
        resetCtxMdls();
        m_CtxModeler.resetNeighborCtx();
        m_BinDecoder.entryPointStart();
        EntryPoint ep = m_BinDecoder.getEntryPoint();
        ep.dqState = stateId;
        uint64_t deltaOffset = ep.totalBitOffset - lastBitOffset;
        lastBitOffset = ep.totalBitOffset;
        ep.totalBitOffset = deltaOffset;
        entryPoints.push_back(ep.getEntryPointInt());
      }

      scanIterator++;
      i++;
    }
  }
}


void CABACDecoder::decodeWeights(int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, uint8_t dq_flag, const int32_t scan_order, uint8_t general_profile_idc, uint8_t parent_node_id_present_flag, uint32_t codebook_size, uint32_t codebook_zero_offset, const HdspOpts& hdspOpts)
{
  const QuantType qtype = QuantType(dq_flag);

  if (qtype == URQ || qtype == TCQ8States)
  {
    return decodeWeightsBase<Trellis8States,false,false >(pWeights,nullptr,layerWidth,numWeights,dq_flag,scan_order,general_profile_idc, parent_node_id_present_flag, m_EntryPoints, codebook_size, codebook_zero_offset, hdspOpts); 
  }
  assert(!"Unsupported TCQType");
}

void CABACDecoder::decodeWeights2(int32_t *pWeights, int32_t *pWeightsBase, uint32_t layerWidth, uint32_t numWeights, uint8_t dq_flag, const int32_t scan_order, uint8_t general_profile_idc, uint8_t parent_node_id_present_flag, uint32_t codebook_size, uint32_t codebook_zero_offset, const HdspOpts& hdspOpts)
{
  const QuantType qtype = QuantType(dq_flag);

  if (qtype == URQ || qtype == TCQ8States)
  {
    return decodeWeightsBase<Trellis8States,false,true>(pWeights, pWeightsBase, layerWidth, numWeights, dq_flag, scan_order, general_profile_idc, parent_node_id_present_flag, m_EntryPoints, codebook_size, codebook_zero_offset, hdspOpts);
  }
  assert(!"Unsupported TCQType");
}


void CABACDecoder::decodeWeightsAndCreateEPs(int32_t *pWeights, uint32_t layerWidth, uint32_t numWeights, uint8_t dq_flag, const int32_t scan_order, uint8_t general_profile_idc, uint8_t parent_node_id_present_flag, std::vector<uint64_t> &entryPoints, uint32_t codebook_size, uint32_t codebook_zero_offset, const HdspOpts& hdspOpts)
{
  const QuantType qtype = QuantType(dq_flag);

  if (qtype == URQ || qtype == TCQ8States)
  {
    return decodeWeightsBase<Trellis8States, true, false >( pWeights, nullptr, layerWidth,  numWeights, dq_flag,  scan_order,  general_profile_idc,  parent_node_id_present_flag, entryPoints, codebook_size, codebook_zero_offset, hdspOpts );
  }
  assert(!"Unsupported TCQType");
}

void CABACDecoder::decodeWeightsAndCreateEPs2(int32_t *pWeights, int32_t *pWeightsBase, uint32_t layerWidth, uint32_t numWeights, uint8_t dq_flag, const int32_t scan_order, uint8_t general_profile_idc, uint8_t parent_node_id_present_flag, std::vector<uint64_t> &entryPoints, uint32_t codebook_size, uint32_t codebook_zero_offset, const HdspOpts& hdspOpts)
{
  const QuantType qtype = QuantType(dq_flag);

  if (qtype == URQ || qtype == TCQ8States)
  {
    return decodeWeightsBase<Trellis8States, true, true >( pWeights, pWeightsBase, layerWidth,  numWeights, dq_flag,  scan_order,  general_profile_idc, parent_node_id_present_flag, entryPoints, codebook_size, codebook_zero_offset, hdspOpts );
  }
  assert(!"Unsupported TCQType");
}

void CABACDecoder::setEntryPoints(uint64_t *pEntryPoints, uint64_t numEntryPoints)
{
  m_EntryPoints.resize(numEntryPoints);

  for (int i = 0; i < m_EntryPoints.size(); i++)
  {
    m_EntryPoints[i] = pEntryPoints[i];
  }
}

void CABACDecoder::decodeWeightVal(int32_t &decodedIntVal, int32_t stateId, uint8_t general_profile_idc, uint32_t codebook_size, uint32_t codebook_zero_offset)
{ 
  if( general_profile_idc == 0 ||  codebook_size != 1 )
  {
    int32_t sigctx = m_CtxModeler.getSigCtxId(stateId);
    uint32_t sigFlag = m_BinDecoder.decodeBin(m_CtxStore[sigctx]);

    if (sigFlag)
    {
      decodedIntVal++;
      uint32_t signFlag = 0; //SignVal
      int32_t signCtx = m_CtxModeler.getSignFlagCtxId();

      if( general_profile_idc == 1 && (codebook_size > 0 && (codebook_zero_offset == 0 || codebook_zero_offset == codebook_size -1)) )
      {
        signFlag = codebook_zero_offset != 0 ? 1 : 0;
      }
      else
      {
        signFlag = m_BinDecoder.decodeBin(m_CtxStore[signCtx]);
      }

      int32_t intermediateVal = signFlag ? -1 : 1;



      uint32_t grXFlag = 0;

      int32_t j = -1;

      int64_t maxAbsVal = 0; 
      if(general_profile_idc == 1 && codebook_size > 0)
      {
        maxAbsVal = signFlag ? codebook_zero_offset : ( codebook_size - codebook_zero_offset - 1 );
      }
      else
      {
        maxAbsVal = -1;
      }

      if( maxAbsVal > 1 || maxAbsVal == -1 )
      {
        int32_t ctxIdx = 0;
        
        do{
          j++;
          ctxIdx = m_CtxModeler.getGtxCtxId(intermediateVal, j, stateId);
          grXFlag = m_BinDecoder.decodeBin(m_CtxStore[ctxIdx]);
          decodedIntVal += grXFlag;
        } while( grXFlag == 1 && j < m_NumGtxFlags-1 && ( maxAbsVal > decodedIntVal || maxAbsVal == -1 ) );

        if( grXFlag == 1 && (maxAbsVal > decodedIntVal || maxAbsVal == -1) )
        {
          uint32_t RemBits = 0;
          j = -1;

          do{
            j++;

            ctxIdx = (8 * 6 + 5 + m_NumGtxFlags * 4) + j;
            grXFlag = m_BinDecoder.decodeBin(m_CtxStore[ctxIdx]);
            if(grXFlag && (maxAbsVal > decodedIntVal || maxAbsVal == -1))
            {
              decodedIntVal += 1 << RemBits;
              RemBits++;
            }
          }while( grXFlag == 1 && j < 30 && ( maxAbsVal >= ( decodedIntVal + ( 1 << RemBits ) ) || maxAbsVal == -1 ) );

          decodedIntVal += (int32_t)m_BinDecoder.decodeBinsEP(RemBits);
        }
      }
      decodedIntVal = signFlag ? -decodedIntVal : decodedIntVal;
    }
  }
}

int32_t CABACDecoder::decodeRemAbsLevel()
{
  int32_t remAbsLevel = 0;
  uint32_t log2NumElemNextGroup = 0;
  uint32_t ctxIdx = (8 * 6 + 5 + m_NumGtxFlags * 4);
  if (m_BinDecoder.decodeBin(m_CtxStore[ctxIdx]))
  {
    remAbsLevel += (1 << log2NumElemNextGroup);
    ctxIdx++;
    log2NumElemNextGroup++;
  }
  else
  {
    return remAbsLevel;
  }

  while (m_BinDecoder.decodeBin(m_CtxStore[ctxIdx]))
  {
    remAbsLevel += (1 << log2NumElemNextGroup);
    ctxIdx++;
    log2NumElemNextGroup++;
  }

  remAbsLevel += (int32_t)m_BinDecoder.decodeBinsEP(log2NumElemNextGroup);
  return remAbsLevel;
}

uint32_t CABACDecoder::getBytesRead()
{
  return m_BinDecoder.getBytesRead();
}

uint32_t CABACDecoder::terminateCabacDecoding()
{
  if (m_BinDecoder.decodeBinTrm())
  {
    m_BinDecoder.finish();
    return m_BinDecoder.getBytesRead();
  }
  CHECK(1, "Terminating Bin not received!");
}
