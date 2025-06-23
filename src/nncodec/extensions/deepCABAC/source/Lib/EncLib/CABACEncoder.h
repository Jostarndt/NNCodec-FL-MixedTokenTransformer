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
#ifndef __CABACENC__
#define __CABACENC__

#include "CommonLib/ContextModel.h"
#include "CommonLib/ContextModeler.h"
#include "CommonLib/Quant.h"
#include "CommonLib/Scan.h"
#include "BinEncoder.h"
#include <bitset>
#include <limits>
#include <iostream>

template< typename TBinEnc >
class TCABACEncoder
{
protected:
  __inline void xInitCtxModels(uint32_t numGtxFlags)
  {
    m_NumGtxFlags = numGtxFlags;
    m_CtxStore.resize(8 * 6 + 5 + m_NumGtxFlags * 4 + 32 + 5);
    for( uint32_t ctxId = 0; ctxId < m_CtxStore.size() ; ctxId++ )
    {
      m_CtxStore[ctxId].initState();
    }
    m_CtxModeler.init(numGtxFlags);
  }

  __inline void xResetCtxModels()
  {
    for (uint32_t ctxId = 0; ctxId < m_CtxStore.size(); ctxId++)
    {
      m_CtxStore[ctxId].resetState();
    }
  }

  template< uint32_t (TBinEnc::*FuncBinEnc)(uint32_t,SBMPCtx&) >
  __inline uint32_t xEncRemAbs( int32_t value, uint32_t remMaxAbsVal )
  {
    uint32_t scaledBits           = 0;
    uint32_t log2NumElemNextGroup = 0;
    int32_t  remAbsBaseLevel      = 0;
    uint32_t ctxIdx               = (8 * 6 + 5 + m_NumGtxFlags * 4);

    if (remMaxAbsVal > remAbsBaseLevel || remMaxAbsVal == -1)
    {
      if( value > 0 )
      {
        scaledBits += (m_BinEncoder.*FuncBinEnc)( 1, m_CtxStore[ ctxIdx ] );
        remAbsBaseLevel += (1 << log2NumElemNextGroup);
        ctxIdx++;
        log2NumElemNextGroup++;
      }
      else
      {
        return (m_BinEncoder.*FuncBinEnc)( 0, m_CtxStore[ ctxIdx ] );
      }
      while( value > ( remAbsBaseLevel + (1 << log2NumElemNextGroup) - 1 ) && (remMaxAbsVal == -1 || remMaxAbsVal >= (remAbsBaseLevel + (1 << log2NumElemNextGroup)  ) ) )
      {
        scaledBits += (m_BinEncoder.*FuncBinEnc)( 1 , m_CtxStore[ ctxIdx ] );
        remAbsBaseLevel += (1 << log2NumElemNextGroup);
        ctxIdx++;
        log2NumElemNextGroup++;
      }
      if(remMaxAbsVal == -1 || remMaxAbsVal >= (remAbsBaseLevel + (1 << log2NumElemNextGroup)  ) )
      {
        scaledBits += (m_BinEncoder.*FuncBinEnc)( 0, m_CtxStore[ ctxIdx ] );
      }
      scaledBits += m_BinEncoder.encodeBinsEP( value - remAbsBaseLevel, log2NumElemNextGroup );
    }
    return scaledBits;
  }

  template< uint32_t (TBinEnc::*FuncBinEnc)(uint32_t,SBMPCtx&) >
  __inline uint32_t xEncWeight( int32_t value, int32_t stateId, uint8_t general_profile_idc=0, uint32_t codebook_size=0, uint32_t codebook_zero_offset=0 )
  {
    if(codebook_size == 1 && general_profile_idc == 1)
    {
      return 0;
    }
  
    uint32_t sigFlag        = value != 0 ? 1 : 0;
    int32_t  sigctx         = m_CtxModeler.getSigCtxId( stateId );
    
    uint32_t scaledBits     = (m_BinEncoder.*FuncBinEnc)(sigFlag, m_CtxStore[ sigctx ]);
    
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
        scaledBits += (m_BinEncoder.*FuncBinEnc)(signFlag, m_CtxStore[ signCtx ]);
      }

      if(maxAbsVal == 1)
      {
        return scaledBits;
      }


      uint32_t remAbsLevel = abs(value) - 1;
      uint32_t grXFlag = remAbsLevel ? 1 : 0; //greater1
      int32_t ctxIdx = m_CtxModeler.getGtxCtxId( value, 0, stateId );
      
      if(maxAbsVal == -1 || maxAbsVal > 1)
      {
        scaledBits += (m_BinEncoder.*FuncBinEnc)(grXFlag, m_CtxStore[ ctxIdx ]);

        uint32_t numGreaterFlagsCoded = 1;

        while (grXFlag && (numGreaterFlagsCoded < m_NumGtxFlags) && ( maxAbsVal == -1 || maxAbsVal > numGreaterFlagsCoded +1 ) )
        {
          remAbsLevel--;
          grXFlag = remAbsLevel ? 1 : 0;
          ctxIdx =  m_CtxModeler.getGtxCtxId(value, numGreaterFlagsCoded, stateId);
          scaledBits += (m_BinEncoder.*FuncBinEnc)(grXFlag, m_CtxStore[ ctxIdx ]);
          numGreaterFlagsCoded++;
        }

        if(maxAbsVal != -1 && maxAbsVal == numGreaterFlagsCoded +1)
        {
          remAbsLevel--;
          numGreaterFlagsCoded++;
          grXFlag = 0;
        }

        if ( grXFlag && ( maxAbsVal == -1 || maxAbsVal > numGreaterFlagsCoded +1 ))
        {
          remAbsLevel--;
          scaledBits += xEncRemAbs<FuncBinEnc>( remAbsLevel, maxAbsVal == -1 ? -1 : maxAbsVal- (numGreaterFlagsCoded+1) );
        }
      }
    }
    return scaledBits;
  }

protected:
  std::vector<SBMPCtx> m_CtxStore;
  ContextModeler       m_CtxModeler;
  TBinEnc              m_BinEncoder;
  uint32_t             m_NumGtxFlags;
  uint8_t              m_ParamOptFlag;
  std::vector<SBMPCtxOptimizer> m_CtxStoreOpt;
};


class CABACEncoder : protected TCABACEncoder<BinEnc>
{
public:
  CABACEncoder() {}
  ~CABACEncoder() {}

  void      startCabacEncoding      (std::vector<uint8_t>* pBytestream);
  void      initCtxMdls             (uint32_t numGtxFlags,uint8_t param_opt_flag);
  void      resetCtxMdls            ();

  void xShiftParameterIds            ( uint8_t dq_flag, bool useTca, bool useHdsp, uint32_t codebook_size, uint32_t codebook_zero_offset );

  void      initOptimizerCtxMdls    (uint32_t numGtxFlags);
  void      resetOptimizerMdls      ();
  void      pseudoEncodeWeightVal   ( int32_t value, int32_t stateId, uint8_t general_profile_idc, uint32_t codebook_size=0, uint32_t codebook_zero_offset=0 );
  void      pseudoEncodeRemAbsLevelNew(uint32_t value, uint32_t remMaxAbsVal);

  void      terminateCabacEncoding  ();
  void      iae_v                   (uint8_t v,int32_t value);
  void      uae_v                   (uint8_t v,uint32_t value);

  int32_t encodeWeights(int32_t* pWeights,uint32_t layerWidth,uint32_t numWeights,const uint8_t dq_flag,const int32_t scan_order, uint8_t general_profile_idc, uint8_t parent_node_id_present_flag, uint8_t rowSkipFlag,int32_t* pChanZeroList, uint32_t codebook_size, uint32_t codebook_zero_offset, const HdspOpts& hdspOpts  );
  int32_t encodeWeights2(int32_t* pWeights,int32_t* pWeightsBase,uint32_t layerWidth,uint32_t numWeights,const uint8_t dq_flag,const int32_t scan_order,uint8_t general_profile_idc,uint8_t parent_node_id_present_flag,uint8_t rowSkipFlag,int32_t* pChanZeroList, uint32_t codebook_size, uint32_t codebook_zero_offset, const HdspOpts& hdspOpts  );

private:
  __inline void encodeWeightVal(int32_t weightInt,int32_t stateId, uint8_t general_profile_idc, uint32_t codebook_size=0, uint32_t codebook_zero_offset=0)
  {
    TCABACEncoder<BinEnc>::xEncWeight<&BinEnc::encodeBin>(weightInt, stateId, general_profile_idc, codebook_size, codebook_zero_offset);
  }

template <class trellisDef,bool usePseudoEnc,bool useTemporalCtx >
void xEncodeWeightsBase(Scan& scanIterator,int32_t* pWeights,int32_t* pWeightsBase,uint32_t layerWidth,uint32_t numWeights,uint8_t dq_flag, uint8_t general_profile_idc, uint8_t hist_dep_sig_prob_enabled_flag, uint8_t rowSkipFlag,int32_t* pChanZeroList, uint32_t codebook_size, uint32_t codebook_zero_offset, const HdspOpts& hdspOpts)
  {
    typename trellisDef::stateTransTab sttab = trellisDef::getStateTransTab();

    int32_t stateId = 0;
    int32_t skipRow = 0;
    
    for(int i = 0; i < (int)numWeights;)
    {
      if(general_profile_idc == 1 && rowSkipFlag && layerWidth > 1 && numWeights > layerWidth&& scanIterator.isFirstPositionOfRowInBlock() && codebook_size != 1)
      {
        uint32_t currRow = scanIterator.getRow();
        skipRow = pChanZeroList[currRow];

        if(skipRow == 1)
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
        int32_t value = pWeights[scanIterator.posInMat()];

        if(dq_flag && value != 0)
        {
          value += value < 0 ? -(stateId & 1) : (stateId & 1);
          value >>= 1;
        }


        if((general_profile_idc == 1) && useTemporalCtx )
        {
          m_CtxModeler.updateBaseMdlCtx(pWeightsBase[scanIterator.posInMat()]);
        }

        if(general_profile_idc == 1)
        {
          m_CtxModeler.updateHdspEnabled( hist_dep_sig_prob_enabled_flag == 1 && hdspOpts.getEnabledAt( scanIterator.posInMat()   )  );
        }
        if(usePseudoEnc)
        {
          // This is done as in the version without clean-up; 
          // Check if not using the modified value (for the case of dq_falg equal to 1) is correct
          pseudoEncodeWeightVal(pWeights[scanIterator.posInMat()],stateId, general_profile_idc);
          m_CtxModeler.updateNeighborCtx(pWeights[scanIterator.posInMat()],scanIterator.posInMat(),layerWidth);
        }
        else
        {
          xEncWeight<&BinEnc::encodeBin>(value,stateId, general_profile_idc, codebook_size, codebook_zero_offset);
          m_CtxModeler.updateNeighborCtx(value,scanIterator.posInMat(),layerWidth);
        }

        if(dq_flag)
        {
          stateId = sttab[stateId][value & 1];
        }
      }

      if(scanIterator.isLastPosOfBlockRowButNotLastPosOfBlock())
      {
        if(usePseudoEnc)
        {
          resetOptimizerMdls();
        }
        else
        {
          resetCtxMdls();
          m_CtxModeler.resetNeighborCtx();
          m_BinEncoder.entryPointStart();
        }
      }

      scanIterator++;
      i++;
    }
  }

  template <class trellisDef,bool useTca >
  int32_t xEncodeWeights(int32_t* pWeights,int32_t* pWeightsBase,uint32_t layerWidth,uint32_t numWeights,uint8_t dq_flag,const int32_t scan_order, uint8_t general_profile_idc, uint8_t parent_node_id_present_flag, uint8_t rowSkipFlag,int32_t* pChanZeroList, uint32_t codebook_size, uint32_t codebook_zero_offset, const HdspOpts& hdspOpts )
  {
    uint8_t hist_dep_sig_prob_enabled_flag = 0;

    m_CtxModeler.resetNeighborCtx();
    if(general_profile_idc == 1 && layerWidth > 1 && numWeights > layerWidth && codebook_size != 1)
    {
      if( parent_node_id_present_flag )
      {
          hist_dep_sig_prob_enabled_flag = hdspOpts.hdspEnabled() ? 1 : 0;
          m_BinEncoder.encodeBinEP( hist_dep_sig_prob_enabled_flag );
      }
      else
      {
        assert( ("hist_dep_sig_prob_enabled_flag can only be enabled if parent_node_id_present_flag equals one!", hist_dep_sig_prob_enabled_flag==0));
      }
      xEncRowSkip( general_profile_idc, rowSkipFlag, layerWidth, numWeights, pChanZeroList, codebook_size);
    }

    Scan scanIterator(ScanType(scan_order),numWeights,layerWidth); 

    if(m_ParamOptFlag && (codebook_size != 1 || general_profile_idc == 0))
    {
      // Pseudo encode with different initial context states and window sizes
      xEncodeWeightsBase<trellisDef,true,useTca>(scanIterator,pWeights,pWeightsBase,layerWidth,numWeights,dq_flag,general_profile_idc,hist_dep_sig_prob_enabled_flag,rowSkipFlag,pChanZeroList,codebook_size,codebook_zero_offset,hdspOpts);
    }
    xShiftParameterIds(dq_flag, useTca, hist_dep_sig_prob_enabled_flag, codebook_size, codebook_zero_offset);
    
    m_CtxModeler.resetNeighborCtx();

    scanIterator.resetScan();
    if(scan_order != 0 && scanIterator.getNumOfBlockRows() > 1)
    {
      m_BinEncoder.entryPointStart();
    }

    xEncodeWeightsBase<trellisDef,false,useTca>(scanIterator,pWeights,pWeightsBase,layerWidth,numWeights,dq_flag,general_profile_idc,hist_dep_sig_prob_enabled_flag,rowSkipFlag,pChanZeroList, codebook_size, codebook_zero_offset, hdspOpts);

    return m_NumGtxFlags;
  }

  void xEncRowSkip     ( uint8_t general_profile_idc, uint8_t rowSkipFlag,uint32_t layerWidth,uint32_t numWeights,int32_t* pChanZeroList, uint32_t codebook_size);
  uint8_t                       m_ParamOptFlag;

};

#endif // !__CABACENCIF__
