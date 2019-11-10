/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __ENC_H
#define __ENC_H

#include <map>

#include "types.h"
#include "instruction.h"
#include "obj.h"
#include "trace.h"
  // } trace_inst_t;

namespace Harp {
  class DataChunk;
  class TextChunk;
  class Ref;

  class Encoder {
  public:
    Encoder() {}
    virtual ~Encoder() {}

    virtual Size encode(Ref *&ref, std::vector<Byte> &v, Size n, 
                        Instruction &i) = 0;
    void encodeChunk(DataChunk &dest, const TextChunk &src);
  };

  class Decoder {
  public:
    Decoder() : haveRefs(false) {}
    Decoder(const std::vector<Ref*> &refVec) : haveRefs(true) {
      setRefs(refVec);
    }

    virtual ~Decoder() {}

    void setRefs(const std::vector<Ref*> &);
    void clearRefs() { refMap.clear(); }
    virtual Instruction *decode(const std::vector<Byte> &v, Size &n, trace_inst_t * trace_inst) = 0;
    virtual Instruction *decode(const std::vector<Byte> &v, Size &n) = 0;
    void decodeChunk(TextChunk &dest, const DataChunk &src);
  protected:
    bool haveRefs;
    std::map <Size, Ref*> refMap;
  };

  class WordDecoder : public Decoder {
  public:
    WordDecoder(const ArchDef &);    
    virtual Instruction *decode(const std::vector<Byte> &v, Size &n, trace_inst_t * trace_inst);
    virtual Instruction *decode(const std::vector<Byte> &v, Size &n) {printf("Not implemented\n");}

  private:
    Size n, o, r, p, i1, i2, i3;
    Word oMask, rMask, pMask, i1Mask, i2Mask, i3Mask;

    // FARES
    Size inst_s, opcode_s, reg_s, func3_s;
    Size shift_opcode, shift_rd, shift_rs1, shift_rs2, shift_func3, shift_func7;
    Size shift_j_u_immed, shift_s_b_immed, shift_i_immed;



    Word reg_mask, func3_mask, func7_mask, opcode_mask, i_immed_mask, 
          s_immed_mask, b_immed_mask, u_immed_mask, j_immed_mask;

  };

};

#endif
