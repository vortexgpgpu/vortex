/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __OBJ_H
#define __OBJ_H

#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <stdlib.h>

#include "types.h"
#include "archdef.h"
#include "instruction.h"
#include "enc.h"
#include "asm-tokens.h"

namespace Harp {
  class Decoder;
  class Encoder;

  class Ref {
  public:
    std::string name;
    Ref(const std::string &n, bool r, Size ib = 0): 
      name(n), bound(false), relative(r), ibase(ib) { }
    virtual ~Ref() { }
    virtual void bind(Addr addr, Addr base = 0) = 0;
    virtual Addr getAddr() const = 0;

    bool bound, relative;
    Size ibase;
  };

  /* Used in not-yet-encoded code objects, plain old data. */
  class SimpleRef : public Ref {
  public:
    SimpleRef(const std::string &name, Addr &addr, bool rel = false) : 
      Ref(name, rel), addr(addr) { }
    virtual void bind(Addr addr, Addr base = 0) {
      std::cout << "Attempted to bind a SimpleRef.\n";
      std::abort();
    } 
    virtual Addr getAddr() const { return this->addr; }
    Byte *getAddrPtr() { return (Byte*)&addr; }

  private:
    Addr &addr;
  };

//   /* Used in already-encoded code objects. */
//   class OffsetRef : public Ref {
//   public:
//     OffsetRef(
//       const std::string &name, std::vector<Byte> &v, Size offset, Size bits,
//       Size ws, bool rel = false, Size ibase = 0
//     ) : Ref(name, rel, ibase), data(v), offset(offset), bits(bits), wordSize(ws)
//     {}

//     virtual void bind(Addr addr, Addr base = 0) {
//       Size bytes(bits/8), remainder(bits%8);

//       if (relative) {
//         addr = addr - base;
//         Word_s addr_s(addr);
//         if ((addr_s >> bits) != ~0ull && (addr_s >> bits) != 0) goto noFit;
//       } else {
//         Addr mask = (1ull<<bits)-1;
//         if (addr > mask) goto noFit;
//       }

//       { Byte mask((1ull<<remainder) - 1);
//         Size i;
//         for (i = 0; i < bytes; i++) {
//           data[offset+i] = addr & 0xff;
//           addr >>= 8;
//         }
//         data[offset+i] &= ~mask;
//         data[offset+i] |= (addr&mask);
//         bound = true; 
//       }

//       return;
//     noFit:
//       std::cout << "Attempt to bind a " << bits << "-bit "
//                 << (relative?"":"non-") << "relative symbol to an address"
//                                            " it cannot reach.\n";
//       std::abort();
//     }

//     virtual Addr getAddr() const {
//       Size bytes = bits/8, remainder = bits%8;
//       Byte mask((1<<remainder)-1);
//       Addr a(data[offset]&mask);

//       for (Size i = 0; i < bytes-1; i++) {
//         a |= data[offset + bytes - i - 1];
//         a <<= 8;
//       }
//       return a;
//     }

//     Size getOffset() const { return offset; }
//     Size getBits() const { return bits; }

//   private:
//     std::vector<Byte> &data;
//     Size offset, bits, wordSize;
//   };

//   class Chunk { 
//   public:
//     Chunk(std::string n, Size a = 0, Word f = 0) : 
//       name(n), alignment(a), bound(false), flags(f), global(false) {}
//     virtual ~Chunk() { for (Size i = 0; i < refs.size(); i++) delete refs[i]; }
//     void bind(Addr a) { address = a; bound = true; }
//     void setGlobal() { global = true; }
//     bool isGlobal() const { return global; }
//     std::string name;
//     Size alignment;
//     bool bound, global;
//     Addr address;
//     Word flags;
//     std::vector<Ref*> refs;    
//   };

//   class TextChunk : public Chunk {
//   public:
//     TextChunk(std::string n, Size a = 0, Word f = 0) 
//       : Chunk(n, a, f), instructions() {}

//     ~TextChunk() { 
//       for (Size i = 0; i < instructions.size(); i++) delete instructions[i]; 
//     }

//     std::vector<Instruction*> instructions;
//   };

//   class DataChunk : public Chunk {
//   public:
//     DataChunk(std::string n, Size a = 0, Word f = 0) 
//       : Chunk(n, a, f), size(0), contents() {}
//     Size size;
//     std::vector<Byte> contents; /* 0 to size bytes in length. */
//   };

//   class Obj {
//   public:
//     ~Obj() { for (Size i = 0; i < chunks.size(); i++) delete chunks[i]; }
//     std::vector<Chunk*> chunks;
//     Size entry;
//   };

//   class DynObj : public Obj {
//   public:
//     std::vector<std::string> deps;
//   };

//   class ObjReader {
//   public:
//     virtual Obj *read(std::istream &input) = 0;
//   private:
//   };

//   class ObjWriter {
//   public:
//     virtual void write(std::ostream &output, const Obj &o) = 0;
//   private:
//   };

//   class AsmReader : public ObjReader {
//   public:
//     AsmReader(ArchDef arch) :
//       wordSize(arch.getWordSize()), nRegs(arch.getNRegs()) {}
//     virtual Obj *read(std::istream &input);
//   private:
//     Size wordSize, nRegs;

//     // Operand type sequences indexed by argument class
//     enum ArgType {AT_END, AT_REG, AT_PREG, AT_LIT};
//     static ArgType operandtype_table[][4]; // ArgClass -> ArgType[arg_idx]
//   };
   
//   class HOFReader : public ObjReader {
//   public:
//     HOFReader(ArchDef &arch) : arch(arch) {}
//     Obj *read(std::istream &input);
//   private:
//     const ArchDef &arch;
//   };

//   class AsmWriter : public ObjWriter {
//   public:
//     AsmWriter(ArchDef arch): wordSize(arch.getWordSize()) {}
//     virtual void write(std::ostream &output, const Obj &obj);
//   private:
//     Size wordSize;
//   };
 
//   class HOFWriter : public ObjWriter {
//   public:
//     HOFWriter(ArchDef &arch) : arch(arch) {}
//     virtual void write(std::ostream &output, const Obj &obj);
//   private:
//     const ArchDef &arch;
//   };
}

#endif
