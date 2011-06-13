/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#include "include/types.h"
#include "include/obj.h"
#include "include/util.h"
#include "include/asm-tokens.h"

#include <iostream>
#include <stdlib.h>
#include <FlexLexer.h>

#include <cctype>
#include <cstdio>

#include <map>

using namespace std;
using namespace Harp;
using namespace HarpTools;

extern struct rval_t { std::string s; uint64_t u; } yylval;
extern unsigned yyline;

static void asmReaderError(unsigned line, const string &message) {
  cout << "Assembly reader error, line " << line << ": " << message << '\n';
  exit(1);
}

static int skip_parens(const string &s, int i) {
  int paren_level = 1;
  do {
    i++;
    if (s[i] == '(') paren_level++;
    if (s[i] == ')') paren_level--;
  } while (paren_level > 0);

  return i;
}

// Probably the worst recursive descent parser ever written, but it's an easy 
// way to make our assembly language pretty.
static uint64_t readParenExpression(const string &s, const map<string, Word> &d,
                                    int start=0, int end=-1)
{
  uint64_t (* const rPE)(const string&, const map<string, Word>&, int, int) 
    = readParenExpression;

  if (end==-1) end = s.length();
  
  while (isspace(s[start])) start++;
  while (isspace(s[end-1])) end--;

  for (int i = start; i < end; i++) {
    if (s[i] == '(') { i = skip_parens(s, i); continue; }
    
    if (s[i] == '<') return rPE(s, d, start, i) << rPE(s, d, i+2, end);
    if (s[i] == '>') return rPE(s, d, start, i) >> rPE(s, d, i+2, end);
  }

  for (int i = start; i < end; i++) {
    if (s[i] == '(') { i = skip_parens(s, i); continue; }
    if (s[i] == '+') return rPE(s, d, start, i) + rPE(s, d, i+1, end);
    if (s[i] == '-') return rPE(s, d, start, i) - rPE(s, d, i+1, end);
    if (s[i] == '|') return rPE(s, d, start, i) | rPE(s, d, i+1, end);
    if (s[i] == '^') return rPE(s, d, start, i) ^ rPE(s, d, i+1, end);
  }

  for (int i = start; i < end; i++) {
    if (s[i] == '(') { i = skip_parens(s, i); continue; }
    if (s[i] == '*') return rPE(s, d, start, i) * rPE(s, d, i+1, end);
    if (s[i] == '/') return rPE(s, d, start, i) / rPE(s, d, i+1, end);
    if (s[i] == '%') return rPE(s, d, start, i) % rPE(s, d, i+1, end);
    if (s[i] == '&') return rPE(s, d, start, i) & rPE(s, d, i+1, end);
  } 

  if (isdigit(s[start])) {
    unsigned long long u;
    sscanf(s.substr(start, end-start).c_str(), "%lli", &u);
    return u;
  }

  if (s[start] == '(') return rPE(s, d, start+1, end-1);

  map<string, Word>::const_iterator it = d.find(s.substr(start, end-start));
  if (it != d.end()) return it->second;

  cout << "TODO: Error message.\n";
  exit(1);
}

Obj *AsmReader::read(std::istream &input) {
  FlexLexer *f = new yyFlexLexer(&input);
  Obj *o = new Obj();
  std::vector<Chunk>::reverse_iterator cur;
  bool permR(true), permW(false), permX(false), entry(false), nextPred(false),
       global(false);

  map <string, Word> defs;

  /* Pre-defined defs. */
  defs["__WORD"] = wordSize;

  map <string, Instruction::Opcode> opMap;

  // Build opMap
  for (size_t i = 0; Instruction::opStrings[i]; i++)
    opMap[std::string(Instruction::opStrings[i])] = Instruction::Opcode(i);

  enum { 
    ST_INIT, ST_DEF1, ST_DEF2, ST_PERM, ST_WORD1, ST_WORD2, ST_STRING1,
    ST_STRING2, ST_BYTE1, ST_BYTE2, ST_ALIGN, ST_INST1, ST_INST2 
  } state(ST_INIT);

  enum { OS_NOCHUNK, OS_TEXTCHUNK, OS_DATACHUNK } outstate(OS_NOCHUNK);
  TextChunk *tc;
  DataChunk *dc;
  Instruction *curInst;
  string string_arg, next_chunk_name;
  Size next_chunk_align(0);
  uint64_t num_arg;
  RegNum nextPredNum;

  AsmTokens t;
  while ((t = (AsmTokens)f->yylex()) != 0) {
    switch (t) {
      case ASM_T_DIR_DEF:
        if (state == ST_INIT) state = ST_DEF1; 
        else { asmReaderError(yyline, "Unexpected .def"); }
        break;
      case ASM_T_DIR_PERM:
        if (state == ST_INIT) {
          state = ST_PERM;
          permR = permW = permX = false;
          if (outstate != OS_NOCHUNK) {
            outstate = OS_NOCHUNK;
            entry = false;
            global = false;
          }
        } else { asmReaderError(yyline, "Unexpected .perm"); }
        break;
      case ASM_T_DIR_BYTE:
        if (state == ST_INIT) {
          state = ST_BYTE1;
        } else { asmReaderError(yyline, "Unexpected .byte"); }
        break;
      case ASM_T_DIR_WORD:
        if (state == ST_INIT) {
          state = ST_WORD1;
        } else { asmReaderError(yyline, "Unexpected .word"); }
        break;
      case ASM_T_DIR_STRING:
        if (state == ST_INIT) {
          state = ST_STRING1;
        } else { asmReaderError(yyline, "Unexpected .string"); }
        break;
      case ASM_T_DIR_ALIGN:
        if (state == ST_INIT) {
          state = ST_ALIGN;
        } else { asmReaderError(yyline, "Unexpected .align"); }
        break;
      case ASM_T_DIR_ENTRY:
        outstate = OS_NOCHUNK;
        entry = true;
        break;
      case ASM_T_DIR_GLOBAL:
        outstate = OS_NOCHUNK;
        global = true;
        break;
      case ASM_T_DIR_ARG_NUM:
        switch (state) {
          case ST_DEF2: defs[string_arg] = yylval.u; state = ST_INIT; break;
          case ST_WORD1: dc->size += (yylval.u)*wordSize;
                         state = ST_INIT;
                         break;
          case ST_WORD2: if (outstate == OS_DATACHUNK) {
                           dc->size += wordSize;
                           dc->contents.resize(dc->size);
                           wordToBytes(&*(dc->contents.end()-wordSize), 
                                       yylval.u, wordSize);
                         } else {
                           asmReaderError(yyline, "Word not in data chunk"
                                                  "(internal error)");
                         }
                         state = ST_INIT;
                         break;
          case ST_BYTE1: dc->size += yylval.u;
                         state = ST_INIT;
                         break;
          case ST_BYTE2: if (outstate == OS_DATACHUNK) {
                           dc->size++;
                           dc->contents.resize(dc->size);
                           *(dc->contents.end() - 1) = yylval.u;
                         } else {
                           asmReaderError(yyline, "Byte not in data chunk"
                                                  "(internal error)");
                         }
                         state = ST_INIT;
                         break;
          case ST_ALIGN: next_chunk_align = yylval.u;
                         if (outstate != OS_NOCHUNK) {
                           outstate = OS_NOCHUNK;
                           entry = false;
                           global = false;
                         }
                         state = ST_INIT;
                         break;
          default: asmReaderError(yyline, "Unexpected literal argument");
        }
        break;
      case ASM_T_DIR_ARG_STRING:
        if (state == ST_STRING2 || 
             (state == ST_STRING1 && outstate == OS_DATACHUNK)) 
        {
          const char *s = yylval.s.c_str();
          do {
            if (*s == '\\') {
              switch (*(++s)) {
                case 'n': dc->contents.push_back('\n');   break;
                case '"': dc->contents.push_back(*s); break;
                default:  dc->contents.push_back(*s); break;
              }
            } else {
              dc->contents.push_back(*s);
            }
            dc->size++;
          } while(*(s++));
        } else {
          asmReaderError(yyline, "Unexpected string literal.");
        }
        state = ST_INIT;
        break;
      case ASM_T_DIR_ARG_SYM:
        switch (state) {
          case ST_DEF1: string_arg = yylval.s; state = ST_DEF2; break;
          case ST_WORD1: {
            state = ST_WORD2; 
            outstate = OS_DATACHUNK;
            dc = new DataChunk(yylval.s, next_chunk_align?
                                           next_chunk_align:wordSize, 
                               flagsToWord(permR, permW, permX));
            next_chunk_align = 0;
            o->chunks.push_back(dc);
            if (entry) o->entry = o->chunks.size() - 1;
            if (global) dc->setGlobal();
          } break;
          case ST_BYTE1: {
            state = ST_BYTE2;
            outstate = OS_DATACHUNK;
            dc = new DataChunk(yylval.s, next_chunk_align,
                               flagsToWord(permR, permW, permX));
            next_chunk_align = 0;
            o->chunks.push_back(dc);
            if (entry) o->entry = o->chunks.size() - 1;
            if (global) dc->setGlobal();
          } break;
          case ST_STRING1: {
            state = ST_STRING2;
            outstate = OS_DATACHUNK;
            dc = new DataChunk(yylval.s, next_chunk_align,
                               flagsToWord(permR, permW, permX));
            next_chunk_align = 0;
            o->chunks.push_back(dc);
            if (entry) o->entry = o->chunks.size() - 1;
            if (global) dc->setGlobal();
          } break;
          default: asmReaderError(yyline, "");
        };
        break;
      case ASM_T_DIR_ARG_R:
        permR = true;
        break;
      case ASM_T_DIR_ARG_W:
        permW = true;
        break;
      case ASM_T_DIR_ARG_X:
        permX = true;
        break;
      case ASM_T_DIR_END:
        if (state == ST_INST1 || state == ST_INST2) {
          if (outstate == OS_TEXTCHUNK) {
            tc->instructions.push_back(curInst);
          } else {
            asmReaderError(yyline, "Inst not in text chunk(internal error)");
          }
        }
        state = ST_INIT;
        break;
      case ASM_T_LABEL:
        if (outstate != OS_NOCHUNK) {
          entry = false;
          global = false;
          outstate = OS_NOCHUNK;
        }
        next_chunk_name = yylval.s;
        break;
      case ASM_T_PRED:
        nextPred = true;
        nextPredNum = yylval.u;
        break;
      case ASM_T_INST:
        if (state == ST_INIT) {
          Instruction::Opcode opc = opMap[yylval.s];
          if (outstate != OS_TEXTCHUNK) {
            tc = new TextChunk(next_chunk_name, next_chunk_align,
                               flagsToWord(permR, permW, permX));
            next_chunk_align = 0;
            o->chunks.push_back(tc);
            if (entry) o->entry = o->chunks.size() - 1;
            if (global) tc->setGlobal();
            outstate = OS_TEXTCHUNK;
          }
          curInst = new Instruction();
          curInst->setOpcode(opc);
          if (nextPred) {
            nextPred = false;
            curInst->setPred(nextPredNum);
          }
          state = Instruction::allSrcArgs[opc]?ST_INST2:ST_INST1;
        } else { asmReaderError(yyline, "Unexpected token"); }
        break;
      case ASM_T_PREG:
        switch (state) {
          case ST_INST1: curInst->setDestPReg(yylval.u);
                         state = ST_INST2;
                         break;
          case ST_INST2: curInst->setSrcPReg(yylval.u);
                         break;
          default: asmReaderError(yyline, "Unexpected predicate register");
        }
        break;
      case ASM_T_REG:
        switch (state) {
          case ST_INST1: curInst->setDestReg(yylval.u);
                         state = ST_INST2;
                         break;
          case ST_INST2: curInst->setSrcReg(yylval.u);
                         break;
          default: asmReaderError(yyline, "Unexpected register");
        }
        break;
      case ASM_T_PEXP:
        // Decode the paren expression.
        yylval.u = readParenExpression(yylval.s, defs);
      case ASM_T_LIT:
        switch (state) {
          case ST_INST1: asmReaderError(yyline, "Unexpected literal");
          case ST_INST2: curInst->setSrcImm(yylval.u);
                         break;
          default: asmReaderError(yyline, "Unexpected literal");
        }
        break;
      case ASM_T_SYM:
        switch (state) {
          case ST_INST1: asmReaderError(yyline, "Unexpected symbol");
          case ST_INST2: if (defs.find(yylval.s) != defs.end()) {
                           curInst->setSrcImm(defs[yylval.s]);
                         } else {
                           Ref *r = new 
                             SimpleRef(yylval.s, *curInst->setSrcImm(),
                               curInst->hasRelImm());
                           tc->refs.push_back(r);
                           curInst->setImmRef(*r);
                         }
                         break;
          default: asmReaderError(yyline, "Unexpected symbol");
        }
        break;
      default: asmReaderError(yyline, "Invalid state(internal error)");
    };
  }

  return o;
}

void AsmWriter::write(std::ostream &output, const Obj &obj) {
  unsigned anonWordNum(0);

  Word prevFlags(0);

  for (size_t j = 0; j < obj.chunks.size(); j++) {
    Chunk * const &c = obj.chunks[j];

    /* Write out the flags. */
    if (c->flags != prevFlags) { 
      bool r, w, x;
      wordToFlags(r, w, x, c->flags);
      output << ".perm ";
      if (r) output << 'r';
      if (w) output << 'w';
      if (x) output << 'x';
      output << '\n';
      prevFlags = c->flags;
    }

    /* Write align if set. */
    if (c->alignment) output << ".align 0x" << hex << c->alignment << '\n';

    TextChunk * const tc = dynamic_cast<TextChunk* const>(c);
    DataChunk * const dc = dynamic_cast<DataChunk* const>(c);

    if (tc) {
      if (j == obj.entry) output << "\n.entry\n";
      if (c->isGlobal()) output << "\n.global\n";
      if (tc->name != "") output << tc->name << ':';

      for (size_t i = 0; i < tc->instructions.size(); i++) {
        output << "\t" << *(tc->instructions[i]) << '\n';
      }
    } else if (dc) {
      Size i;
      for (i = 0; i < dc->contents.size();) {
        Size tmpWordSize = (dc->contents.size() - i < wordSize) ? 
                             dc->contents.size() - i : wordSize;

        i += tmpWordSize;
        Word w = 0;
        for (size_t j = 0; j < tmpWordSize; j++) {
          w <<= 8;
          w |= dc->contents[i - j - 1];
        }

        if (i == tmpWordSize && c->name != "") {
          output << ".word " << c->name << " 0x" << hex << w << '\n';
        } else {
          output << ".word __anonWord" << anonWordNum++ << " 0x" 
                 << hex << w << '\n';
        }
      }

      if (i % wordSize) i += (wordSize - (i%wordSize));

      if (dc->size > i) {
        Size fillSize = (dc->size - i)/wordSize;
        output << ".word 0x" << hex << fillSize << '\n';
      }
    } else {
      cout << "Unrecognized chunk type in AsmWriter.\n";
      exit(1);
    }
  }
}

enum HOFFlag { HOF_GLOBAL = 1 };

Word getHofFlags(Chunk &c) {
  Word w = 0;
  if (c.isGlobal()) w |= HOF_GLOBAL;
  return w;
}

static void outputWord(std::ostream &out, Word w, 
                       vector<Byte> &tmp, Size wordSize) 
{
  Size n(0);
  writeWord(tmp, n, wordSize, w);
  out.write((char*)&tmp[0], wordSize);
}

void HOFWriter::write(std::ostream &output, const Obj &obj) {
  string archString(arch);
  Size wordSize(arch.getWordSize()), n, offsetVectorPos;

  vector<Byte> tmp;
  vector<Size> offsets(obj.chunks.size());

  /* Magic number, arch string, and padding. */
  output.write("HARP", 4);
  output.write(archString.c_str(), archString.length()+1); 
  Size padBytes = (wordSize-(4+archString.length()+1)%wordSize)%wordSize;
  for (Size i = 0; i < padBytes; i++) output.put(0);

  /* Write out the entry chunk index. */
  outputWord(output, obj.entry, tmp, wordSize);

  /* Write out the number of chunks. */
  outputWord(output, obj.chunks.size(), tmp, wordSize);

  /* Skip the chunk size offset vector. */
  offsetVectorPos = output.tellp();
  output.seekp(output.tellp() + streampos(wordSize * obj.chunks.size()));

  /* Write out the chunks, keeping track of their offsets. */
  for (Size i = 0; i < obj.chunks.size(); i++) {
    offsets[i] = output.tellp();

    /* Chunk name */
    DataChunk *dc = dynamic_cast<DataChunk*>(obj.chunks[i]);
    if (!dc) { cout << "HOFWriter::write(): invalid chunk type.\n"; exit(1); }
    output.write(dc->name.c_str(), dc->name.length() + 1);

    /* Padding */
    padBytes = (wordSize - (dc->name.length()+1)%wordSize)%wordSize;
    for (Size i = 0; i < padBytes; i++) output.put(0);

    /* Chunk alignment, flags, address, size (in RAM and disk) */
    outputWord(output, dc->alignment, tmp, wordSize);
    outputWord(output, dc->flags, tmp, wordSize);
    outputWord(output, getHofFlags(*dc), tmp, wordSize);
    outputWord(output, dc->bound?dc->address:0, tmp, wordSize);
    outputWord(output, dc->size, tmp, wordSize);
    outputWord(output, dc->contents.size(), tmp, wordSize);

    /* References */
    outputWord(output, dc->refs.size(), tmp, wordSize);
    for (Size j = 0; j < dc->refs.size(); j++) {
      OffsetRef *r = dynamic_cast<OffsetRef*>(dc->refs[j]);
      if (!r) { cout << "HOFWriter::write(): invalid ref type.\n"; exit(1); }
      /* Reference name */
      output.write(r->name.c_str(), r->name.length() + 1);
      /* Padding */
      padBytes = (wordSize - (r->name.length() + 1)%wordSize)%wordSize;
      for (Size i = 0; i < padBytes; i++) output.put(0);
      /* Compute flags word. */
      Word rFlags(0);
      if (r->relative) rFlags |= 1;
      /* Output flags word. */
      outputWord(output, rFlags, tmp, wordSize);
      /* Offset from which relative branches are computed. */
      outputWord(output, r->ibase, tmp, wordSize);
      /* Reference offset in block. */
      outputWord(output, r->getOffset(), tmp, wordSize);
      /* Reference size in bits. */
      outputWord(output, r->getBits(), tmp, wordSize);
    }

    /* Chunk data. */
    output.write((char*)&(dc->contents[0]), dc->contents.size());

    /* Chunk padding. */
    padBytes = (wordSize - dc->contents.size()%wordSize)%wordSize;
    for (Size i = 0; i < padBytes; i++) output.put(0);
  }

  /* Write out the chunk offset vector. */
  output.seekp(offsetVectorPos);
  for (Size i = 0; i < obj.chunks.size(); i++) {
    outputWord(output, offsets[i], tmp, wordSize);
  }
}

static Word inputWord(std::istream &input, Size wordSize, vector<Byte> &tmp) {
  Size n(0), pos(input.tellg());
  if (tmp.size() < wordSize) tmp.resize(wordSize);

  /* Seek to the next word-aligned place. */
  if (input.tellg()%wordSize) {
    input.seekg(input.tellg() + 
                streampos((wordSize - input.tellg()%wordSize)%wordSize));
  }

  input.read((char*)&tmp[0], wordSize);

  return readWord(tmp, n, wordSize);
}

static string inputString(std::istream &input) {
  string s;
  char c;

  while (input && (c = input.get()) != '\0') s += c;

  return s;
}

Obj *HOFReader::read(std::istream &input) {
  Size wordSize(arch.getWordSize());
  Obj *o = new Obj();

  vector<Byte> tmp(4);

  input.read((char*)&tmp[0], 4);
  if (tmp[0] != 'H' || tmp[1] != 'A' || tmp[2] != 'R' || tmp[3] != 'P') {
    cout << "Bad magic number in HOFReader::read().\n";
    exit(1);
  }

  string archString(inputString(input));
  ArchDef fileArch(archString);
  if (fileArch != arch) {
    cout << "File arch does not match reader arch in HOFReader::read().\n";
    exit(1);
  }

  o->entry = inputWord(input, wordSize, tmp);

  Size nChunks(inputWord(input, wordSize, tmp));

  vector<Size> chunkOffsets(nChunks);

  /* Read in the chunk offsets. */
  for (Size i = 0; i < nChunks; i++) {
    chunkOffsets[i] = inputWord(input, wordSize, tmp);
  }

  /* Read in the chunks. */
  o->chunks.resize(nChunks);
  for (Size i = 0; i < nChunks; i++) {
    input.seekg(chunkOffsets[i]);
    string name(inputString(input));
    Word alignment(inputWord(input, wordSize, tmp)), 
         flags(inputWord(input, wordSize, tmp)), 
         hofFlags(inputWord(input, wordSize, tmp)),
         address(inputWord(input, wordSize, tmp)), 
         size(inputWord(input, wordSize, tmp)), 
         dSize(inputWord(input, wordSize, tmp)),
         nRefs(inputWord(input, wordSize, tmp));
    DataChunk *dc = new DataChunk(name, alignment, flags);
    if (hofFlags & HOF_GLOBAL) dc->setGlobal();
    dc->address = address;
    dc->bound = address?true:false;
    dc->contents.resize(dSize);

    /* Get the refs. */
    for (Size j = 0; j < nRefs; j++) {
      string rName(inputString(input));
      Word rFlags(inputWord(input, wordSize, tmp)),
           ibase(inputWord(input, wordSize, tmp)),
           offset(inputWord(input, wordSize, tmp)),
           bits(inputWord(input, wordSize, tmp));
      OffsetRef *r =
        new OffsetRef(rName, dc->contents, offset, bits, wordSize, rFlags&1, 
                      ibase);
      dc->refs.push_back(r);
    }

    /* Get the contents. */
    input.read((char*)&dc->contents[0], dSize);
    dc->size = size;

    o->chunks[i] = dc;
  }
  
  return o;
}
