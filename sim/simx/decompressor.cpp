// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "decompressor.h"
#include "instr_trace.h"
#include <cstdint>
#include <cstring>
#include <iostream>

namespace vortex {

static inline uint32_t bit(uint32_t x, int b) { return (x >> b) & 1u; }
static inline uint32_t bits(uint32_t x, int hi, int lo) { return (x >> lo) & ((1u << (hi - lo + 1)) - 1u); }
static inline uint32_t sext_imm(uint32_t val, int width) {
    uint32_t m = 1u << (width - 1);
    return (val ^ m) - m;
}

// compressed register mapping: rd' (3 bits) -> x8..x15
static inline uint32_t rcp(uint32_t r3) { return 8u + r3; }

// Build I-type, S-type, U-type, R-type 32-bit encodings (RV32I)
static inline uint32_t ENCI(uint32_t imm12, uint32_t rs1, uint32_t funct3, uint32_t rd, uint32_t opcode) {
    return ((imm12 & 0xFFF) << 20) | ((rs1 & 31) << 15) | ((funct3 & 7) << 12) | ((rd & 31) << 7) | (opcode & 0x7F);
}
static inline uint32_t ENCR(uint32_t funct7, uint32_t rs2, uint32_t rs1, uint32_t funct3, uint32_t rd, uint32_t opcode) {
    return ((funct7 & 0x7F) << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | ((funct3 & 7) << 12) | ((rd & 31) << 7) | (opcode & 0x7F);
}
static inline uint32_t ENCS(uint32_t imm12, uint32_t rs2, uint32_t rs1, uint32_t funct3, uint32_t opcode) {
    uint32_t imm11_5 = (imm12 >> 5) & 0x7F;
    uint32_t imm4_0  = imm12 & 0x1F;
    return (imm11_5 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | ((funct3 & 7) << 12) | (imm4_0 << 7) | (opcode & 0x7F);
}
static inline uint32_t ENCU(uint32_t imm20, uint32_t rd, uint32_t opcode) {
    return ((imm20 & 0xFFFFF) << 12) | ((rd & 31) << 7) | (opcode & 0x7F);
}
static inline uint32_t ENCUJ(uint32_t imm21, uint32_t rd, uint32_t opcode) { // JAL
    // J-type bit shuffle: [20|10:1|11|19:12]
    uint32_t i = imm21 & 0x001FFFFF;
    uint32_t enc = ((i & (1<<20)) << 11)         // 20 -> 31
                 | ((i & 0x000007FE) << 20)      // 10:1 -> 30:21
                 | ((i & (1<<11)) << 9)          // 11 -> 20
                 | ((i & 0x000FF000));           // 19:12 -> 19:12
    return enc | ((rd & 31) << 7) | (opcode & 0x7F);
}
static inline uint32_t ENCB(uint32_t imm13, uint32_t rs2, uint32_t rs1, uint32_t funct3, uint32_t opcode) {
    // B-type: imm[12|10:5|4:1|11] -> [31|30:25|11:8|7]
    uint32_t i = imm13 & 0x1FFF;
    uint32_t imm12 = (i >> 12) & 1;
    uint32_t imm10_5 = (i >> 5) & 0x3F;
    uint32_t imm4_1 = (i >> 1) & 0xF;
    uint32_t imm11 = (i >> 11) & 1;
    uint32_t enc = (imm12 << 31) | (imm10_5 << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) |
                   ((funct3 & 7) << 12) | (imm4_1 << 8) | (imm11 << 7) | (opcode & 0x7F);
    return enc;
}

// --- main ---------------------------------------------------------------

DecompResult rvc_decompress(uint32_t word) {
    DecompResult out{};

    if ((word & 0x3) == 0x3) {
        out.instr32 = word;
        out.size    = 4;
        out.illegal = false;
        return out;
    }

    // 16-bit compressed
    const uint16_t h = static_cast<uint16_t>(word & 0xFFFF);
    const uint32_t quadrant = h & 0x3;              // 0,1,2
    const uint32_t funct3   = (h >> 13) & 0x7;

    out.size    = 2;
    out.illegal = false;

    switch (quadrant) {
    // ---------------- Quadrant 0 (opcode 00) ----------------
    case 0:
        switch (funct3) {
        case 0b000: { // C.ADDI4SPN -> ADDI rd', x2, nzimm
            uint32_t rd_ = rcp(bits(h, 4, 2));
            uint32_t nzuimm = (bits(h, 12, 11) << 4) | (bits(h,10, 7) << 6) | (bit(h, 5) << 3) | (bit(h, 6) << 2);
            if (nzuimm == 0) { out.illegal = true; break; }
            out.instr32 = ENCI(nzuimm, 2, 0b000, rd_, 0b0010011);
            break;
        }
        case 0b010: { // C.LW -> LW rd', offset(rs1')
            uint32_t rd_  = rcp(bits(h, 4, 2));
            uint32_t rs1_ = rcp(bits(h, 9, 7));
            uint32_t uimm = (bit(h, 5) << 6) | (bits(h, 12, 10) << 3) | (bit(h, 6) << 2);
            out.instr32 = ENCI(uimm, rs1_, 0b010, rd_, 0b0000011);
            break;
        }
        case 0b011: { // C.FLW -> FLW rd', offset(rs1')
            uint32_t rd_  = rcp(bits(h, 4, 2));
            uint32_t rs1_ = rcp(bits(h, 9, 7));
            uint32_t uimm = (bit(h, 5) << 6) | (bits(h, 12, 10) << 3) | (bit(h, 6) << 2);
            out.instr32 = ENCI(uimm, rs1_, 0b010, rd_, 0b0000111);
            break;
        }
        case 0b110: { // C.SW -> SW rs2', offset(rs1')
            uint32_t rs2_ = rcp(bits(h, 4, 2));
            uint32_t rs1_ = rcp(bits(h, 9, 7));
            uint32_t uimm = (bit(h, 5) << 6) | (bits(h, 12, 10) << 3) | (bit(h, 6) << 2);
            out.instr32 = ENCS(uimm, rs2_, rs1_, 0b010, 0b0100011);
            break;
        }
        case 0b111: { // C.FSW
            uint32_t rs2_ = rcp(bits(h, 4, 2));
            uint32_t rs1_ = rcp(bits(h, 9, 7));
            uint32_t uimm = (bit(h, 5) << 6) | (bits(h, 12, 10) << 3) | (bit(h, 6) << 2);
            out.instr32 = ENCS(uimm, rs2_, rs1_, 0b010, 0b0100111);
            break;
        }
        default:
            out.illegal = true; break;
        }
        break;

    // ---------------- Quadrant 1 (opcode 01) ----------------
    case 1:
        switch (funct3) {
        case 0b000: { // C.ADDI -> ADDI rd, rd, imm  (and C.NOP when rd=0,imm=0)
            uint32_t rd = bits(h, 11, 7);
            int32_t imm = static_cast<int32_t>(sext_imm(((bit(h,12)<<5) | bits(h,6,2)), 6));
            out.instr32 = ENCI(imm & 0xFFF, rd, 0b000, rd, 0b0010011);
            if (rd == 0 && imm == 0) {
                out.illegal = false;
                out.instr32 = ENCI(0, 0, 0b000, 0, 0b0010011); // C.NOP -> ADDI x0,x0,0
            }
            break;
        }
        case 0b001: { // C.JAL -> JAL x1, imm  (RV32 only)
            uint32_t imm =
                (bit(h,12)<<11) | (bit(h,8)<<10) | (bits(h,10,9)<<8) | (bit(h,6)<<7) |
                (bit(h,7)<<6) | (bit(h,2)<<5) | (bit(h,11)<<4) | (bits(h,5,3)<<1);
            int32_t simm = static_cast<int32_t>(sext_imm(imm, 12));
            uint32_t imm21 = static_cast<uint32_t>(simm) & 0x001FFFFF;
            out.instr32 = ENCUJ(imm21, 1, 0b1101111);
            break;
        }
        case 0b010: { // C.LI -> ADDI rd, x0, imm
            uint32_t rd = bits(h, 11, 7);
            int32_t imm = static_cast<int32_t>(sext_imm(((bit(h,12)<<5) | bits(h,6,2)), 6));
            if (rd == 0) { out.illegal = true; break; }
            out.instr32 = ENCI(imm & 0xFFF, 0, 0b000, rd, 0b0010011);
            break;
        }
        case 0b011: {
            uint32_t rd = bits(h, 11, 7);
            if (rd == 2) {
                // C.ADDI16SP -> ADDI x2, x2, imm
                int32_t imm = (bit(h,12)<<9) | (bit(h,6)<<4) | (bit(h,5)<<6) | (bits(h,4,3)<<7) | (bit(h,2)<<5);
                imm = static_cast<int32_t>(sext_imm(imm, 10));
                if (imm == 0) { out.illegal = true; break; }
                out.instr32 = ENCI(imm & 0xFFF, 2, 0b000, 2, 0b0010011);
            } else {
                // C.LUI -> LUI rd, imm (rd != x0, x2)
                int32_t imm = static_cast<int32_t>(sext_imm((bit(h,12)<<17) | (bits(h,6,2)<<12), 18));
                if (rd == 0 || rd == 2 || imm == 0) { out.illegal = true; break; }
                out.instr32 = ENCU((imm >> 12), rd, 0b0110111);
            }
            break;
        }
        case 0b100: {
            uint32_t subfunct = bits(h, 11, 10);
            if (subfunct == 0b00) { // C.SRLI
                uint32_t rd_ = rcp(bits(h,9,7));
                uint32_t sh  = (bit(h,12)<<5) | bits(h,6,2);
                out.instr32 = ENCI(sh, rd_, 0b101, rd_, 0b0010011);
            } else if (subfunct == 0b01) { // C.SRAI
                uint32_t rd_ = rcp(bits(h,9,7));
                uint32_t sh  = (bit(h,12)<<5) | bits(h,6,2);
                out.instr32 = ENCI(sh, rd_, 0b101, rd_, 0b0010011) | (0x40000000u);
            } else if (subfunct == 0b10) { // C.ANDI
                uint32_t rd_ = rcp(bits(h,9,7));
                int32_t imm = static_cast<int32_t>(sext_imm(((bit(h,12)<<5) | bits(h,6,2)), 6));
                out.instr32 = ENCI(imm & 0xFFF, rd_, 0b111, rd_, 0b0010011);
            } else { // 0b11: C.SUB/XOR/OR/AND (register form)
                uint32_t rd_  = rcp(bits(h,9,7));
                uint32_t rs2_ = rcp(bits(h,4,2));
                uint32_t op2  = bits(h, 6,5);
                switch (op2) {
                    case 0b00: out.instr32 = ENCR(0b0100000, rs2_, rd_, 0b000, rd_, 0b0110011); break; // C.SUB
                    case 0b01: out.instr32 = ENCR(0b0000000, rs2_, rd_, 0b100, rd_, 0b0110011); break; // C.XOR
                    case 0b10: out.instr32 = ENCR(0b0000000, rs2_, rd_, 0b110, rd_, 0b0110011); break; // C.OR
                    case 0b11: out.instr32 = ENCR(0b0000000, rs2_, rd_, 0b111, rd_, 0b0110011); break; // C.AND
                }
            }
            break;
        }
        case 0b101: { // C.J -> JAL x0, imm
            uint32_t imm =
                (bit(h,12)<<11) | (bit(h,8)<<10) | (bits(h,10,9)<<8) | (bit(h,6)<<7) |
                (bit(h,7)<<6) | (bit(h,2)<<5) | (bit(h,11)<<4) | (bits(h,5,3)<<1);
            int32_t simm = static_cast<int32_t>(sext_imm(imm, 12));
            uint32_t imm21 = static_cast<uint32_t>(simm) & 0x001FFFFF;
            out.instr32 = ENCUJ(imm21, 0, 0b1101111);
            break;
        }
        case 0b110: { // C.BEQZ
            uint32_t rs1_ = rcp(bits(h, 9,7));
            uint32_t imm = (bit(h,12)<<8) | (bit(h,6)<<7) | (bit(h,5)<<6) | (bit(h,2)<<5) | (bits(h,11,10)<<3) | (bits(h,4,3)<<1);
            int32_t simm = static_cast<int32_t>(sext_imm(imm, 9));
            uint32_t imm13 = static_cast<uint32_t>(simm) & 0x1FFF;
            out.instr32 = ENCB(imm13, 0, rs1_, 0b000, 0b1100011);
            break;
        }
        case 0b111: { // C.BNEZ
            uint32_t rs1_ = rcp(bits(h, 9,7));
            uint32_t imm = (bit(h,12)<<8) | (bit(h,6)<<7) | (bit(h,5)<<6) | (bit(h,2)<<5) | (bits(h,11,10)<<3) | (bits(h,4,3)<<1);
            int32_t simm = static_cast<int32_t>(sext_imm(imm, 9));
            uint32_t imm13 = static_cast<uint32_t>(simm) & 0x1FFF;
            out.instr32 = ENCB(imm13, 0, rs1_, 0b001, 0b1100011);
            break;
        }
        default:
            out.illegal = true; break;
        }
        break;

    // ---------------- Quadrant 2 (opcode 10) ----------------
    case 2:
        switch (funct3) {
        case 0b000: { // C.SLLI
            uint32_t rd = bits(h, 11, 7);
            uint32_t sh = (bit(h,12)<<5) | bits(h,6,2);
            if (rd == 0) { out.illegal = true; break; }
            out.instr32 = ENCI(sh, rd, 0b001, rd, 0b0010011);
            break;
        }
        case 0b010: { // C.LWSP
            uint32_t rd = bits(h, 11, 7);
            if (rd == 0) { out.illegal = true; break; }
            uint32_t uimm = (bit(h, 12) << 5) | (bits(h, 6, 4) << 2) | (bits(h, 3, 2) << 6);
            out.instr32 = ENCI(uimm, 2, 0b010, rd, 0b0000011);
            break;
        }
        case 0b100: {
            uint32_t rd  = bits(h, 11, 7);
            uint32_t rs2 = bits(h, 6, 2);
            uint32_t s12 = bit(h, 12);
            if (rs2 == 0) {
                if (s12 == 0) {
                    // C.JR
                    if (rd == 0) { out.illegal = true; break; }
                    out.instr32 = ENCI(0, rd, 0b000, 0, 0b1100111);
                } else {
                    if (rd == 0) {
                        // C.EBREAK
                        out.instr32 = 0x00100073;
                    } else {
                        // C.JALR
                        out.instr32 = ENCI(0, rd, 0b000, 1, 0b1100111);
                    }
                }
            } else {
                if (rd == 0) { out.illegal = true; break; }
                if (s12 == 0) {
                    // C.MV
                    out.instr32 = ENCR(0b0000000, rs2, 0, 0b000, rd, 0b0110011);
                } else {
                    // C.ADD
                    out.instr32 = ENCR(0b0000000, rs2, rd, 0b000, rd, 0b0110011);
                }
            }
            break;
        }
        case 0b110: { // C.SWSP
            uint32_t rs2 = bits(h, 6, 2);
            uint32_t imm = (bits(h, 12, 9) << 2) | (bits(h, 8, 7)  << 6);
            out.instr32 = ENCS(imm, rs2, 2, 0b010, 0b0100011);
            break;
        }
        default:
            out.illegal = true; break;
        }
        break;

    default:
        out.illegal = true; break;
    }
    if (out.illegal) {
        std::cerr << "Illegal 16-bit RVC: word=0x" << std::hex << (word & 0xFFFFu) << std::dec
                  << ", quadrant=" << quadrant << ", func3=" << funct3 << std::endl;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Decompressor — per-core RVC fetch SimObject
// ---------------------------------------------------------------------------

Decompressor::Decompressor(const SimContext& ctx, const char* name, uint32_t num_warps)
    : SimObject<Decompressor>(ctx, name)
    , state_(num_warps)
{}

Decompressor::~Decompressor() {}

void Decompressor::on_reset() {
    for (auto& s : state_) s = RvcSlot{};
    while (!refetch_queue_.empty()) refetch_queue_.pop();
}

Decompressor::Pick Decompressor::pick_request(instr_trace_t* fetch_latch_head) {
    Pick out{};
    if (!refetch_queue_.empty()) {
        // Cross-word completion in flight — drain its second fetch first.
        out.trace = refetch_queue_.front();
        const auto& rvc = state_.at(out.trace->wid);
        out.addr = (rvc.inst_pc & ~Word(3)) + 4;
        out.from_refetch = true;
    } else if (fetch_latch_head != nullptr) {
        out.trace = fetch_latch_head;
        out.addr  = fetch_latch_head->PC & ~Word(3);
        out.from_refetch = false;
    }
    return out;
}

void Decompressor::commit_request(bool from_refetch) {
    if (from_refetch && !refetch_queue_.empty()) {
        refetch_queue_.pop();
    }
}

Word Decompressor::fetch_addr(const instr_trace_t* trace) const {
    const auto& rvc = state_.at(trace->wid);
    if (rvc.needs_second) {
        return (rvc.inst_pc & ~Word(3)) + 4;
    }
    return trace->PC & ~Word(3);
}

bool Decompressor::on_icache_rsp(instr_trace_t* trace, const mem_block_t& line) {
    // Extract the 4-byte word at the address we fetched. fetch_addr()
    // factors in whether this rsp is a refetch completion (high half of a
    // cross-word 32b) or a fresh fetch.
    uint32_t line_offset = fetch_addr(trace) & (MEM_BLOCK_SIZE - 1);
    uint32_t word = 0;
    std::memcpy(&word, line.data() + line_offset, sizeof(uint32_t));

    auto& rvc = state_.at(trace->wid);

    if (rvc.needs_second) {
        // Combine: low half from buffer, high half = low16 of new word.
        uint16_t high = word & 0xFFFFu;
        trace->code = (uint32_t(high) << 16) | rvc.low_half;
        rvc = RvcSlot{};
        return true;
    }

    // Halfword at PC depends on PC[1].
    uint16_t hword = (trace->PC & Word(2)) ? uint16_t(word >> 16)
                                            : uint16_t(word & 0xFFFFu);
    if ((hword & 0x3u) != 0x3u) {
        // 16-bit RVC: emit the raw hword zero-extended; the decoder
        // detects RVC from code[1:0] and decompresses internally.
        trace->code = uint32_t(hword);
        return true;
    }
    if ((trace->PC & Word(2)) == 0) {
        // 4-byte aligned 32-bit: full word at PC is the instruction.
        trace->code = word;
        return true;
    }
    // 2-byte aligned 32-bit: low half = high16 of fetched word; high half
    // lives in the next aligned word — buffer + queue for refetch.
    rvc.needs_second = true;
    rvc.low_half     = uint16_t(word >> 16);
    rvc.inst_pc      = trace->PC;
    refetch_queue_.push(trace);
    return false;
}

} // namespace vortex
