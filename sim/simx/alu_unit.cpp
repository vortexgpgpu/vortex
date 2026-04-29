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

#include "alu_unit.h"
#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include "debug.h"
#include "core.h"
#include "scheduler.h"
#include "constants.h"

using namespace vortex;

AluUnit::AluUnit(const SimContext& ctx, const char* name, Core* core)
	: FuncUnit<NUM_ALU_BLOCKS>(ctx, name, core)
{}

uint32_t AluUnit::latency_of(const instr_trace_t* trace) const {
	if (std::get_if<AluType>(&trace->op_type)) {
		auto alu_type = std::get<AluType>(trace->op_type);
		switch (alu_type) {
		case AluType::LUI:
		case AluType::AUIPC:
		case AluType::ADD:
		case AluType::SUB:
		case AluType::SLL:
		case AluType::SRL:
		case AluType::SRA:
		case AluType::SLT:
		case AluType::SLTU:
		case AluType::XOR:
		case AluType::AND:
		case AluType::OR:
		case AluType::CZERO:
			return 2;
		default:
			std::abort();
		}
	} else if (std::get_if<VoteType>(&trace->op_type)) {
		return 2;
	} else if (std::get_if<ShflType>(&trace->op_type)) {
		return 2;
	} else if (std::get_if<WgatherType>(&trace->op_type)) {
		return 2;
	} else if (std::get_if<BrType>(&trace->op_type)) {
		auto br_type = std::get<BrType>(trace->op_type);
		switch (br_type) {
		case BrType::BR:
		case BrType::JAL:
		case BrType::JALR:
		case BrType::SYS:
			return 2;
		default:
			std::abort();
		}
	} else if (std::get_if<MdvType>(&trace->op_type)) {
		auto mdv_type = std::get<MdvType>(trace->op_type);
		switch (mdv_type) {
		case MdvType::MUL:
		case MdvType::MULHU:
		case MdvType::MULH:
		case MdvType::MULHSU:
			return 2;
		case MdvType::DIV:
		case MdvType::DIVU:
		case MdvType::REM:
		case MdvType::REMU:
			return XLEN+2;
		default:
			std::abort();
		}
	}
	std::abort();
}

void AluUnit::execute(instr_trace_t* trace) {
	auto& sched = core_->scheduler();
	auto& warp = sched.warp(trace->wid);
	// Use trace->tmask captured at issue for per-thread active checks.
	// `warp.tmask` is the live warp state and may change due to divergent
	// control flow before this trace executes; the rest of the pipeline
	// (commit, writeback) keys off trace->tmask, so a mismatch leaves lanes
	// with stale dst_data. See LsuUnit::execute for the same pattern.
	auto& tmask = trace->tmask;
	auto& instr = *trace->instr_ptr;
	auto instrArgs = instr.get_args();
	uint32_t num_threads = NUM_THREADS;
	auto& rs1_data = trace->src_data[0];
	auto& rs2_data = trace->src_data[1];
	auto& rs3_data = trace->src_data[2];

	// derive thread bounds from operand mask
	uint32_t thread_start = 0;
	for (; thread_start < num_threads; ++thread_start) {
		if (tmask.test(thread_start))
			break;
	}
	int32_t thread_last = num_threads - 1;
	for (; thread_last >= 0; --thread_last) {
		if (tmask.test(thread_last))
			break;
	}

	bool is_w_enabled = false;
#ifdef XLEN_64
	is_w_enabled = true;
#endif

	// always size dst_data so per-op compute can write unconditionally;
	// commit_writeback gates regfile actual write on wb (skipped for x0 dst).
	trace->dst_data.assign(num_threads, reg_data_t{});
	auto& rd_data = trace->dst_data;

	if (std::get_if<AluType>(&trace->op_type)) {
		auto alu_type = std::get<AluType>(trace->op_type);
		auto aluArgs = std::get<IntrAluArgs>(instrArgs);
		Word imm = sext<Word>(aluArgs.imm, 32);
		switch (alu_type) {
		case AluType::LUI: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				rd_data[t].i = imm;
			}
		} break;
		case AluType::AUIPC: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				rd_data[t].i = imm + trace->PC;
			}
		} break;
		case AluType::ADD: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				if (is_w_enabled && aluArgs.is_w) {
					auto result = rs1_data[t].i32 + (int32_t)(aluArgs.is_imm ? aluArgs.imm : rs2_data[t].i32);
					rd_data[t].i = sext((uint64_t)result, 32);
				} else {
					rd_data[t].i = rs1_data[t].i + (aluArgs.is_imm ? imm : rs2_data[t].i);
				}
			}
		} break;
		case AluType::SUB: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				if (is_w_enabled && aluArgs.is_w) {
					auto result = rs1_data[t].i32 - (int32_t)(aluArgs.is_imm ? aluArgs.imm : rs2_data[t].i32);
					rd_data[t].i = sext((uint64_t)result, 32);
				} else {
					rd_data[t].i = rs1_data[t].i - (aluArgs.is_imm ? imm : rs2_data[t].i);
				}
			}
		} break;
		case AluType::SLT: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				rd_data[t].i = rs1_data[t].i < (aluArgs.is_imm ? WordI(imm) : rs2_data[t].i);
			}
		} break;
		case AluType::SLTU: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				rd_data[t].i = rs1_data[t].u < (aluArgs.is_imm ? imm : rs2_data[t].u);
			}
		} break;
		case AluType::SLL: {
			Word shamt_mask = (Word(1) << log2up(XLEN)) - 1;
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				if (is_w_enabled && aluArgs.is_w) {
					uint32_t shamt = (aluArgs.is_imm ? aluArgs.imm : rs2_data[t].i32) & shamt_mask;
					uint32_t result = (uint32_t)rs1_data[t].i << shamt;
					rd_data[t].i = sext((uint64_t)result, 32);
				} else {
					Word shamt = (aluArgs.is_imm ? imm : rs2_data[t].i) & shamt_mask;
					rd_data[t].i = rs1_data[t].i << shamt;
				}
			}
		} break;
		case AluType::SRA: {
			Word shamt_mask = (Word(1) << log2up(XLEN)) - 1;
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				if (is_w_enabled && aluArgs.is_w) {
					uint32_t shamt = (aluArgs.is_imm ? aluArgs.imm : rs2_data[t].i32) & shamt_mask;
					uint32_t result = (int32_t)rs1_data[t].i >> shamt;
					rd_data[t].i = sext((uint64_t)result, 32);
				} else {
					Word shamt = (aluArgs.is_imm ? imm : rs2_data[t].i) & shamt_mask;
					rd_data[t].i = rs1_data[t].i >> shamt;
				}
			}
		} break;
		case AluType::SRL: {
			Word shamt_mask = (Word(1) << log2up(XLEN)) - 1;
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				if (is_w_enabled && aluArgs.is_w) {
					uint32_t shamt = (aluArgs.is_imm ? aluArgs.imm : rs2_data[t].i32) & shamt_mask;
					uint32_t result = (uint32_t)rs1_data[t].i >> shamt;
					rd_data[t].i = sext((uint64_t)result, 32);
				} else {
					Word shamt = (aluArgs.is_imm ? imm : rs2_data[t].i) & shamt_mask;
					rd_data[t].i = rs1_data[t].u >> shamt;
				}
			}
		} break;
		case AluType::AND: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				rd_data[t].i = rs1_data[t].i & (aluArgs.is_imm ? imm : rs2_data[t].i);
			}
		} break;
		case AluType::OR: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				rd_data[t].i = rs1_data[t].i | (aluArgs.is_imm ? imm : rs2_data[t].i);
			}
		} break;
		case AluType::XOR: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				rd_data[t].i = rs1_data[t].i ^ (aluArgs.is_imm ? imm : rs2_data[t].i);
			}
		} break;
		case AluType::CZERO: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				bool cond = (rs2_data[t].i == 0) ^ aluArgs.imm;
				rd_data[t].i = cond ? 0 : rs1_data[t].i;
			}
		} break;
		default:
			std::abort();
		}
		DT(3, this->name() << " execute: op=" << alu_type << ", " << *trace);
	} else if (std::get_if<VoteType>(&trace->op_type)) {
		auto vote_type = std::get<VoteType>(trace->op_type);
		bool has_vote_true = false;
		bool has_vote_false = false;
		Word ballot = 0;
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			auto is_pred = rs1_data[t].i & 0x1;
			if (is_pred) {
				has_vote_true = true;
				ballot |= (Word(1) << t);
			} else {
				has_vote_false = true;
			}
		}
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			switch (vote_type) {
			case VoteType::ALL: rd_data[t].i = !has_vote_false; break;
			case VoteType::ANY: rd_data[t].i = has_vote_true; break;
			case VoteType::UNI: rd_data[t].i = !has_vote_true || !has_vote_false; break;
			case VoteType::BAL: rd_data[t].i = ballot; break;
			default: std::abort();
			}
		}
	} else if (std::get_if<ShflType>(&trace->op_type)) {
		auto shfl_type = std::get<ShflType>(trace->op_type);
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			auto bc  = rs2_data[t].i;
			int bval = (bc >>  0) & 0x3f;
			int cval = (bc >>  6) & 0x3f;
			int mask = (bc >> 12) & 0x3f;
			int maxLane = (t & mask) | (cval & ~mask);
			int minLane = (t & mask);
			int lane = 0;
			int pval = 0;
			switch (shfl_type) {
			case ShflType::UP:   lane = t - bval; pval = (lane >= minLane); break;
			case ShflType::DOWN: lane = t + bval; pval = (lane <= maxLane); break;
			case ShflType::BFLY: lane = t ^ bval; pval = (lane <= maxLane); break;
			case ShflType::IDX:  lane = minLane | (bval & ~mask); pval = (lane <= maxLane); break;
			default: std::abort();
			}
			if (!pval) lane = t;
			if (lane < (int)num_threads) {
				rd_data[t].i = rs1_data[lane].i;
			} else {
				rd_data[t].i = rs1_data[t].i;
			}
		}
	} else if (std::get_if<WgatherType>(&trace->op_type)) {
		// Each group of 4 lanes operates independently; source lane within a
		// group is suppressed by clearing its tmask bit so the standard
		// writeback path skips it (regfile keeps its prior value).
		auto wgArgs = std::get<IntrWgatherArgs>(instrArgs);
		uint32_t src_offset = wgArgs.src_lane;
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			if ((t & 0x3u) == src_offset) {
				trace->tmask.reset(t); // suppress writeback for source lane
				continue;
			}
			uint32_t group_base = t & ~0x3u;
			uint32_t sl         = group_base + src_offset;
			uint32_t offset     = (t - sl) & 0x3u;
			if      (offset == 1) rd_data[t].i = rs1_data[sl].i;
			else if (offset == 2) rd_data[t].i = rs2_data[sl].i;
			else if (offset == 3) rd_data[t].i = rs3_data[sl].i;
		}
	} else if (std::get_if<BrType>(&trace->op_type)) {
		auto br_type = std::get<BrType>(trace->op_type);
		auto brArgs = std::get<IntrBrArgs>(instrArgs);
		Word offset = sext<Word>(brArgs.offset, 32);
		switch (br_type) {
		case BrType::BR: {
			bool curr_taken = false;
			uint32_t t = static_cast<uint32_t>(thread_last);
			switch (brArgs.cmp) {
			case 0: curr_taken = (rs1_data[t].i == rs2_data[t].i); break;
			case 1: curr_taken = (rs1_data[t].i != rs2_data[t].i); break;
			case 4: curr_taken = (rs1_data[t].i <  rs2_data[t].i); break;
			case 5: curr_taken = (rs1_data[t].i >= rs2_data[t].i); break;
			case 6: curr_taken = (rs1_data[t].u <  rs2_data[t].u); break;
			case 7: curr_taken = (rs1_data[t].u >= rs2_data[t].u); break;
			default: std::abort();
			}
			if (curr_taken) {
				warp.PC = trace->PC + offset;
			}
			core_->perf_stats().branches += 1;
		} break;
		case BrType::JAL: {
			Word link_pc = trace->PC + 4;
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				rd_data[t].i = link_pc;
			}
			warp.PC = trace->PC + offset;
			core_->perf_stats().branches += 1;
		} break;
		case BrType::JALR: {
			Word link_pc = trace->PC + 4;
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				rd_data[t].i = link_pc;
			}
			warp.PC = rs1_data[thread_last].i + offset;
			core_->perf_stats().branches += 1;
		} break;
		case BrType::SYS:
			switch (brArgs.offset) {
			case 0x000: sched.trigger_ecall();  break;
			case 0x001: sched.trigger_ebreak(); break;
			case 0x002: case 0x102: case 0x302: break;
			default: std::abort();
			}
			core_->perf_stats().branches += 1;
			break;
		default:
			std::abort();
		}
		DT(3, this->name() << " execute: op=" << br_type << ", " << *trace);
	} else if (std::get_if<MdvType>(&trace->op_type)) {
		auto mdv_type = std::get<MdvType>(trace->op_type);
		auto mdvArgs = std::get<IntrMdvArgs>(instrArgs);
		switch (mdv_type) {
		case MdvType::MUL: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				if (is_w_enabled && mdvArgs.is_w) {
					auto product = rs1_data[t].i32 * rs2_data[t].i32;
					rd_data[t].i = sext((uint64_t)product, 32);
				} else {
					rd_data[t].i = rs1_data[t].i * rs2_data[t].i;
				}
			}
		} break;
		case MdvType::MULH: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				auto first = static_cast<DWordI>(rs1_data[t].i);
				auto second = static_cast<DWordI>(rs2_data[t].i);
				rd_data[t].i = (first * second) >> XLEN;
			}
		} break;
		case MdvType::MULHSU: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				auto first = static_cast<DWordI>(rs1_data[t].i);
				auto second = static_cast<DWord>(rs2_data[t].u);
				rd_data[t].i = (first * second) >> XLEN;
			}
		} break;
		case MdvType::MULHU: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				auto first = static_cast<DWord>(rs1_data[t].u);
				auto second = static_cast<DWord>(rs2_data[t].u);
				rd_data[t].i = (first * second) >> XLEN;
			}
		} break;
		case MdvType::DIV: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				if (is_w_enabled && mdvArgs.is_w) {
					auto dividen = rs1_data[t].i32;
					auto divisor = rs2_data[t].i32;
					int32_t largest_negative = 0x80000000;
					int32_t quotient;
					if (divisor == 0)                                   quotient = -1;
					else if (dividen == largest_negative && divisor == -1) quotient = dividen;
					else                                                quotient = dividen / divisor;
					rd_data[t].i = sext((uint64_t)quotient, 32);
				} else {
					auto dividen = rs1_data[t].i;
					auto divisor = rs2_data[t].i;
					auto largest_negative = WordI(1) << (XLEN-1);
					WordI quotient;
					if (divisor == 0)                                   quotient = -1;
					else if (dividen == largest_negative && divisor == -1) quotient = dividen;
					else                                                quotient = dividen / divisor;
					rd_data[t].i = quotient;
				}
			}
		} break;
		case MdvType::DIVU: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				if (is_w_enabled && mdvArgs.is_w) {
					auto dividen = rs1_data[t].u32;
					auto divisor = rs2_data[t].u32;
					uint32_t quotient;
					if (divisor != 0) quotient = dividen / divisor;
					else              quotient = -1;
					rd_data[t].i = sext((uint64_t)quotient, 32);
				} else {
					auto dividen = rs1_data[t].u;
					auto divisor = rs2_data[t].u;
					Word quotient;
					if (divisor != 0) quotient = dividen / divisor;
					else              quotient = -1;
					rd_data[t].i = quotient;
				}
			}
		} break;
		case MdvType::REM: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				if (is_w_enabled && mdvArgs.is_w) {
					auto dividen = rs1_data[t].i32;
					auto divisor = rs2_data[t].i32;
					int32_t largest_negative = 0x80000000;
					int32_t remainder;
					if (divisor == 0)                                   remainder = dividen;
					else if (dividen == largest_negative && divisor == -1) remainder = 0;
					else                                                remainder = dividen % divisor;
					rd_data[t].i = sext((uint64_t)remainder, 32);
				} else {
					auto dividen = rs1_data[t].i;
					auto divisor = rs2_data[t].i;
					auto largest_negative = WordI(1) << (XLEN-1);
					WordI remainder;
					if (rs2_data[t].i == 0)                             remainder = dividen;
					else if (dividen == largest_negative && divisor == -1) remainder = 0;
					else                                                remainder = dividen % divisor;
					rd_data[t].i = remainder;
				}
			}
		} break;
		case MdvType::REMU: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				if (is_w_enabled && mdvArgs.is_w) {
					auto dividen = (uint32_t)rs1_data[t].u32;
					auto divisor = (uint32_t)rs2_data[t].u32;
					uint32_t remainder;
					if (divisor != 0) remainder = dividen % divisor;
					else              remainder = dividen;
					rd_data[t].i = sext((uint64_t)remainder, 32);
				} else {
					auto dividen = rs1_data[t].u;
					auto divisor = rs2_data[t].u;
					Word remainder;
					if (rs2_data[t].i != 0) remainder = dividen % divisor;
					else                    remainder = dividen;
					rd_data[t].i = remainder;
				}
			}
		} break;
		default:
			std::abort();
		}
		DT(3, this->name() << " execute: op=" << mdv_type << ", " << *trace);
	} else {
		std::abort();
	}
}

void AluUnit::on_tick() {
  for (uint32_t b = 0; b < NUM_ALU_BLOCKS; ++b) {
		auto& input = Inputs.at(b);
		if (input.empty())
			continue;
		auto& output = Outputs.at(b);
		if (output.full())
			continue; // stall
		auto trace = input.peek();
    this->execute(trace);
		uint32_t delay = this->latency_of(trace);
		output.send(trace, delay);
		if (trace->eop && trace->fetch_stall) {
			core_->resume(trace->wid);
		}
		input.pop();
	}
}
