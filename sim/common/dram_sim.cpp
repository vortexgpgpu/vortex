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

#include "dram_sim.h"
#include "util.h"
#include <fstream>

DISABLE_WARNING_PUSH
DISABLE_WARNING_UNUSED_PARAMETER
DISABLE_WARNING_MISSING_FIELD_INITIALIZERS
#include <base/base.h>
#include <base/request.h>
#include <base/config.h>
#include <frontend/frontend.h>
#include <memory_system/memory_system.h>
DISABLE_WARNING_POP

using namespace vortex;

class DramSim::Impl {
private:
	struct mem_req_t {
		uint64_t addr;
		bool is_write;
		ResponseCallback callback;
		void* arg;
	};

	Ramulator::IFrontEnd* ramulator_frontend_;
	Ramulator::IMemorySystem* ramulator_memorysystem_;
	uint32_t cpu_channel_size_;
	uint64_t cpu_cycles_;
	uint32_t scaled_dram_cycles_;
	static const uint32_t tick_cycles_ = 1000;
	static const uint32_t dram_channel_size_ = 16; // 128 bits
	std::queue<mem_req_t> pending_reqs_;

	void handle_pending_requests() {
		if (pending_reqs_.empty())
			return;
		auto& req = pending_reqs_.front();
		auto req_type = req.is_write ? Ramulator::Request::Type::Write : Ramulator::Request::Type::Read;
		std::function<void(Ramulator::Request&)> callback = nullptr;
		if (req.callback) {
			callback = [req_callback = std::move(req.callback), req_arg = std::move(req.arg)](Ramulator::Request& /*dram_req*/) {
				req_callback(req_arg);
			};
		}
		if (ramulator_frontend_->receive_external_requests(req_type, req.addr, 0, callback)) {
			if (req.is_write) {
				// Ramulator does not handle write responses, so we fire the callback ourselves.
				if (req.callback) {
					req.callback(req.arg);
				}
			}
			pending_reqs_.pop();
		}
	}

public:
	Impl(uint32_t num_channels, uint32_t channel_size, float clock_ratio) {
		YAML::Node dram_config;
		dram_config["Frontend"]["impl"] = "GEM5";
		dram_config["MemorySystem"]["impl"] = "GenericDRAM";
		dram_config["MemorySystem"]["clock_ratio"] = 1;
		dram_config["MemorySystem"]["DRAM"]["impl"] = "HBM2";
		dram_config["MemorySystem"]["DRAM"]["org"]["preset"] = "HBM2_8Gb";
		dram_config["MemorySystem"]["DRAM"]["org"]["density"] = 8192;
		dram_config["MemorySystem"]["DRAM"]["org"]["channel"] = num_channels;
		dram_config["MemorySystem"]["DRAM"]["timing"]["preset"] = "HBM2_2Gbps";
		dram_config["MemorySystem"]["Controller"]["impl"] = "Generic";
		dram_config["MemorySystem"]["Controller"]["Scheduler"]["impl"] = "FRFCFS";
		dram_config["MemorySystem"]["Controller"]["RefreshManager"]["impl"] = "AllBank";
		dram_config["MemorySystem"]["Controller"]["RowPolicy"]["impl"] = "OpenRowPolicy";
		{
			YAML::Node draw_plugin;
			draw_plugin["ControllerPlugin"]["impl"] = "TraceRecorder";
			draw_plugin["ControllerPlugin"]["path"] = "./trace/ramulator.log";
			dram_config["MemorySystem"]["Controller"]["plugins"].push_back(draw_plugin);
		}
		dram_config["MemorySystem"]["AddrMapper"]["impl"] = "RoBaRaCoCh";

		ramulator_frontend_ = Ramulator::Factory::create_frontend(dram_config);
		ramulator_memorysystem_ = Ramulator::Factory::create_memory_system(dram_config);
		ramulator_frontend_->connect_memory_system(ramulator_memorysystem_);
		ramulator_memorysystem_->connect_frontend(ramulator_frontend_);

		cpu_channel_size_ = channel_size;
		scaled_dram_cycles_ = static_cast<uint64_t>(clock_ratio * tick_cycles_);
		this->reset();
	}

	~Impl() {
		std::ofstream nullstream("ramulator.stats.log");
		auto original_buf = std::cout.rdbuf();
		std::cout.rdbuf(nullstream.rdbuf());
		ramulator_frontend_->finalize();
  	ramulator_memorysystem_->finalize();
		std::cout.rdbuf(original_buf);
	}

	void reset() {
		cpu_cycles_ = 0;
	}

	void tick() {
		cpu_cycles_ += tick_cycles_;
		while (cpu_cycles_ >= scaled_dram_cycles_) {
			this->handle_pending_requests();
			ramulator_memorysystem_->tick();
			cpu_cycles_ -= scaled_dram_cycles_;
		}
	}

	void send_request(uint64_t addr, bool is_write, ResponseCallback response_cb, void* arg) {
		// enqueue the request
		if (cpu_channel_size_ > dram_channel_size_) {
			uint32_t n = cpu_channel_size_ / dram_channel_size_;
			for (uint32_t i = 0; i < n; ++i) {
				uint64_t dram_byte_addr = (addr / cpu_channel_size_) * dram_channel_size_ + (i * dram_channel_size_);
				if (i == 0) {
					pending_reqs_.push({dram_byte_addr, is_write, response_cb, arg});
				} else {
					pending_reqs_.push({dram_byte_addr, is_write, nullptr, nullptr});
				}
			}
		} else if (cpu_channel_size_ < dram_channel_size_) {
			uint64_t dram_byte_addr = (addr / cpu_channel_size_) * dram_channel_size_;
			pending_reqs_.push({dram_byte_addr, is_write, response_cb, arg});
		} else {
			uint64_t dram_byte_addr = addr;
			pending_reqs_.push({dram_byte_addr, is_write, response_cb, arg});
		}
	}
};

///////////////////////////////////////////////////////////////////////////////

DramSim::DramSim(uint32_t num_channels, uint32_t channel_size, float clock_ratio)
	: impl_(new Impl(num_channels, channel_size, clock_ratio))
{}

DramSim::~DramSim() {
  delete impl_;
}

void DramSim::reset() {
  impl_->reset();
}

void DramSim::tick() {
  impl_->tick();
}

void DramSim::send_request(uint64_t addr, bool is_write, ResponseCallback callback, void* arg) {
  impl_->send_request(addr, is_write, callback, arg);
}