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
	Ramulator::IFrontEnd* ramulator_frontend_;
	Ramulator::IMemorySystem* ramulator_memorysystem_;

public:
	Impl(int clock_ratio) {
		YAML::Node dram_config;
		dram_config["Frontend"]["impl"] = "GEM5";
		dram_config["MemorySystem"]["impl"] = "GenericDRAM";
		dram_config["MemorySystem"]["clock_ratio"] = clock_ratio;
		dram_config["MemorySystem"]["DRAM"]["impl"] = "HBM2";
		dram_config["MemorySystem"]["DRAM"]["org"]["preset"] = "HBM2_8Gb";
		dram_config["MemorySystem"]["DRAM"]["org"]["density"] = 8192;
		dram_config["MemorySystem"]["DRAM"]["timing"]["preset"] = "HBM2_2Gbps";
		dram_config["MemorySystem"]["Controller"]["impl"] = "Generic";
		dram_config["MemorySystem"]["Controller"]["Scheduler"]["impl"] = "FRFCFS";
		dram_config["MemorySystem"]["Controller"]["RefreshManager"]["impl"] = "AllBank";
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
		//--
	}

	void tick() {
		ramulator_memorysystem_->tick();
	}

  bool send_request(bool is_write, uint64_t addr, int source_id, ResponseCallback callback, void* arg) {
    if (!ramulator_frontend_->receive_external_requests(
			is_write ? Ramulator::Request::Type::Write : Ramulator::Request::Type::Read,
			addr,
			source_id,
			[callback_ = std::move(callback), arg_ = std::move(arg)](Ramulator::Request& /*dram_req*/) {
				callback_(arg_);
			}
		)) {
			return false;
		}
		if (is_write) {
			// Ramulator does not handle write responses, so we call the callback ourselves
			callback(arg);
		}
		return true;
  }
};

///////////////////////////////////////////////////////////////////////////////

DramSim::DramSim(int clock_ratio)
	: impl_(new Impl(clock_ratio))
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

bool DramSim::send_request(bool is_write, uint64_t addr, int source_id, ResponseCallback callback, void* arg) {
  return impl_->send_request(is_write, addr, source_id, callback, arg);
}