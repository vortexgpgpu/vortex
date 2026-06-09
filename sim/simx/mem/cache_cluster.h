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

#pragma once

#include "cache.h"

namespace vortex {

class CacheCluster : public SimObject<CacheCluster> {
public:
	std::vector<std::vector<SimChannel<MemReq>>> core_req_in;
	std::vector<std::vector<SimChannel<MemRsp>>> core_rsp_out;
	std::vector<SimChannel<MemReq>> mem_req_out;
	std::vector<SimChannel<MemRsp>> mem_rsp_in;

	CacheCluster(const SimContext& ctx,
							const char* name,
							uint32_t num_inputs,
							uint32_t num_units,
							const Cache::Config& cache_config);
	~CacheCluster();

	Cache::PerfStats perf_stats() const;

	void flush_begin();
	bool flush_done() const;

protected:
	void on_reset() {}
	void on_tick() {}

private:
	std::vector<Cache::Ptr> caches_;

	friend class SimObject<CacheCluster>;
};

}
