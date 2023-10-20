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


extern "C" {
    void load_file   (char * filename);
    void dbus_driver(bool clk, unsigned o_m_read_addr, unsigned o_m_evict_addr, bool o_m_valid, svLogicVecVal * o_m_writedata, bool o_m_read_or_write, unsigned cache_banks, unsigned num_words_per_block, svLogicVecVal * i_m_readdata, bool * i_m_ready);
    void ibus_driver(bool clk, unsigned o_m_read_addr, unsigned o_m_evict_addr, bool o_m_valid, svLogicVecVal * o_m_writedata, bool o_m_read_or_write, unsigned cache_banks, unsigned num_words_per_block, svLogicVecVal * i_m_readdata, bool * i_m_ready);
    void io_handler  (bool clk, bool io_valid, unsigned io_data);
    void gracefulExit();
}