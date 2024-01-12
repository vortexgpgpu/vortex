#include "execute.h"

void loadVector(std::vector<std::vector<Byte>> &vreg_file, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, uint32_t rdest, std::vector<Byte> mask, uint32_t vsew, uint32_t vl, uint32_t vmask);
void storeVector(std::vector<std::vector<Byte>> &vreg_file, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, uint32_t rsrc3, std::vector<Byte> mask, uint32_t vsew, uint32_t vl, uint32_t vmask);
void executeVector(const Instr &instr, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, std::vector<reg_data_t> &rddata, std::vector<std::vector<Byte>> &vreg_file_, vtype &vtype_, uint32_t &vl_, uint32_t warp_id_);