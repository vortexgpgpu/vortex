#include "execute.h"

void executeVector(const Instr &instr, vortex::Core *core_, std::vector<reg_data_t[3]> &rsdata, std::vector<reg_data_t> &rddata, std::vector<std::vector<Byte>> &vreg_file_, vtype vtype_, uint32_t vl_, uint32_t warp_id_);