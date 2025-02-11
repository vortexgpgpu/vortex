#ifdef EXT_V_ENABLE

#include "vec_unit.h"
#include "emulator.h"

using namespace vortex;

class VecUnit::Impl {
public:
    Impl(VecUnit* simobject, const Arch& /*arch*/)
        : simobject_(simobject)
    {
        this->clear();
    }

    ~Impl() {}

    void clear() {
        perf_stats_ = PerfStats();
    }

    void tick() {
    }

/*
    void load(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata) {
    }

    void store(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata) {
    }

    void execute(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata, std::vector<reg_data_t> &rddata) {
    }
*/

    const PerfStats& perf_stats() const {
        return perf_stats_;
    }

private:

    VecUnit* simobject_;
    std::vector<std::vector<Byte>>  vreg_file_;
    vtype_t                         vtype_;
    uint32_t                        vl_;
    Word                            vlmax_;
    PerfStats perf_stats_;
};

VecUnit::VecUnit(const SimContext& ctx,
                 const char* name,
                 const Arch &arch)
    : SimObject<VecUnit>(ctx, name)
    , Input(this)
    , Output(this)
    , impl_(new Impl(this, arch))
{}

VecUnit::~VecUnit() {
    delete impl_;
}

void VecUnit::reset() {
    impl_->clear();
}

void VecUnit::tick() {
    impl_->tick();
}

/*
void VecUnit::load(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata) {
    return impl_->load(instr, wid, rsdata);
}

void VecUnit::store(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata) {
    return impl_->store(instr, wid, rsdata);
}

void VecUnit::execute(const Instr &instr, uint32_t wid, std::vector<reg_data_t[3]> &rsdata, std::vector<reg_data_t> &rddata) {
    return impl_->execute(instr, wid, rsdata, rddata);
}
*/

const VecUnit::PerfStats& VecUnit::perf_stats() const {
    return impl_->perf_stats();
}
#endif