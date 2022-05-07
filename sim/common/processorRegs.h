#pragma once

#include <cstdint>

namespace vortex{
    // according to spec https://github.com/riscv/virtual-memory/blob/main/specs/668-Svinval.pdf
    // mode is defined in the followig way:
    // - 0 = No translation or protection
    // - 1-7 = Reserved for standart use
    // - 8 = SV39 Page-based 39-bit virtual addressing
    // - 9 = SV48 Page based 48-bit virtual addressing
    // - 10 = SV57 Page-based 57-bit virtual addressing
    // - 11 = SV64 Page based virtual addressing
    // - 12 - 13 = Reserved for standart use
    // - 14 - 15 = Designed for custom use
    enum VirtualAddressTranslationMode:uint8_t{
        NoTranslation = 0,
        SV39 = 8
    };

    class SATP{
        public:
        SATP(VirtualAddressTranslationMode mode);
        void updateMode(VirtualAddressTranslationMode mode);
        void updateRootPageTable(uint64_t ppn);
        VirtualAddressTranslationMode Mode(){
            return static_cast<VirtualAddressTranslationMode>(regBits_ >> (length - modeBitsLength-1)); 
        }

       uint64_t getRootPageNumber();

        private:
        static const int modeBitsLength = 4;
        static const int rootPageLength = 44;
        static const int length = 64;
        uint64_t regBits_;
    };

    class SCAUSE{
        public:
        void setPageFaultExceptionAccured(bool isAccured);
        bool checkIsPageFaultExceptionAccured();
        private:
        void setBit(bool isSet, int number);
        bool isBitSet(int number);
        static const int pageFaultBitNumber = 12;
        uint64_t regBits_;
    };

    class STVAL{
        public:
        void updateValue(uint64_t value){
            regBits_ = value;
        };

        uint64_t value(){
            return regBits_;
        }
        private:
        uint64_t regBits_;
    };

    class SupervisorRegisterContainer{
        public:
        SupervisorRegisterContainer(VirtualAddressTranslationMode mode):
            satp(mode){};
        // according to spec https://github.com/riscv/virtual-memory/blob/main/specs/668-Svinval.pdf
        // these registers are used to control virtual page translation.
        // Satp is used to define virtual address/page table entry/root page table
        // Scause defines reason why exception/interruption accured
        // Stval register is used to get the reason why exception is accured
        SATP satp;
        SCAUSE scause;
        STVAL stval;
    };
}