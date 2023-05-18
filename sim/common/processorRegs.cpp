#include "processorRegs.h"

using namespace vortex;

SATP:: SATP(VirtualAddressTranslationMode mode){
    updateMode(mode);
}

void SATP::updateMode(VirtualAddressTranslationMode mode){
    uint64_t bitsMode = mode;
    regBits_ =(regBits_ & (((uint64_t)1 << 60)-1)) | (bitsMode<< 59);
}

void SATP::updateRootPageTable(uint64_t ppn){
    regBits_ = ((regBits_ >> rootPageLength) << rootPageLength) 
        | (ppn & ((((uint64_t)1 << rootPageLength) -1)));
}

uint64_t SATP::getRootPageNumber(){
    auto rootNumber =  regBits_ & (((uint64_t)1 << rootPageLength) -1);
    return rootNumber;
}

bool SCAUSE::checkIsPageFaultExceptionAccured(){
   return isBitSet(pageFaultBitNumber); 
}

void SCAUSE::setPageFaultExceptionAccured(bool isAccured){
    setBit(isAccured, pageFaultBitNumber);
}

void SCAUSE::setBit(bool isSet, int number){
    regBits_ =  regBits_ | ((uint64_t) 1 << number);
}

bool SCAUSE::isBitSet(int number){
    return (regBits_ & ((uint64_t)1 << number)) !=0;   
}



