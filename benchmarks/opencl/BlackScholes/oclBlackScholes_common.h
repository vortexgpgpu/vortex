/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#include <oclUtils.h>



////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    float *h_Call, //Call option price
    float *h_Put,  //Put option price
    float *h_S,    //Current stock price
    float *h_X,    //Option strike price
    float *h_T,    //Option years
    float R,       //Riskless rate of return
    float V,       //Stock volatility
    unsigned int optionCount
);


////////////////////////////////////////////////////////////////////////////////
// OpenCL Black-Scholes kernel launcher
////////////////////////////////////////////////////////////////////////////////
extern "C" void initBlackScholes(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv);

extern "C" void closeBlackScholes(void);

extern "C" void BlackScholes(
    cl_command_queue cqCommandQueue,
    cl_mem d_Call, //Call option price
    cl_mem d_Put,  //Put option price
    cl_mem d_S,    //Current stock price
    cl_mem d_X,    //Option strike price
    cl_mem d_T,    //Option years
    cl_float R,    //Riskless rate of return
    cl_float V,    //Stock volatility
    cl_uint optionCount
);
