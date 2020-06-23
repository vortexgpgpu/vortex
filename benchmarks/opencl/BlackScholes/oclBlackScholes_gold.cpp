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



#include <math.h>
#include "oclBlackScholes_common.h"



///////////////////////////////////////////////////////////////////////////////
// Rational approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
static double CND(double d){
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double
        K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double
        cnd = RSQRT2PI * exp(- 0.5 * d * d) * 
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}

///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
static void BlackScholesBodyCPU(
    float& call, //Call option price
    float& put,  //Put option price
    float Sf,    //Current stock price
    float Xf,    //Option strike price
    float Tf,    //Option years
    float Rf,    //Riskless rate of return
    float Vf     //Stock volatility
){
    double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

    double sqrtT = sqrt(T);
    double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double    d2 = d1 - V * sqrtT;
    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);

    //Calculate Call and Put simultaneously
    double expRT = exp(- R * T);
    call = (float)(S * CNDD1 - X * expRT * CNDD2);
    put  = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options
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
){
    for(unsigned int i = 0; i < optionCount; i++)
        BlackScholesBodyCPU(
            h_Call[i],
            h_Put[i],
            h_S[i],
            h_X[i],
            h_T[i],
            R,
            V
        );
}
