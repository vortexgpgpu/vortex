/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#include <math.h>
#include <stdlib.h>

#include "types.h"

#ifdef DEBUG
#include <iostream>

#define DEBUGMSG(x) do { \
  std::cout << __FILE__ << ':' <<__LINE__ << ": " << x << '\n'; \
} while(0)
#else
#define DEBUGMSG(x) do { } while(0)
#endif

namespace Harp {
  // This class serves to handle the strange-precision floating point that can 
  // crop up in HARP. 
  class Float {
  public:
    Float(Word_u bin, Size n): sz(n) {
      DEBUGMSG("Float(0x" << std::hex << bin << ", " << std::dec << n << ')');
      
      bool sign(bin >> (n*8 - 1));
      
      Size expSz;
      if (n < 4) {
        expSz = 5;
      } else if (n < 8) {
        expSz = 8;
      } else {
        expSz = 11;
      }

      Size sigSz = n*8 - expSz - 1;

      DEBUGMSG("  exp: " << std::dec << expSz <<
               " bits, sig: " << std::dec << sigSz << " bits.");

      int exp = (bin >> sigSz) & ((1<<expSz) - 1);
      Word_u sig = bin & ((1llu<<sigSz) - 1);
      DEBUGMSG("  sig=" << std::dec << sig << " exp=" << exp);

      if (exp == 0) {
        // Subnormal 
        d = sig / pow(2, ((1<<(expSz-1))-2)) / pow(2, sigSz);
        DEBUGMSG("  Denorm.");
      } else if (exp == ((1<<expSz) - 1)) {
        // Infinity
        d = HUGE_VAL;
        DEBUGMSG("  Inf.");
      } else {
        // Normalized, implied 1.
        exp -= (1<<(expSz - 1)) - 1;
        d = pow(2.0, exp - int(sigSz)) * double((1ll << sigSz) + sig);
        DEBUGMSG("  Norm, exp=" << exp);
      }

      if (sign) d = -d;

      DEBUGMSG("Set to " << d);
    }

    Float(double d, Size n): sz(n), d(d) { DEBUGMSG("Float(double, size)"); }

    operator Word_u() {
      DEBUGMSG("Float -> Word_u: " << d);
      Size expSz;
      if (sz < 4) {
        expSz = 5;
      } else if (sz < 8) {
        expSz = 8;
      } else {
        expSz = 11;
      }

      Size sigSz = 8*sz - expSz - 1;

      bool sign(d < 0);

      bool inf(std::isinf(d)), zero(d == 0.0);
      int exp;

      if (!inf && !zero) exp = floor(log2(fabs(d)));

      Word_u rval;
      if (inf) {
        // Infinity
        DEBUGMSG("  Inf.");
        rval = ((1llu<<expSz)-1llu)<<sigSz;
      } else if (!zero && abs(exp) < (1<<(expSz-1)) - 1) {
        // Normalized with implied 1.
        Word_u sig = (fabs(d) * pow(2.0, -exp) - 1.0) * pow(2.0, sigSz);
        DEBUGMSG("  Norm, exp=" << exp << ", sig=" << sig);
        rval = ((((exp + ((1llu<<(expSz-1)) - 1llu))
                                          &((1llu<<expSz)-1llu)))<<sigSz) | sig;
      } else if (!zero && exp > -(1<<(expSz-1)) - sigSz) {
        // Subnormal number.
        Word_u sig = round(fabs(d)*pow(2.0,((1<<(expSz-1))-2))*pow(2.0, sigSz));
        DEBUGMSG("  Denorm, exp=" << exp << ", sig=" << sig);
        rval = sig;
      } else {
        // Zero.
        rval = 0;
      }
      
      if (sign) rval |= 1llu<<(sz*8 - 1);

      DEBUGMSG("  Returning 0x" << std::hex << rval);

      return rval;
    }

    operator double() { DEBUGMSG("Float->double " << d); return d; }

  private:
    double d;
    Size sz;
  };
};
