/*******************************************************************************
 HARPtools by Chad D. Kersey, Spring 2013
*******************************************************************************/
#ifndef __DEBUG_H
#define __DEBUG_H

#ifdef USE_DEBUG
#include <iostream>

#define D(lvl, x) do { \
  using namespace std; \
  if ((lvl) <= USE_DEBUG) { \
    cout << "DEBUG " << __FILE__ << ':' << dec << __LINE__ << ": " \
         << x << endl; \
  } \
} while(0)
#else

#define D(lvl, x) do {} while(0)

#endif

#endif
