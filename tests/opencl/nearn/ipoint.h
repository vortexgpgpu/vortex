/*********************************************************** 
*  --- OpenSURF ---                                        *
*  This library is distributed under the GNU GPL. Please   *
*  contact chris.evans@irisys.co.uk for more information.  *
*                                                          *
*  C. Evans, Research Into Robust Visual Features,         *
*  MSc University of Bristol, 2008.                        *
*                                                          *
************************************************************/

#ifndef IPOINT_H
#define IPOINT_H

#include <vector>
#include <math.h>



//-------------------------------------------------------
typedef struct{
        int x;
        int y;
		float descriptor[64];
	} Ipoint;

//-------------------------------------------------------

  typedef std::vector<Ipoint> IpVec;
#endif
