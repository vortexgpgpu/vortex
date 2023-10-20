/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef _LBM_H_
#define _LBM_H_

#include "ocl.h"
#include "lbm_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

void LBM_allocateGrid( float** ptr );
void LBM_freeGrid( float** ptr );
void LBM_initializeGrid( LBM_Grid grid );
void LBM_initializeSpecialCellsForLDC( LBM_Grid grid );
void LBM_loadObstacleFile( LBM_Grid grid, const char* filename );
void LBM_swapGrids( cl_mem* grid1, cl_mem* grid2 );
void LBM_showGridStatistics( LBM_Grid Grid );
void LBM_storeVelocityField( LBM_Grid grid, const char* filename,
                           const BOOL binary );

/* OpenCL *********************************************************************/

void OpenCL_LBM_allocateGrid( const OpenCL_Param* prm, cl_mem* ptr );
void OpenCL_LBM_freeGrid( cl_mem ptr );
void OpenCL_LBM_initializeGrid( const OpenCL_Param* prm, cl_mem d_grid, LBM_Grid h_grid );
void OpenCL_LBM_getDeviceGrid( const OpenCL_Param* prm, cl_mem d_grid, LBM_Grid h_grid );
void OpenCL_LBM_performStreamCollide( const OpenCL_Param* prm, cl_mem srcGrid, cl_mem dstGrid );

#ifdef __cplusplus
}
#endif

#endif /* _LBM_H_ */
