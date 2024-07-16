/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef _MAIN_H_
#define _MAIN_H_

/*############################################################################*/

typedef struct {
	int nTimeSteps;
	char* resultFilename;
	char* obstacleFilename;
} MAIN_Param;

/*############################################################################*/

void MAIN_parseCommandLine( int nArgs, char* arg[], MAIN_Param* param, struct pb_Parameters* );
void MAIN_printInfo( const MAIN_Param* param );
void MAIN_initialize( const MAIN_Param* param, const OpenCL_Param* prm );
int MAIN_finalize( const MAIN_Param* param, const OpenCL_Param* prm );

void OpenCL_initialize(struct pb_Parameters*, OpenCL_Param* prm);

/*############################################################################*/

#endif /* _MAIN_H_ */
