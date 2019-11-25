/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

unsigned short read16u(FILE *f);
short read16i(FILE *f);

void write32u(FILE *f, unsigned int i);
void write16u(FILE *f, unsigned short h);
void write16i(FILE *f, short h);

#ifdef __cplusplus
}
#endif
