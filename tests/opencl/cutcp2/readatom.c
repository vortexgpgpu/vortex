/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "atom.h"


#define LINELEN 96
#define INITLEN 20


Atoms *read_atom_file(const char *fname)
{
  FILE *file;
  char line[LINELEN];

  Atom *atom;			/* Atom array */
  int len = INITLEN;		/* Size of atom array */
  int cnt = 0;			/* Number of atoms read */

  /* allocate initial atom array */
  atom = (Atom *) malloc(len * sizeof(Atom));
  if (NULL==atom) {
    fprintf(stderr, "can't allocate memory\n");
    return NULL;
  }

  int i;
  for (i = 0; i < len; ++i) {
    atom[i].x = i+0;
    atom[i].y = i+1;
    atom[i].z = i+2;
    atom[i].q = 1;
  }

#if 0
  /* open atom "pqr" file */
  file = fopen(fname, "r");
  if (NULL==file) {
    fprintf(stderr, "can't open file \"%s\" for reading\n", fname);
    return NULL;
  }

  /* loop to read pqr file line by line */
  while (fgets(line, LINELEN, file) != NULL) {

    if (strncmp(line, "ATOM  ", 6) != 0 && strncmp(line, "HETATM", 6) != 0) {
      continue;  /* skip anything that isn't an atom record */
    }

    if (cnt==len) {  /* extend atom array */
      void *tmp = realloc(atom, 2*len*sizeof(Atom));
      if (NULL==tmp) {
        fprintf(stderr, "can't allocate more memory\n");
        return NULL;
      }
      atom = (Atom *) tmp;
      len *= 2;
    }

    /* read position coordinates and charge from atom record */
    if (sscanf(line, "%*s %*d %*s %*s %*d %f %f %f %f", &(atom[cnt].x),
          &(atom[cnt].y), &(atom[cnt].z), &(atom[cnt].q)) != 4) {
      fprintf(stderr, "atom record %d does not have expected format\n", cnt+1);
      return NULL;
    }

    cnt++;  /* count atoms as we store them */
  }

  /* verify EOF and close file */
  if ( !feof(file) ) {
    fprintf(stderr, "did not find EOF\n");
    return NULL;
  }
  if (fclose(file)) {
    fprintf(stderr, "can't close file\n");
    return NULL;
  }
#endif

  /* Build the output data structure */
  {
    Atoms *out = (Atoms *)malloc(sizeof(Atoms));

    if (NULL == out) {
      fprintf(stderr, "can't allocate memory\n");
      return NULL;
    }

    out->size = cnt;
    out->atoms = atom;

    return out;
  }
}


void free_atom(Atoms *atom)
{
  if (atom) {
    free(atom->atoms);
    free(atom);
  }
}

void
get_atom_extent(Vec3 *out_lo, Vec3 *out_hi, Atoms *atom)
{
  Atom *atoms = atom->atoms;
  int natoms = atom->size;
  Vec3 lo;
  Vec3 hi;
  int n;

  hi.x = lo.x = atoms[0].x;
  hi.y = lo.y = atoms[0].y;
  hi.z = lo.z = atoms[0].z;

  for (n = 1; n < natoms; n++) {
    lo.x = fminf(lo.x, atoms[n].x);
    hi.x = fmaxf(hi.x, atoms[n].x);
    lo.y = fminf(lo.y, atoms[n].y);
    hi.y = fmaxf(hi.y, atoms[n].y);
    lo.z = fminf(lo.z, atoms[n].z);
    hi.z = fmaxf(hi.z, atoms[n].z);
  }

  *out_lo = lo;
  *out_hi = hi;
}
