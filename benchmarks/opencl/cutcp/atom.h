/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef ATOM_H
#define ATOM_H

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct Atom_t {
    float x, y, z, q;
  } Atom;

  typedef struct Atoms_t {
    Atom *atoms;
    int size;
  } Atoms;

  typedef struct Vec3_t {
    float x, y, z;
  } Vec3;

  Atoms *read_atom_file(const char *fname);
  void free_atom(Atoms *atom);
  void get_atom_extent(Vec3 *lo, Vec3 *hi, Atoms *atom);

#ifdef __cplusplus
}
#endif

#endif /* ATOM_H */
