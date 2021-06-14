/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef CUTOFF_H
#define CUTOFF_H

#ifdef __cplusplus
extern "C" {
#endif

#define SHIFTED

  /* A structure to record how points in 3D space map to array
     elements.  Array element (z, y, x)
     where 0 <= x < nx, 0 <= y < ny, 0 <= z < nz
     maps to coordinate (xlo, ylo, zlo) + h * (x, y, z).
  */
  typedef struct LatticeDim_t {
    /* Number of lattice points in x, y, z dimensions */
    int nx, ny, nz;

    /* Lowest corner of lattice */
    Vec3 lo;

    /* Lattice spacing */
    float h;
  } LatticeDim;

  /* An electric potential field sampled on a regular grid.  The
     lattice size and grid point positions are specified by 'dim'.
  */
  typedef struct Lattice_t {
    LatticeDim dim;
    float *lattice;
  } Lattice;

  LatticeDim lattice_from_bounding_box(Vec3 lo, Vec3 hi, float h);

  Lattice *create_lattice(LatticeDim dim);
  void destroy_lattice(Lattice *);

  int gpu_compute_cutoff_potential_lattice(
      struct pb_TimerSet *timers,
      Lattice *lattice,
      float cutoff,                      /* cutoff distance */
      Atoms *atom,                       /* array of atoms */
      int verbose,                        /* print info/debug messages */
      struct pb_Parameters *parameters
    );

  int cpu_compute_cutoff_potential_lattice(
      Lattice *lattice,                  /* the lattice */
      float cutoff,                      /* cutoff distance */
      Atoms *atoms                       /* array of atoms */
    );

  int remove_exclusions(
      Lattice *lattice,                  /* the lattice */
      float exclcutoff,                  /* exclusion cutoff distance */
      Atoms *atom                        /* array of atoms */
    );

#ifdef __cplusplus
}
#endif

#endif /* CUTOFF_H */
