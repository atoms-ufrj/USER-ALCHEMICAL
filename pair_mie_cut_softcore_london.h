/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(mie/cut/softcore/london,PairMieCutSoftcoreLondon)

#else

#ifndef LMP_PAIR_MIE_CUT_SOFTCORELONDON_H
#define LMP_PAIR_MIE_CUT_SOFTCORELONDON_H

#include "pair_alchemical.h"

namespace LAMMPS_NS {

class PairMieCutSoftcoreLondon : public PairAlchemical {
 public:
  PairMieCutSoftcoreLondon(class LAMMPS *);
  virtual ~PairMieCutSoftcoreLondon();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  void init_list(int, class NeighList *);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

  void compute_inner();
  void compute_middle();
  void compute_outer(int, int);

 protected:
  double cut_global;
  double **cut;
  double **epsilon,**sigma;
  double **gamR,**Cmie;
  double **mie1,**mie2,**mie3,**offset;
  double *cut_respa;

  virtual void allocate();

  double **asq;
  double ***mie1n,***mie2n,***mie3n,***asqn,***offsetn;
  double atanx_x(double x);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
