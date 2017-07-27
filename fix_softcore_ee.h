/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(softcore/ee,FixSoftcoreEE)

#else

#ifndef LMP_FIX_SOFTCORE_EE_H
#define LMP_FIX_SOFTCORE_EE_H

#include "fix.h"
#include "random_park.h"
#include "pair_lj_cut_softcore.h"

namespace LAMMPS_NS {

class FixSoftcoreEE : public Fix {
 public:
  FixSoftcoreEE(class LAMMPS *, int, char **);
  ~FixSoftcoreEE();
  int setmask();
  void init();
  void pre_force(int);
  void pre_reverse(int,int);
  void end_of_step();
  double compute_vector(int);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);

 private:
  int current_node;
  int new_node;
  int seed;
  double minus_beta;

  int gridsize;
  double *weight;

  int downhill;
  RanPark *random;
  void change_node(int);
  int select_node(double*);
  int number_of_atoms();

  int hybrid;
  int npairs;
  PairLJCutSoftcore **pair;
  int *compute_flag;

  int nmax;
  double **f_old;
  double **f_new;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Variable name for fix adapt does not exist

Self-explanatory.

E: Variable for fix adapt is invalid style

Only equal-style variables can be used.

E: Fix adapt pair style does not exist

Self-explanatory

E: Fix adapt pair style param not supported

The pair style does not know about the parameter you specified.

E: Fix adapt type pair range is not valid for pair hybrid sub-style

Self-explanatory.

E: Fix adapt kspace style does not exist

Self-explanatory.

E: Fix adapt requires atom attribute diameter

The atom style being used does not specify an atom diameter.

*/
