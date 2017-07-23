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

namespace LAMMPS_NS {

class FixSoftcoreEE : public Fix {
 public:
  FixSoftcoreEE(class LAMMPS *, int, char **);
  ~FixSoftcoreEE();
  int setmask();
  void initial_integrate(int);
  void init();
  void end_of_step();
  void setup(int);
  double compute_scalar();

 private:
  int calculate;
  int current_node;
  int seed;
  int gridsize,acfreq;
  double minus_beta,ratiocriteria;
  double *weight;
  double *lambdanode;
  char *lambda_arg[4];
  int downhill;
  int idump;
  FILE *ee_file;
  int nvt_flag, hmc_flag;
  RanPark *random;
  class Compute *pe;
  void change_node(int);
  class FixNVT *fix_ee_nvt;
  class FixHMC *fix_ee_hmc;
  void add_new_compute();
 protected:
  int external_force_clear;   // clear forces locally or externally

  int torqueflag,erforceflag;
  
  int e_flag,rho_flag;

  virtual void force_clear();
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
