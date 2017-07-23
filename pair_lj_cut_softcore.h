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

#ifdef PAIR_CLASS

PairStyle(lj/cut/softcore,PairLJCutSoftcore)

#else

#ifndef LMP_PAIR_LJ_CUT_SOFTCORE_H
#define LMP_PAIR_LJ_CUT_SOFTCORE_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCutSoftcore : public Pair {
 public:
  PairLJCutSoftcore(class LAMMPS *);
  virtual ~PairLJCutSoftcore();
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
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);
  void modify_params(int narg, char **arg);

  void compute_inner();
  void compute_middle();
  void compute_outer(int, int);

  void init_grid_one(int, int);
  void compute_grid();

 protected:
  double cut_global;
  double **cut;
  double **epsilon,**sigma;
  double **lj1,**lj2,**lj3,**lj4,**asq,**offset;
  double *cut_respa;

  double alpha, exponent_n, exponent_p;  // softcore model parameters
  double lambda;                         // coupling parameter value
  int    **linkflag;                     /* +1 for directly linked pairs
                                             0 for unlinked pairs
                                            -1 for reversely linked pairs */
  int    *linkedtype;                    // 1 if type belongs to any linked pair

  int    gridflag;    // 1 if grid must be computed in the current step
  int    uptodate;    // 1 if grid was computed in the latest step
  int    gridsize;    // number of nodes in the grid
  double *lambdanode; // lambda value at each node
  double *evdwlnode;  // total pairwise interaction energy at each node
  double *etailnode;  // tail correction for energy at each node
  double *weight;     // sampling weight for expanded ensemble
  double ***lj3n,***lj4n,***asqn,***offn; // variables for grid calculations

  void allocate();
  double atanx_x(double x);
  void add_node_to_grid(double, double);

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
