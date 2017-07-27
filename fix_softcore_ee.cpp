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

/* ----------------------------------------------------------------------
   Contributing authors: Ana J. Silveira (asilveira@plapiqui.edu.ar)
                         Charlles R. A. Abreu (abreu@eq.ufrj.br)
------------------------------------------------------------------------- */

#include "fix_softcore_ee.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "error.h"
#include "comm.h"
#include "random_park.h"
#include "string.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "modify.h"
#include "compute.h"
#include "timer.h"
#include "pair_hybrid_ee.h"
#include "memory.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSoftcoreEE::FixSoftcoreEE(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  // Retrieve fix softcore/ee command arguments:
  if (narg < 6 || narg == 7)
    error->all(FLERR,"Illegal fix softcore/ee command");

  nevery = force->numeric(FLERR,arg[3]);
  if (nevery <= 0)
    error->all(FLERR,"Illegal fix softcore/ee command");

  seed = force->numeric(FLERR,arg[4]);
  if (seed <= 0)
    error->all(FLERR,"Illegal fix softcore/ee command");
  minus_beta = -1.0/(force->boltz*force->numeric(FLERR,arg[5]));

  if (narg > 7) {
    if (strcmp(arg[6],"weights") != 0)
      error->all(FLERR,"Illegal fix softcore/ee command");
    gridsize = narg - 7;
    memory->create(weight,gridsize,"fix_softcore_ee::weight");
    for (int i = 0; i < gridsize; i++)
      weight[i] = force->numeric(FLERR,arg[7+i]);
  }
  else
    weight = NULL;

  // Set fix softcore/ee properties:
  scalar_flag = 1;
  global_freq = 1;

  // Retrieve all lambda-related pair styles:
  if (strcmp(force->pair_style,"hybrid/ee") == 0) {
    PairHybridEE *pair_hybrid = (PairHybridEE *) force->pair;
    npairs = 0;
    for (int i = 0; i < pair_hybrid->nstyles; i++)
      if (strcmp(pair_hybrid->keywords[i],"lj/cut/softcore") == 0)
        npairs++;
    pair = new PairLJCutSoftcore*[npairs];
    int j = 0;
    for (int i = 0; i < pair_hybrid->nstyles; i++)
      if (strcmp(pair_hybrid->keywords[i],"lj/cut/softcore") == 0)
        pair[j++] = (PairLJCutSoftcore *) pair_hybrid->styles[i];
    }
    else if (strcmp(force->pair_style,"lj/cut/softcore") == 0) {
      npairs = 1;
      pair[0] = (PairLJCutSoftcore *) force->pair;
    }

  if (npairs == 0)
    error->all(FLERR,"fix softcore/ee: no pair styles associated to coupling parameter lambda");

  // Allocate force buffer:
  nmax = atom->nlocal;
  if (force->newton_pair)
    nmax += atom->nghost;
  memory->create(f_new,nmax,3,"fix_softcore_ee::f_new");

  // Prepare array for lambda value change in pair styles:
  lambda_arg[0] = (char*)"lambda";
  lambda_arg[1] = new char[18];

  // Initialize random number generator:
  random = new RanPark(lmp,seed);
}

/* ---------------------------------------------------------------------- */

FixSoftcoreEE::~FixSoftcoreEE()
{
  memory->destroy(f_new);
  if (weight) memory->destroy(weight);
}

/* ---------------------------------------------------------------------- */

int FixSoftcoreEE::setmask()
{
  return PRE_FORCE;
}

/* ---------------------------------------------------------------------- */

void FixSoftcoreEE::init()
{
  // Determine the number of nodes in the lambda grid:
  int nodes = pair[0]->gridsize;
  int all_equal = 1;
  for (int i = 1; i < npairs; i++)
    all_equal &= pair[i]->gridsize == nodes;

  if (all_equal) {
    if (nodes == 0)
      error->all(FLERR,"fix softcore/ee: no lambda grid has been defined");
    else if (!weight) {
      gridsize = nodes;
      memory->create(weight,gridsize,"fix_softcore_ee::weight");
      for (int i = 0; i < gridsize; i++)
        weight[i] = 0.0;
    }
  }
  else
    error->all(FLERR,"fix softcore/ee: all lambda grids must have the same number of nodes");

  if (comm->me == 0) {
    FILE* unit[2] = {screen,logfile};
    for (int i = 0; i < 2; i++)
      if (unit[i]) {
        fprintf(unit[i],"Expanded ensemble weights: (");
        for (int k = 0; k < gridsize-1; k++)
          fprintf(unit[i],"%g; ",weight[k]);
        fprintf(unit[i],"%g)\n",weight[gridsize-1]);
      }
  }

  change_node(0);
  downhill = 0;

  for (int i = 0; i < npairs; i++)
    pair[i]->gridflag = 0;
}

/*----------------------------------------------------------------------------*/

void FixSoftcoreEE::pre_force(int vflag)
{
  int i;
  if (update->ntimestep % nevery == 0) {

    // Compute lambda-related energy at every node:
    double energy[gridsize] = {0.0};
    for (i = 0; i < npairs; i++)
      add_energies( &energy[0], pair[i] );

    // Select a node based on their Boltzmann weights:
    new_node = select_node( energy );

    if (new_node != current_node) {

      for (i = 0; i < npairs; i++)
        pair[i]->compute(0,0);

      change_node(new_node);

      int n = force_array_size();
      for (i = 0; i < n; i++)
        f_new[i][0] = f_new[i][1] = f_new[i][2] = 0.0;

      std::swap(atom->f,f_new);
      for (i = 0; i < npairs; i++)
        pair[i]->compute(0,0);
      std::swap(atom->f,f_new);

      for (i = 0; i < n; i++) {
        atom->f[i][0] -= f_new[i][0];
        atom->f[i][1] -= f_new[i][1];
        atom->f[i][2] -= f_new[i][2];
      }
    }
  }
}

/*----------------------------------------------------------------------------*/

int FixSoftcoreEE::select_node(double *energy)
{
  int i, node;
  double umax, usum, acc, r;
  double P[gridsize];

  P[0] = minus_beta*energy[0] + weight[0];
  umax = P[0];
  for (i = 1; i < gridsize; i++) {
    P[i] = minus_beta*energy[i] + weight[i];
    umax = MAX(umax,P[i]);
  }

  usum = 0.0;
  for (i = 0; i < gridsize; i++) {
    P[i] = exp(P[i] - umax);
    usum += P[i];
  }
  for (i = 0; i < gridsize; i++)
    P[i] /= usum;

  r = random->uniform();
  acc = P[0];
  node = 0;
  while (r > acc) {
    node++;
    acc += P[node];
  }
  MPI_Bcast(&node,1,MPI_INT,0,world);

  return node;
}

/*----------------------------------------------------------------------------*/

void FixSoftcoreEE::add_energies(double *energy, PairLJCutSoftcore *pair)
{
  pair->compute_grid();
  double node_energy[gridsize];
  MPI_Allreduce(pair->evdwlnode,&node_energy[0],gridsize,MPI_DOUBLE,MPI_SUM,world);
  for (int k = 0; k < gridsize; k++)
    energy[k] += node_energy[k];
  if (pair->tail_flag) {
    double volume = domain->xprd * domain->yprd * domain->zprd;
    for (int k = 0; k < gridsize; k++)
      energy[k] += pair->etailnode[k]/volume;
  }
}

/* ---------------------------------------------------------------------- */

void FixSoftcoreEE::change_node(int node)
{
  current_node = node;
  for (int i = 0; i < npairs; i++) {
    sprintf(lambda_arg[1],"%18.16f",pair[i]->lambdanode[node]);
    pair[i]->modify_params(2,lambda_arg);
    pair[i]->reinit();
  }
  if (downhill)
    downhill = current_node != 0;
  else
    downhill = current_node == gridsize - 1;
}

/* ----------------------------------------------------------------------

------------------------------------------------------------------------- */

int FixSoftcoreEE::force_array_size()
{
  int n = atom->nlocal;
  if (force->newton_pair)
    n += atom->nghost;
  if (n > nmax) {
    nmax = n;
    memory->grow(f_new,nmax,3,"fix_softcore_ee::f_new");
  }
  return n;
}

/* ----------------------------------------------------------------------
   Return node
------------------------------------------------------------------------- */

double FixSoftcoreEE::compute_scalar()
{
  
  return current_node;
}

/* ---------------------------------------------------------------------- */

