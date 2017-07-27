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
#include "integrate.h"

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
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;

  // Retrieve all lambda-related pair styles:
  hybrid = strcmp(force->pair_style,"hybrid/ee") == 0;
  if (hybrid) {
    PairHybridEE *pair_hybrid = (PairHybridEE *) force->pair;
    npairs = 0;
    for (int i = 0; i < pair_hybrid->nstyles; i++)
      if (strcmp(pair_hybrid->keywords[i],"lj/cut/softcore") == 0)
        npairs++;
    pair = new PairLJCutSoftcore*[npairs];
    int j = 0;
    for (int i = 0; i < pair_hybrid->nstyles; i++)
      if (strcmp(pair_hybrid->keywords[i],"lj/cut/softcore") == 0) {
        pair[j++] = (PairLJCutSoftcore *) pair_hybrid->styles[i];
      }
  }
  else if (strcmp(force->pair_style,"lj/cut/softcore") == 0) {
    npairs = 1;
    pair[0] = (PairLJCutSoftcore *) force->pair;
  }

  if (npairs == 0)
    error->all(FLERR,"fix softcore/ee: no pair styles associated to coupling parameter lambda");

  compute_flag = new int[npairs];

  // Allocate force buffer:
  nmax = atom->nlocal;
  if (force->newton_pair)
    nmax += atom->nghost;
  memory->create(f_old,nmax,3,"fix_softcore_ee::f_old");
  memory->create(f_new,nmax,3,"fix_softcore_ee::f_new");

  // Initialize random number generator:
  random = new RanPark(lmp,seed);
}

/* ---------------------------------------------------------------------- */

FixSoftcoreEE::~FixSoftcoreEE()
{
  memory->destroy(f_old);
  memory->destroy(f_new);
  if (weight) memory->destroy(weight);
}

/* ---------------------------------------------------------------------- */

int FixSoftcoreEE::setmask()
{
  return PRE_FORCE | PRE_REVERSE | END_OF_STEP;
}

/* ---------------------------------------------------------------------- */

void FixSoftcoreEE::init()
{
  // Determine the number of nodes in the lambda grid:
  int nodes = pair[0]->gridsize;
  int all_equal = 1;
  for (int i = 1; i < npairs; i++)
    all_equal &= pair[i]->gridsize == nodes;
  if (!all_equal)
    error->all(FLERR,"fix softcore/ee: lambda grids have different numbers of nodes");
  if (nodes == 0)
    error->all(FLERR,"fix softcore/ee: no lambda grid has been defined");

  // Check if weights were specified in the required amount:
  if (!weight) {
    gridsize = nodes;
    memory->create(weight,gridsize,"fix_softcore_ee::weight");
    for (int i = 0; i < gridsize; i++)
      weight[i] = 0.0;
  }
  else if (gridsize != nodes)
    error->all(FLERR,"fix softcore/ee: numbers of weights and lambda nodes are different");

  // Print the weights:
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

  // Store compute flags of pair styles:
  for (int i = 0; i < npairs; i++)
    compute_flag[i] = pair[i]->compute_flag;

  // Go to the first node:
  change_node(0);
  downhill = 0;
}

/*----------------------------------------------------------------------------*/

void FixSoftcoreEE::pre_force(int vflag)
{
  if (update->ntimestep % nevery) return;

  if (hybrid)
    for (int i = 0; i < npairs; i++)
      pair[i]->compute_flag = 0;
  else
    pair[0]->skip = 1;
}

/*----------------------------------------------------------------------------*/

void FixSoftcoreEE::pre_reverse(int eflag, int vflag)
{
  if (update->ntimestep % nevery) return;

  // Restore compute flags:
  if (hybrid)
    for (int i = 0; i < npairs; i++)
      pair[i]->compute_flag = compute_flag[i];
  else
    pair[0]->skip = !compute_flag[0];

  // Compute and store pair interactions with current lambda value:
  double **f = atom->f;
  int n = number_of_atoms();
  for (int i = 0; i < n; i++)
    f_old[i][0] = f_old[i][1] = f_old[i][2] = 0.0;
  std::swap(f,f_old);
  for (int i = 0; i < npairs; i++) {
    pair[i]->gridflag = 1;
    pair[i]->compute(eflag,vflag);
  }
  std::swap(f,f_old);

  // Add the forces computed above so as to complete the time step:
  for (int i = 0; i < n; i++) {
    f[i][0] += f_old[i][0];
    f[i][1] += f_old[i][1];
    f[i][2] += f_old[i][2];
  }

  // Compute lambda-related energy at every node:
  double energy[gridsize] = {0.0};
  for (int i = 0; i < npairs; i++) {
    double node_energy[gridsize];
    MPI_Allreduce(pair[i]->evdwlnode,&node_energy[0],gridsize,MPI_DOUBLE,MPI_SUM,world);
    if (pair[i]->tail_flag) {
      double volume = domain->xprd*domain->yprd*domain->zprd;
      for (int j = 0; j < gridsize; j++)
        node_energy[j] += pair[i]->etailnode[j]/volume;
    }
    for (int j = 0; j < gridsize; j++)
      energy[j] += node_energy[j];
  }

  // Select a node from the expanded ensemble:
  new_node = select_node( energy );

  // Change node if necessary:
  if (new_node != current_node) {
    change_node(new_node);

    // Compute and store pair interactions with new lambda value:
    for (int i = 0; i < n; i++)
      f_new[i][0] = f_new[i][1] = f_new[i][2] = 0.0;
    std::swap(f,f_new);
    for (int i = 0; i < npairs; i++) {
      pair[i]->compute(eflag,vflag);
      pair[i]->uptodate = 1;
    }
    std::swap(f,f_new);
  }

  // Update energies and virials with most recently computed values:
  if (hybrid) {
    class Pair *p = force->pair;
    for (int i = 0; i < npairs; i++) {
      if (p->eflag_global) {
        p->eng_vdwl += pair[i]->eng_vdwl;
        p->eng_coul += pair[i]->eng_coul;
      }
      if (p->vflag_global)
        for (int k = 0; k < 6; k++)
          p->virial[k] += pair[i]->virial[k];
      if (p->eflag_atom)
        for (int j = 0; j < n; j++)
          p->eatom[j] += pair[i]->eatom[j];
      if (vflag_atom)
        for (int j = 0; j < n; j++)
          for (int k = 0; k < 6; k++)
            p->vatom[j][k] += pair[i]->vatom[j][k];
    }
  }

}

/*----------------------------------------------------------------------------*/

void FixSoftcoreEE::end_of_step()
{
  if (new_node != current_node) {

    int n = number_of_atoms();

    // Subtract old forces from the new ones:
    for (int i = 0; i < n; i++) {
      f_new[i][0] -= f_old[i][0];
      f_new[i][1] -= f_old[i][1];
      f_new[i][2] -= f_old[i][2];
    }

    // Reverse communicate force differences:
    if (force->newton_pair)
      comm->reverse_comm_fix(this,3);

    // Update forces:
    double **f = atom->f;
    for (int i = 0; i < n; i++) {
      f[i][0] += f_new[i][0];
      f[i][1] += f_new[i][1];
      f[i][2] += f_new[i][2];
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

/* ---------------------------------------------------------------------- */

void FixSoftcoreEE::change_node(int node)
{
  current_node = node;
  for (int i = 0; i < npairs; i++) {
    pair[i]->lambda = pair[i]->lambdanode[node];
    pair[i]->reinit();
  }
  if (downhill)
    downhill = current_node != 0;
  else
    downhill = current_node == gridsize - 1;
}

/* ----------------------------------------------------------------------
   Return the size of per-atom arrays (increase storage space if needed)
------------------------------------------------------------------------- */

int FixSoftcoreEE::number_of_atoms()
{
  int n = atom->nlocal;
  if (force->newton_pair)
    n += atom->nghost;
  if (n > nmax) {
    nmax = n;
    memory->grow(f_old,nmax,3,"fix_softcore_ee::f_old");
    memory->grow(f_new,nmax,3,"fix_softcore_ee::f_new");
  }
  return n;
}

/* ----------------------------------------------------------------------
   Return current node or downhill status
------------------------------------------------------------------------- */

double FixSoftcoreEE::compute_vector(int i)
{
  if (i == 0)
    return current_node;
  else if (i == 1)
    return downhill;
}

/* ---------------------------------------------------------------------- */

int FixSoftcoreEE::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = f_new[i][0];
    buf[m++] = f_new[i][1];
    buf[m++] = f_new[i][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixSoftcoreEE::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    f_new[j][0] += buf[m++];
    f_new[j][1] += buf[m++];
    f_new[j][2] += buf[m++];
  }
}

/* ---------------------------------------------------------------------- */
