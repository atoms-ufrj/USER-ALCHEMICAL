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
#include "pair_hybrid_softcore.h"
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

  // Certify the use of pair style hybrid:
  if (strcmp(force->pair_style,"hybrid/softcore") != 0)
    error->all(FLERR,"fix softcore/ee: use of pair style hybrid/softcore is mandatory");

  // Look for lambda-related pair styles:
  PairHybridSoftcore *pair_hybrid = (PairHybridSoftcore *) force->pair;
  npairs = 0;
  for (int i = 0; i < pair_hybrid->nstyles; i++)
    if (strcmp(pair_hybrid->keywords[i],"lj/cut/softcore") == 0)
      npairs++;
  if (npairs == 0)
    error->all(FLERR,"fix softcore/ee: no pair styles associated to coupling parameter lambda");

  // Retrieve all lambda-related pair styles:
  pair = new class PairSoftcore*[npairs];
  compute_flag = new int[npairs];
  int j = 0;
  for (int i = 0; i < pair_hybrid->nstyles; i++)
    if (strcmp(pair_hybrid->keywords[i],"lj/cut/softcore") == 0) {
      pair[j++] = (class PairSoftcore *) pair_hybrid->styles[i];
    }

  // Allocate force buffer:
  nmax = atom->nlocal;
  if (force->newton_pair) nmax += atom->nghost;
  memory->create(f_old,nmax,3,"fix_softcore_ee::f_old");
  memory->create(f,nmax,3,"fix_softcore_ee::f");
  memory->create(eatom,nmax,"fix_softcore_ee::eatom");
  memory->create(vatom,nmax,6,"fix_softcore_ee::vatom");

  // Initialize random number generator:
  random = new RanPark(lmp,seed);
}

/* ---------------------------------------------------------------------- */

FixSoftcoreEE::~FixSoftcoreEE()
{
  memory->destroy(f_old);
  memory->destroy(f);
  memory->destroy(eatom);
  memory->destroy(vatom);
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

  // Disable computation of lambda-related pair styles:
  for (int i = 0; i < npairs; i++)
    pair[i]->compute_flag = 0;
}

/*----------------------------------------------------------------------------*/

void FixSoftcoreEE::pre_reverse(int eflag, int vflag)
{
  if (update->ntimestep % nevery) return;

  int n = number_of_atoms();
  PairHybridSoftcore *hybrid = (PairHybridSoftcore *) force->pair;

  // Restore original compute flags:
  for (int i = 0; i < npairs; i++)
    pair[i]->compute_flag = compute_flag[i];

  // Compute and store pair interactions using the current lambda value:
  for (int i = 0; i < n; i++)
    f_old[i][0] = f_old[i][1] = f_old[i][2] = 0.0;
  std::swap(atom->f,f_old);
  for (int i = 0; i < npairs; i++) {
    pair[i]->gridflag = 1;
    pair[i]->compute(eflag,vflag);
  }
  std::swap(f_old,atom->f);

  // Compute lambda-related energy at every grid node:
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
  node_changed = new_node != current_node;
  if (node_changed) {

    // Store previously computed forces, energies, and virials:
    for (int i = 0; i < n; i++) {
      this->f[i][0] = atom->f[i][0];
      this->f[i][1] = atom->f[i][1];
      this->f[i][2] = atom->f[i][2];
    }
    if (hybrid->eflag_global) {
      this->eng_vdwl = hybrid->eng_vdwl;
      this->eng_coul = hybrid->eng_coul;
    }
    if (hybrid->vflag_global)
      for (int k = 0; k < 6; k++)
        this->virial[k] = hybrid->virial[k];
    if (hybrid->eflag_atom)
      for (int j = 0; j < n; j++)
        this->eatom[j] = hybrid->eatom[j];
    if (hybrid->vflag_atom)
      for (int j = 0; j < n; j++)
        for (int k = 0; k < 6; k++)
          this->vatom[j][k] = hybrid->vatom[j][k];

    this->eflag = eflag;
    this->vflag = vflag;
  }

  // Add the terms corresponding to the current lambda value:
  for (int i = 0; i < n; i++) {
    atom->f[i][0] += f_old[i][0];
    atom->f[i][1] += f_old[i][1];
    atom->f[i][2] += f_old[i][2];
  }
  for (int i = 0; i < npairs; i++) {
    class PairSoftcore *ipair = pair[i];
    if (ipair->eflag_global) {
      hybrid->eng_vdwl += ipair->eng_vdwl;
      hybrid->eng_coul += ipair->eng_coul;
    }
    if (ipair->vflag_global)
      for (int k = 0; k < 6; k++)
        hybrid->virial[k] += ipair->virial[k];
    if (ipair->eflag_atom)
      for (int j = 0; j < n; j++)
        hybrid->eatom[j] += ipair->eatom[j];
    if (ipair->vflag_atom)
      for (int j = 0; j < n; j++)
        for (int k = 0; k < 6; k++)
          hybrid->vatom[j][k] += ipair->vatom[j][k];
  }
}

/*----------------------------------------------------------------------------*/

void FixSoftcoreEE::end_of_step()
{
  if (node_changed) {

    int n = number_of_atoms();
    class Pair *hybrid = force->pair;

    // Restore previous forces, energies, and virials:
    for (int i = 0; i < n; i++) {
      atom->f[i][0] = this->f[i][0];
      atom->f[i][1] = this->f[i][1];
      atom->f[i][2] = this->f[i][2];
    }
    if (hybrid->eflag_global) {
      hybrid->eng_vdwl = this->eng_vdwl;
      hybrid->eng_coul = this->eng_coul;
    }
    if (hybrid->vflag_global)
      for (int k = 0; k < 6; k++)
        hybrid->virial[k] = this->virial[k];
    if (hybrid->eflag_atom)
      for (int j = 0; j < n; j++)
        hybrid->eatom[j] = this->eatom[j];
    if (hybrid->vflag_atom)
      for (int j = 0; j < n; j++)
        for (int k = 0; k < 6; k++)
          hybrid->vatom[j][k] = this->vatom[j][k];

    // Change to the new node:
    change_node(new_node);

    // Compute and add pair interactions using the new lambda value:
    for (int i = 0; i < npairs; i++) {
      class PairSoftcore *ipair = pair[i];
      ipair->compute(eflag,vflag);
      ipair->uptodate = 1;
      if (ipair->eflag_global) {
        hybrid->eng_vdwl += ipair->eng_vdwl;
        hybrid->eng_coul += ipair->eng_coul;
      }
      if (ipair->vflag_global)
        for (int k = 0; k < 6; k++)
          hybrid->virial[k] += ipair->virial[k];
      if (ipair->eflag_atom)
        for (int j = 0; j < n; j++)
          hybrid->eatom[j] += ipair->eatom[j];
      if (ipair->vflag_atom)
        for (int j = 0; j < n; j++)
          for (int k = 0; k < 6; k++)
            hybrid->vatom[j][k] += ipair->vatom[j][k];
    }

    // Reverse communicate forces:
    if (force->newton_pair)
      comm->reverse_comm();

    // Perform post-force actions:
    if (modify->n_post_force)
      modify->post_force(vflag);
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
    memory->grow(f,nmax,3,"fix_softcore_ee::f");
    memory->grow(eatom,nmax,"fix_softcore_ee::eatom");
    memory->grow(vatom,nmax,6,"fix_softcore_ee::vatom");
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
