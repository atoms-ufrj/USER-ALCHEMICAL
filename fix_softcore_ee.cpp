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

  // Check if this fix preceeds all fixes with initial_integrate:
  for (int i = 0; i < modify->nfix; i++)
    if (modify->fmask[i] && INITIAL_INTEGRATE)
      error->all(FLERR,"fix softcore/ee must preceed all time integration fixes");

  // Set fix softcore/ee properties:
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;

  // Certify the use of pair style hybrid:
  PairHybridSoftcore *hybrid = dynamic_cast<PairHybridSoftcore*>(force->pair);
  if (!hybrid)
    error->all(FLERR,"fix softcore/ee: use of pair style hybrid/softcore is mandatory");

  // Look for lambda-related pair styles:
  pair = new class PairSoftcore*[hybrid->nstyles];
  npairs = 0;
  for (int i = 0; i < hybrid->nstyles; i++) {
    pair[npairs] = dynamic_cast<class PairSoftcore*>(hybrid->styles[i]);
    if (pair[npairs])
      npairs++;
  }
  if (npairs == 0)
    error->all(FLERR,"fix softcore/ee: no pair styles associated to coupling parameter lambda");

  // Allocate array for storing compute flags of softcore pair styles:
  compute_flag = new int[npairs];

  // Allocate force buffer:
  nmax = atom->nlocal;
  if (force->newton_pair) nmax += atom->nghost;
  memory->create(f_soft,nmax,3,"fix_softcore_ee::f_soft");
  memory->create(f,nmax,3,"fix_softcore_ee::f");
  memory->create(eatom,nmax,"fix_softcore_ee::eatom");
  memory->create(vatom,nmax,6,"fix_softcore_ee::vatom");

  // Initialize random number generator:
  random = new RanPark(lmp,seed);
}

/* ---------------------------------------------------------------------- */

FixSoftcoreEE::~FixSoftcoreEE()
{
  memory->destroy(f_soft);
  memory->destroy(f);
  memory->destroy(eatom);
  memory->destroy(vatom);
  if (weight) memory->destroy(weight);
  delete [] pair;
  delete [] compute_flag;
  delete random;
}

/* ---------------------------------------------------------------------- */

int FixSoftcoreEE::setmask()
{
  return INITIAL_INTEGRATE | PRE_REVERSE | END_OF_STEP;
}

/* ----------------------------------------------------------------------
   Before starting a run, certify that all softcore pair styles have
   equally sized lambda grids. Then, check if the corresponding number
   of weights has been specified. If there are no weights at all, specify
   all weights as zero. Finally, call change_node(0) so that the all pair
   styles are reinitialized with the corresponding lambda value.
------------------------------------------------------------------------- */

void FixSoftcoreEE::init()
{
  // Determine the number of nodes in the lambda grid:
  int nodes = pair[0]->gridsize;
  int all_equal = 1;
  for (int i = 1; i < npairs; i++)
    all_equal &= pair[i]->gridsize == nodes;
  if (!all_equal)
    error->all(FLERR,"fix softcore/ee: pair styles have different numbers of nodes");
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

  // Store compute flags of lambda-related pair styles:
  for (int i = 0; i < npairs; i++)
    compute_flag[i] = pair[i]->compute_flag;

  // Start simulation at the first lambda node:
  downhill = 0;
  change_node(0);
  must_change_node = 0;
}

/* ----------------------------------------------------------------------
   Enact node changing if this was decided in the previous time step.
   This is done before the initial_integrate routines of integration
   methods are executed.
------------------------------------------------------------------------- */

void FixSoftcoreEE::initial_integrate(int vflag)
{
  int cycle = update->ntimestep % nevery;

  if (cycle == 1 && must_change_node) { // Node change has been decided at the lattest step

    int n = number_of_atoms();
    class Pair *hybrid = force->pair;

    // Restore lambda-free forces, energies, and virials previously stored:
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
      ipair->compute(this->eflag,this->vflag);
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
      modify->post_force(this->vflag);
  }
  else if (cycle == 0) // Node change will be tested at this step
    for (int i = 0; i < npairs; i++)
      pair[i]->compute_flag = 0;
}

/* ----------------------------------------------------------------------
   After lambda-free forces, energies, and virials have been computed,
   compute the softcore pair interactions with the current lambda value
   and decide if a node change must be carried out. If so, store the
   lambda-free properties before adding the newly computed terms.
------------------------------------------------------------------------- */

void FixSoftcoreEE::pre_reverse(int eflag, int vflag)
{
  if (update->ntimestep % nevery) return;

  int n = number_of_atoms();
  PairHybridSoftcore *hybrid = (PairHybridSoftcore *) force->pair;

  // Compute and store pair interactions using the current lambda value:
  for (int i = 0; i < n; i++)
    f_soft[i][0] = f_soft[i][1] = f_soft[i][2] = 0.0;
  std::swap(atom->f,f_soft);
  for (int i = 0; i < npairs; i++) {
    pair[i]->gridflag = 1;
    pair[i]->compute(eflag,vflag);
  }
  std::swap(f_soft,atom->f);

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
  must_change_node = new_node != current_node;
  if (must_change_node) {

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
    atom->f[i][0] += f_soft[i][0];
    atom->f[i][1] += f_soft[i][1];
    atom->f[i][2] += f_soft[i][2];
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

  // Restore compute flags:
  for (int i = 0; i < npairs; i++)
    pair[i]->compute_flag = compute_flag[i];
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

/* ----------------------------------------------------------------------
   Perform lambda node changing
------------------------------------------------------------------------- */

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
    memory->grow(f_soft,nmax,3,"fix_softcore_ee::f_soft");
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
