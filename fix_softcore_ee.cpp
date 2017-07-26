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
  int dim;
  int *size = (int *) force->pair->extract("gridsize",dim);
  gridsize = *size;
  if (gridsize == 0)
    error->all(FLERR,"fix softcore/ee: no lambda grid defined");

  if (narg < 6)
    error->all(FLERR,"Illegal fix softcore/ee command");

  nevery = force->numeric(FLERR,arg[3]);
  if (nevery <= 0)
    error->all(FLERR,"Illegal fix softcore/ee command");

  seed = force->numeric(FLERR,arg[4]);
  if (seed <= 0)
    error->all(FLERR,"Illegal fix softcore/ee command");
  minus_beta = -1.0/(force->boltz*force->numeric(FLERR,arg[5]));

  add_new_compute();
  scalar_flag = 1;
  global_freq = 1;

  // set flags for arrays to clear in force_clear()
  torqueflag = extraflag = 0;
  if (atom->torque_flag) torqueflag = 1;
  if (atom->avec->forceclearflag) extraflag = 1;

  // orthogonal vs triclinic simulation box
  triclinic = domain->triclinic;

  // Retrieve lambda-related pair styles:
  if (strcmp(force->pair_style,"hybrid/ee") == 0) {
    class PairHybridEE *pair_hybrid;
    pair_hybrid = (PairHybridEE *) force->pair;
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

    // Allocate force buffer:
    nmax = atom->nlocal;
    if (force->newton_pair) nmax += atom->nghost;
    memory->create(f_new,nmax,3,"fix_softcore_ee::f_new");
}

/* ---------------------------------------------------------------------- */

FixSoftcoreEE::~FixSoftcoreEE()
{
  modify->delete_compute("ee_pe");
  memory->destroy(f_new);
}

/* ---------------------------------------------------------------------- */

int FixSoftcoreEE::setmask()
{
  return INITIAL_INTEGRATE | PRE_FORCE | END_OF_STEP;
}

/* ---------------------------------------------------------------------- */

void FixSoftcoreEE::init()
{
  int dim;
  weight = (double *) force->pair->extract("weight",dim);
  lambdanode = (double *) force->pair->extract("lambdanode",dim);

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen,"Expanded ensemble weights: (");
      for (int k = 0; k < gridsize-1; k++)
        fprintf(screen,"%g; ",weight[k]);
      fprintf(screen,"%g)\n",weight[gridsize-1]);
    }
    if (logfile) {
      fprintf(logfile,"Expanded ensemble weights: (");
      for (int k = 0; k < gridsize-1; k++)
        fprintf(logfile,"%g; ",weight[k]);
      fprintf(logfile,"%g)\n",weight[gridsize-1]);
    }
  }

  lambda_arg[0] = (char*)"lambda";
  lambda_arg[1] = new char[18];

  change_node(0);
  downhill = 0;

  random = new RanPark(lmp,seed);
}
/* ----------------------------------------------------------------------
 activate computes
------------------------------------------------------------------------- */

void FixSoftcoreEE::setup(int vflag)
{
  pe->compute_scalar();
  // Activate potential energy and other necessary calculations:
  int nextstep = update->ntimestep + nevery;
  pe->addstep(nextstep);
}

/* ---------------------------------------------------------------------- */

void FixSoftcoreEE::initial_integrate(int vflag)
{
  int dim,*flag,step;
  step = update->ntimestep;
  calculate = step % nevery == 0;

  if (calculate)
    flag = (int *) force->pair->extract("gridflag",dim);

  vflag_local = vflag;
}

/*----------------------------------------------------------------------------*/

void FixSoftcoreEE::pre_force(int vflag)
{
  if (update->ntimestep % nevery == 0) {

    int i;
    double umax, usum, acc, r;
    double P[gridsize], energy[gridsize] = {0.0};

    for (i = 0; i < npairs; i++)
      add_energies( &energy[0], pair[i] );

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
    new_node = 0;
    while (r > acc) {
      new_node++;
      acc += P[new_node];
    }
    MPI_Bcast(&new_node,1,MPI_INT,0,world);

    node_changed = new_node != current_node;
    if (node_changed) {

      int n = atom->nlocal;
      if (force->newton_pair) n += atom->nghost;
      if (n > nmax) {
        nmax = n;
        memory->grow(f_new,nmax,3,"fix_softcore_ee::f_new");
      }

      for (i = 0; i < n; i++)
        f_new[i][0] = f_new[i][1] = f_new[i][2] = 0.0;

      for (i = 0; i < npairs; i++)
        pair[i]->compute(0,0);
  
      change_node(new_node);

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

    // Prepare for calculations at next expanded ensemble step:
    int nextstep = update->ntimestep + nevery;
    if (nextstep <= update->laststep) 
      pe->addstep(nextstep);
  }
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
  sprintf(lambda_arg[1],"%18.16f",lambdanode[node]);
  for (int i = 0; i < npairs; i++) {
    pair[i]->modify_params(2,lambda_arg);
    pair[i]->reinit();
  }
  if (downhill)
    downhill = current_node != 0;
  else
    downhill = current_node == gridsize - 1;
}

/* ----------------------------------------------------------------------
   Return node
------------------------------------------------------------------------- */

double FixSoftcoreEE::compute_scalar()
{
  
  return current_node;
}
/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   clear other arrays as needed
------------------------------------------------------------------------- */

void FixSoftcoreEE::force_clear()
{
  size_t nbytes;

  // clear force on all particles
  // if either newton flag is set, also include ghosts
  // when using threads always clear all forces.

  int nlocal = atom->nlocal;

  if (neighbor->includegroup == 0) {
    nbytes = sizeof(double) * nlocal;
    if (force->newton) nbytes += sizeof(double) * atom->nghost;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
      if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

  // neighbor includegroup flag is set
  // clear force only on initial nfirst particles
  // if either newton flag is set, also include ghosts

  } else {
    nbytes = sizeof(double) * atom->nfirst;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
      if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

    if (force->newton) {
      nbytes = sizeof(double) * atom->nghost;

      if (nbytes) {
        memset(&atom->f[nlocal][0],0,3*nbytes);
        if (torqueflag) memset(&atom->torque[nlocal][0],0,3*nbytes);
        if (extraflag) atom->avec->force_clear(nlocal,nbytes);
      }
    }
  }
}
/* ---------------------------------------------------------------------- */
void FixSoftcoreEE::add_new_compute()
{
  char **newarg = new char*[3];

  // Potential energy:
  newarg[0] = (char *) "ee_pe";
  newarg[1] = (char *) "all";
  newarg[2] = (char *) "pe";
  modify->add_compute(3,newarg);
  pe = modify->compute[modify->ncompute-1];

  delete [] newarg;
}
