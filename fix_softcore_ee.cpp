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

// TODO:
// 1) Eliminar saíde em arquivo (usar somente saída via compute_vector)
// 2) Eliminar compute softcore/grid (fazer diretamente neste fix)
// 3) Apagar recálculo de energia após troca de lambda

#include "math.h"
#include "fix_softcore_ee.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "error.h"
#include "comm.h"
#include "domain.h"
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
#include "neighbor.h"
#include "fix_nh.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSoftcoreEE::FixSoftcoreEE(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  int dim;
  int *size = (int *) force->pair->extract("gridsize",dim);
  gridsize = size[0];
  if (gridsize == 0)
    error->all(FLERR,"fix softcore/ee: no lambda grid defined");
  if (narg < 7)
    error->all(FLERR,"Illegal fix softcore/ee command");
  nevery = force->numeric(FLERR,arg[3]);
  if (nevery <= 0)
    error->all(FLERR,"Illegal fix softcore/ee command");
  acfreq = force->numeric(FLERR,arg[4]);
  if (acfreq <= 0)
   error->all(FLERR,"Illegal fix softcore/ee command");
  seed = force->numeric(FLERR,arg[5]);
  if (seed <= 0)
    error->all(FLERR,"Illegal fix softcore/ee command");
  minus_beta = -1.0/(force->boltz*force->numeric(FLERR,arg[6]));
  
  int iarg = 7;
  ee_file = NULL;
  idump = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"dump") == 0) {
      if (iarg+3 > narg)
        error->all(FLERR,"Illegal fix softcore/ee command");
      idump = force->numeric(FLERR,arg[iarg+1]);
      if (idump <= 0)
        error->all(FLERR,"Illegal fix softcore/ee command");
      int n = strlen(arg[iarg+2]) + 1;
      char *string = new char[n];
      strcpy(string,arg[iarg+2]);
      if (comm->me == 0)
        ee_file = fopen(arg[iarg+2],"w");
      iarg += 3;
    }
    else
      error->all(FLERR,"Illegal fix softcore/ee command");
  }
  add_new_compute();
  ratiocriteria = 1.0/acfreq;
  scalar_flag = 1;
  global_freq = 1;

  // set flags for arrays to clear in force_clear()

  torqueflag = extraflag = 0;
  if (atom->torque_flag) torqueflag = 1;
  if (atom->avec->forceclearflag) extraflag = 1;

  // orthogonal vs triclinic simulation box

  triclinic = domain->triclinic;
}

/* ---------------------------------------------------------------------- */

FixSoftcoreEE::~FixSoftcoreEE()
{
  if (ee_file)
    fclose(ee_file);
}

/* ---------------------------------------------------------------------- */

int FixSoftcoreEE::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= END_OF_STEP;
  return mask;
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

  lambda_arg[0] = (char*)"pair";
  lambda_arg[1] = (char*)"lj/cut/softcore";
  lambda_arg[2] = (char*)"lambda";
  lambda_arg[3] = new char[18];

  change_node(0);
  downhill = 0;

  random = new RanPark(lmp,seed);

  if (ee_file) {
    fprintf(ee_file,"step node lambda downhill");
    for (int k = 0; k < gridsize; k++) {

      fprintf(ee_file," energy[%d]",k);
    fprintf(ee_file,"\n");
    }
  }

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
  if (idump != 0) 
    calculate = calculate || (step % idump == 0);

  if (calculate)
    flag = (int *) force->pair->extract("gridflag",dim);
}

/*----------------------------------------------------------------------------*/
void FixSoftcoreEE::end_of_step()
{
  int new_node;
  int dim,*tail_flag,k;
  double *grid_energy,*etailnode,*energy,volume;

  grid_energy = (double *) force->pair->extract("energy_grid",dim);
  energy = new double[gridsize];
  MPI_Allreduce(grid_energy,energy,gridsize,MPI_DOUBLE,MPI_SUM,world);
  tail_flag = (int *) force->pair->extract("tail_flag",dim);
  if (tail_flag[0]) {
    etailnode = (double *) force->pair->extract("etailnode",dim);
    volume = domain->xprd * domain->yprd * domain->zprd;
    for (k = 0; k < gridsize; k++)
      energy[k] += etailnode[k]/volume;
  }

  double P[gridsize], umax, usum;
  P[0] = minus_beta*energy[0] + weight[0];
  umax = P[0];
  
  for (k = 1; k < gridsize; k++) {
    P[k] = minus_beta*energy[k] + weight[k];
    umax = MAX(umax,P[k]);
  }
  usum = 0.0;
  for (k = 0; k < gridsize; k++) {
    P[k] = exp(P[k]-umax);
    usum += P[k];
  }
  for (k = 0; k < gridsize; k++) P[k] /= usum;
  
  double r = random->uniform();
  new_node = 0;
  double acc = P[0];
  while (r > acc) {
    new_node++;
    acc += P[new_node];
  }
  MPI_Bcast(&new_node,1,MPI_INT,0,world);

  if (new_node != current_node) {
    change_node(new_node);
    force_clear();
    int eflag = 1; 
    int vflag = 1;

    if (force->pair && force->pair->compute_flag) {
      force->pair->compute(eflag,vflag);
      timer->stamp(Timer::PAIR);
    }

    if (atom->molecular) {
      if (force->bond) force->bond->compute(eflag,vflag);
      if (force->angle) force->angle->compute(eflag,vflag);
      if (force->dihedral) force->dihedral->compute(eflag,vflag);
      if (force->improper) force->improper->compute(eflag,vflag);
      timer->stamp(Timer::BOND);
    }

    if (force->kspace && force->kspace->compute_flag) {
      force->kspace->compute(eflag,vflag);
      timer->stamp(Timer::KSPACE);
    }

     // reverse communication of forces

    if (force->newton) {
      comm->reverse_comm();
      timer->stamp(Timer::COMM);
    }
      
    //MODIFIED :: GRID ENERGY CALCULATED AFTER NODE CHANGE
    grid_energy = (double *) force->pair->extract("energy_grid",dim);
    energy = new double[gridsize];
    MPI_Allreduce(grid_energy,energy,gridsize,MPI_DOUBLE,MPI_SUM,world);
    tail_flag = (int *) force->pair->extract("tail_flag",dim);
    if (tail_flag[0]) {
      etailnode = (double *) force->pair->extract("etailnode",dim);
      volume = domain->xprd * domain->yprd * domain->zprd;
      for (k = 0; k < gridsize; k++)  energy[k] += etailnode[k]/volume;
    }      
  }
  double PE = pe->compute_scalar();
  int step;
  step = update->ntimestep;

  if ( ee_file && (step % idump == 0) ) {
    fprintf(ee_file,"%d %d %g %d %g", 
           step, current_node, lambdanode[current_node], downhill, PE);
    for (int k = 0; k < gridsize; k++)
      fprintf(ee_file," %g",energy[k]);
    fprintf(ee_file,"\n");
  }
  int nextstep = update->ntimestep + nevery;
  if (nextstep <= update->laststep) 
    pe->addstep(nextstep);
}

/* ---------------------------------------------------------------------- */

void FixSoftcoreEE::change_node(int node)
{
  current_node = node;
  sprintf(lambda_arg[3],"%18.16f",lambdanode[node]);
  force->pair->modify_params(4,lambda_arg);
  force->pair->reinit();
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
