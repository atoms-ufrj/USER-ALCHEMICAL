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
   Contributing author: Charlles Abreu (abreu@eq.ufrj.br)
                        Applied Thermodynamics & Molecular Simulation (ATOMS)
                        Federal University of Rio de Janeiro / Brazil
------------------------------------------------------------------------- */

#include "mpi.h"
#include "compute_softcore_grid.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "domain.h"
#include "error.h"
#include "pair_hybrid.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSoftcoreGrid::ComputeSoftcoreGrid(LAMMPS *lmp, int narg, char **arg) : 
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute softcore/grid command");
  if (igroup) error->all(FLERR,"Compute softcore/grid must use group all");

  int dim;
  int *size = (int*) force->pair->extract("gridsize",dim);
  size_vector = *size;

  printf("%d GRIDSIZE \n", size_vector);  

  if (size_vector == 0)
  error->all(FLERR,"Compute softcore/grid error: lambda grid not defined");
  peflag = 1;
  timeflag = 1;
  scalar_flag = 0;
  vector_flag = 1;
  extvector = 1;
  vector = new double[size_vector];
}

/* ---------------------------------------------------------------------- */

void ComputeSoftcoreGrid::compute_vector()
{
 // invoked_scalar = update->ntimestep;
 // if (update->eflag_global != invoked_scalar)
  //  error->all(FLERR,"Energy was not tallied on needed timestep");

  int dim;
  double *grid_energy = (double *) force->pair->extract("energy_grid",dim);
  MPI_Allreduce(grid_energy,vector,size_vector,MPI_DOUBLE,MPI_SUM,world);
 
  int tail_flag;
  force->pair->extract("tail_flag",tail_flag);
  if (tail_flag) {
    double *etailnode = (double *) force->pair->extract("etailnode",dim);
    double volume = domain->xprd * domain->yprd * domain->zprd;
    for (int k = 0; k < size_vector; k++)
      vector[k] += etailnode[k]/volume;
  }
}

