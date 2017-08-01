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
#include "pair_hybrid_softcore.h"
#include "pair_softcore.h"
#include "force.h"
#include "domain.h"
#include "error.h"
#include "string.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSoftcoreGrid::ComputeSoftcoreGrid(LAMMPS *lmp, int narg, char **arg) : 
  Compute(lmp, narg, arg)
{
  if (narg != 3)
    error->all(FLERR,"Illegal compute softcore/grid command");

  if (igroup)
    error->all(FLERR,"Compute softcore/grid must use group all");

  // Retrieve all lambda-related pair styles:
  PairHybridSoftcore *hybrid = dynamic_cast<PairHybridSoftcore*>(force->pair);
  if (hybrid) {
    pair = new class PairSoftcore*[hybrid->nstyles];
    npairs = 0;
    for (int i = 0; i < hybrid->nstyles; i++)
      if (pair[npairs] = dynamic_cast<class PairSoftcore*>(hybrid->styles[i]))
        npairs++;
    if (npairs == 0)
      error->all(FLERR,"Compute softcore/grid requires a softcore-type pair style");
  }
  else {
    pair = new class PairSoftcore*[1];
    if (!(pair[0] = dynamic_cast<class PairSoftcore*>(force->pair)))
      error->all(FLERR,"Compute softcore/grid requires a softcore-type pair style");
  }

  // Determine the number of nodes in the lambda grid:
  int nodes = pair[0]->gridsize;
  int all_equal = 1;
  for (int i = 1; i < npairs; i++)
    all_equal &= pair[i]->gridsize == nodes;
  if (!all_equal)
    error->all(FLERR,"compute softcore/grid: lambda grids have different numbers of nodes");
  if (nodes == 0)
    error->all(FLERR,"compute softcore/grid: no lambda grid has been defined");

  vector_flag = 1;
  size_vector = nodes;
  vector = new double[size_vector];
}

/* ---------------------------------------------------------------------- */

void ComputeSoftcoreGrid::compute_vector()
{
  // Compute lambda-related energy at every grid node:
  for (int i = 0; i < size_vector; i++)
    vector[i] = 0.0;
  for (int i = 0; i < npairs; i++) {
    if (pair[i]->gridsize != size_vector)
      error->all(FLERR,"compute softcore/grid: number of lambda nodes has changed");
    double node_energy[size_vector];
    if (!pair[i]->uptodate)
      pair[i]->compute_grid();
    MPI_Allreduce(pair[i]->evdwlnode,&node_energy[0],size_vector,MPI_DOUBLE,MPI_SUM,world);
    if (pair[i]->tail_flag) {
      double volume = domain->xprd*domain->yprd*domain->zprd;
      for (int j = 0; j < size_vector; j++)
        node_energy[j] += pair[i]->etailnode[j]/volume;
    }
    for (int j = 0; j < size_vector; j++)
      vector[j] += node_energy[j];
  }
}

