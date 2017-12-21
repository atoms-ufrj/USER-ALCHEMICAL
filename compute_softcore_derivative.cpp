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
#include "compute_softcore_derivative.h"
#include "pair_hybrid_softcore.h"
#include "pair_alchemical.h"
#include "force.h"
#include "domain.h"
#include "error.h"
#include "string.h"
#include "atom.h"
#include "memory.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSoftcoreDerivative::ComputeSoftcoreDerivative(LAMMPS *lmp, int narg, char **arg) : 
  Compute(lmp, narg, arg)
{
  if (narg != 3)
    error->all(FLERR,"Illegal compute softcore/derivative command");

  if (igroup)
    error->all(FLERR,"Compute softcore/derivative must use group all");

  // Retrieve all lambda-related pair styles:
  PairHybridSoftcore *hybrid = dynamic_cast<PairHybridSoftcore*>(force->pair);
  if (hybrid) {
    pair = new class PairAlchemical*[hybrid->nstyles];
    npairs = 0;
    for (int i = 0; i < hybrid->nstyles; i++)
      if (pair[npairs] = dynamic_cast<class PairAlchemical*>(hybrid->styles[i]))
        npairs++;
    if (npairs == 0)
      error->all(FLERR,"Compute softcore/derivative requires a softcore-type pair style");
  }
  else {
    pair = new class PairAlchemical*[1];
    if (!(pair[0] = dynamic_cast<class PairAlchemical*>(force->pair)))
      error->all(FLERR,"Compute softcore/derivative requires a softcore-type pair style");
  }

  scalar_flag = 1;
  extscalar = 1;
}

/* ---------------------------------------------------------------------- */

double ComputeSoftcoreDerivative::compute_scalar()
{
  // Compute lambda-derivative of energy for every pair style:
  double one = 0.0;
  for (int i = 0; i < npairs; i++)
    one += pair[i]->derivative();

  MPI_Allreduce(&one,&scalar,1,MPI_DOUBLE,MPI_SUM,world);

  double volume = domain->xprd*domain->yprd*domain->zprd;
  for (int i = 0; i < npairs; i++)
    if (pair[i]->tail_flag)
      scalar += pair[i]->detaildl/volume;

  return scalar;
}
