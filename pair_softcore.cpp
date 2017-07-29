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
   Contributing author: Charlles Abreu
                        Chemical Engineering Department
                        Federal University of Rio de Janeiro, Brazil
------------------------------------------------------------------------- */

#include "string.h"
#include "pair_softcore.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSoftcore::PairSoftcore(LAMMPS *lmp) : Pair(lmp)
{
  alpha = 0.5;
  exponent_n = 1.0;
  exponent_p = 1.0;
  lambda = 1.0;

  gridflag = 0;
  gridsize = 0;
  uptodate = 0;
  memory->create(lambdanode,0,"pair_softcore:lambdanode");
  memory->create(evdwlnode,0,"pair_softcore:evdwlnode");
  memory->create(etailnode,0,"pair_softcore:etailnode");
  memory->create(weight,0,"pair_softcore:weight");
}

/* ---------------------------------------------------------------------- */

PairSoftcore::~PairSoftcore()
{
  memory->destroy(linkflag);
  memory->destroy(lambdanode);
  memory->destroy(evdwlnode);
  memory->destroy(etailnode);
  memory->destroy(weight);
}

/* ----------------------------------------------------------------------
   adds a new node to the lambda grid, in increasing order of lambdas
------------------------------------------------------------------------- */

void PairSoftcore::add_node_to_grid(double lambda_value, double weight_value)
{
  int i,j;
  double *backup = new double[gridsize];

  if ( (lambda_value < 0.0) || (lambda_value > 1.0) )
    error->all(FLERR,"Coupling parameter value is out of range");
  memcpy(backup,lambdanode,sizeof(double)*gridsize);
  gridsize++;
  memory->grow(lambdanode,gridsize,"pair_softcore:lambdanode");
  j = 0;
  for (i = 0; i < gridsize-1; i++)
    if (backup[i] < lambda_value) {
      lambdanode[i] = backup[i];
      j = i+1;
    } else
      lambdanode[i+1] = backup[i];
  lambdanode[j] = lambda_value;

  memcpy(backup,weight,sizeof(double)*(gridsize-1));
  memory->grow(weight,gridsize,"pair_softcore:weight");
  for (i = 0; i < j; i++)
    weight[i] = backup[i];
  weight[j] = weight_value;
  for (i = j+1; i < gridsize; i++)
    weight[i] = backup[i-1];

}

/* ---------------------------------------------------------------------- */

void PairSoftcore::modify_params(int narg, char **arg)
{
}

/* ---------------------------------------------------------------------- */

