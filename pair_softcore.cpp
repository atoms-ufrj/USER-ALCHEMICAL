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
#include "force.h"

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
  if (narg == 0)
    error->all(FLERR,"Illegal pair_modify command");

  int nkwds = 8;
  char *keyword[nkwds];
  keyword[0] = (char*)"alpha";
  keyword[1] = (char*)"n";
  keyword[2] = (char*)"p";
  keyword[3] = (char*)"lambda";
  keyword[4] = (char*)"set_grid";
  keyword[5] = (char*)"add_node";
  keyword[6] = (char*)"set_weights";
  keyword[7] = (char*)"add_weight";

  int ns = 0;
  int skip[narg];

  int m;
  int iarg = 0;
  while (iarg < narg) {

    // Search for a keyword:
    for (m = 0; m < nkwds; m++)
      if (strcmp(arg[iarg],keyword[m]) == 0)
        break;

    if (m < 4) { // alpha, n, p, or lambda:
      if (iarg+2 > narg) error->all(FLERR,"Illegal pair_modify command");
      double value = force->numeric(FLERR,arg[iarg+1]);
      if (m == 0) alpha = value;
      else if (m == 1) exponent_n = value;
      else if (m == 2) exponent_p = value;
      else {
        if ( (value < 0.0) || (value > 1.0) )
          error->all(FLERR,"Coupling parameter value out of range");
        lambda = value;
      }
      iarg += 2;
    }
    else if (m == 4) { // set_grid:
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal pair_modify command");
      int nodes = force->numeric(FLERR,arg[iarg+1]);
      gridsize = 0;
      if (iarg+2+nodes > narg) 
        error->all(FLERR,"Illegal pair_modify command");
      for (int i = 0; i < nodes; i++) 
        add_node_to_grid(force->numeric(FLERR,arg[iarg+2+i]),0.0);
      iarg += 2+nodes;
    }
    else if (m == 5) { // add_node:
      if (iarg+2 > narg) error->all(FLERR,"Illegal pair_modify command");
      add_node_to_grid(force->numeric(FLERR,arg[iarg+1]),0.0);
      iarg += 2;
    }
    else if (m == 6) { // set_weights:
      if (gridsize == 0)
        error->all(FLERR,"Softcore lambda grid has not been defined");
      if (iarg+1+gridsize > narg)
        error->all(FLERR,"Illegal pair_modify command");
      for (int i = 0; i < gridsize; i++)
        weight[i] = force->numeric(FLERR,arg[iarg+1+i]);
      iarg += 1+gridsize;
    }
    else if (m == 7) { // add_weight:
      if (iarg+3 > narg)
        error->all(FLERR,"Illegal pair_modify command");
      int i = force->numeric(FLERR,arg[iarg+1]);
      if ( (i < 1) || (i > gridsize))
        error->all(FLERR,"Node index out of bounds");
      weight[i-1] = force->numeric(FLERR,arg[iarg+2]);
      iarg += 3;
    }
    else // no keyword found - skip argument:
      skip[ns++] = iarg++;
  }

  // Call parent-class routine with skipped arguments:
  if (ns > 0) {
    for (int i = 0; i < ns; i++)
      arg[i] = arg[skip[i]];
      Pair::modify_params(ns, arg);
  }
}

/* ---------------------------------------------------------------------- */
