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
#include "pair_alchemical.h"
#include "memory.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairAlchemical::PairAlchemical(LAMMPS *lmp) : Pair(lmp)
{
  alpha = 0.5;
  exponent_n = 1.0;
  exponent_p = 1.0;
  lambda = 1.0;
  efactor = 1.0;
  detaildl = 0.0;

  gridflag = 0;
  gridsize = 0;
  grid_uptodate = 0;
  derivflag = 0;
  deriv_uptodate = 0;
  allocate();
}

/* ---------------------------------------------------------------------- */

PairAlchemical::~PairAlchemical()
{
  memory->destroy(lambdanode);
  memory->destroy(evdwlnode);
  memory->destroy(ecoulnode);
  memory->destroy(ecoulnode);
  memory->destroy(etailnode);
}
/* ---------------------------------------------------------------------- */

void PairAlchemical::init_style()
{
  // print grid information:
  if ( (gridsize > 0) && (comm->me == 0) ) {
    if (screen) fprintf(screen,"Lambda grid: (");
    if (logfile) fprintf(logfile,"Lambda grid: (");
    for (int k = 0; k < gridsize-1; k++) {
      if (screen) fprintf(screen,"%g; ",lambdanode[k]);
      if (logfile) fprintf(logfile,"%g; ",lambdanode[k]);
    }
    if (screen) fprintf(screen,"%g)\n",lambdanode[gridsize-1]);
    if (logfile) fprintf(logfile,"%g)\n",lambdanode[gridsize-1]);
  }
}

/* ---------------------------------------------------------------------- */

void PairAlchemical::allocate()
{
  memory->create(lambdanode,0,"pair_softcore:lambdanode");
  memory->create(evdwlnode,0,"pair_softcore:evdwlnode");
  memory->create(ecoulnode,0,"pair_softcore:ecoulnode");
  memory->create(ecoulnode,0,"pair_softcore:ecoulnode");
  memory->create(etailnode,0,"pair_softcore:etailnode");
}

/* ----------------------------------------------------------------------
   adds a new node to the lambda grid, in increasing order of lambdas
------------------------------------------------------------------------- */

void PairAlchemical::add_node_to_grid(double lambda_value)
{
  int i,j;

  if ( (lambda_value < 0.0) || (lambda_value > 1.0) )
    error->all(FLERR,"Coupling parameter value is out of range");

  double *backup = new double[gridsize];
  memcpy(backup,lambdanode,sizeof(double)*gridsize);

  gridsize++;
  memory->grow(lambdanode,gridsize,"pair_softcore:lambdanode");
  memory->grow(evdwlnode,gridsize,"pair_softcore:evdwlnode");
  memory->grow(ecoulnode,gridsize,"pair_softcore:ecoulnode");
  memory->grow(etailnode,gridsize,"pair_softcore:etailnode");

  j = 0;
  for (i = 0; i < gridsize-1; i++)
    if (backup[i] < lambda_value) {
      lambdanode[i] = backup[i];
      j = i+1;
    } else
      lambdanode[i+1] = backup[i];
  lambdanode[j] = lambda_value;
  evdwlnode[j] = ecoulnode[j] = etailnode[j] = 0.0;
  delete [] backup;
}

/* ---------------------------------------------------------------------- */

void PairAlchemical::modify_params(int narg, char **arg)
{
  if (narg == 0)
    error->all(FLERR,"Illegal pair_modify command");

  int nkwds = 6;
  char *keyword[nkwds];
  keyword[0] = (char*)"alpha";
  keyword[1] = (char*)"n";
  keyword[2] = (char*)"p";
  keyword[3] = (char*)"lambda";
  keyword[4] = (char*)"set_grid";
  keyword[5] = (char*)"add_node";

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
      efactor = pow(lambda, exponent_n);
      diff_efactor = exponent_n*pow(lambda, exponent_n - 1.0);
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
        add_node_to_grid(force->numeric(FLERR,arg[iarg+2+i]));
      iarg += 2+nodes;
    }
    else if (m == 5) { // add_node:
      if (iarg+2 > narg) error->all(FLERR,"Illegal pair_modify command");
      add_node_to_grid(force->numeric(FLERR,arg[iarg+1]));
      iarg += 2;
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

void PairAlchemical::write_restart(FILE *fp)
{
  fwrite(&gridsize,sizeof(int),1,fp);
  fwrite(lambdanode,sizeof(double),gridsize,fp);
}

/* ---------------------------------------------------------------------- */

void PairAlchemical::read_restart(FILE *fp)
{
  if (comm->me == 0) fread(&gridsize,sizeof(int),1,fp);
  MPI_Bcast(&gridsize,1,MPI_INT,0,world);
  allocate();
  if (comm->me == 0) fread(lambdanode,sizeof(double),gridsize,fp);
  MPI_Bcast(lambdanode,gridsize,MPI_DOUBLE,0,world);
}
