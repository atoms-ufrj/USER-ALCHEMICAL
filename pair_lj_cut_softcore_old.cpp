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

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_lj_cut_softcore_old.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "domain.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairLJCutSoftcoreOld::PairLJCutSoftcoreOld(LAMMPS *lmp) : PairSoftcore(lmp)
{
  respa_enable = 1;

//  alpha = 0.5;
//  exponent_n = 1.0;
//  exponent_p = 1.0;
//  lambda = 1.0;

//  gridsize = 0;
//  memory->create(lambdanode,0,"pair:lambdanode");
//  memory->create(evdwlnode,0,"pair:evdwlnode");
//  memory->create(etailnode,0,"pair:etailnode");
//  memory->create(weight,0,"pair:weight");
}

/* ---------------------------------------------------------------------- */

PairLJCutSoftcoreOld::~PairLJCutSoftcoreOld()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(asq);
    memory->destroy(offset);
    memory->destroy(linkflag);

    memory->destroy(lj3n);
    memory->destroy(lj4n);
    memory->destroy(asqn);
    memory->destroy(offn);

    memory->destroy(lambdanode);
    memory->destroy(evdwlnode);
    memory->destroy(etailnode);
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutSoftcoreOld::compute(int eflag, int vflag)
{
  int i,j,k,ii,jj,inum,jnum,itype,jtype,fullcount;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r6,sinv,forcelj,factor_lj,evdwlk;
  int *ilist,*jlist,*numneigh,**firstneigh;

  if (eflag || vflag)
    ev_setup(eflag,vflag);
  else
    evflag = vflag_fdotr = 0;

  if (eflag) {
    evdwl = 0.0;
    if (gridsize)
      for (k = 0; k < gridsize; k++)
        evdwlnode[k] = 0.0;
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {

      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
      
        r6 = rsq*rsq*rsq;
        sinv = 1.0/(r6 + asq[itype][jtype]);
        forcelj = r6*sinv*sinv*(lj1[itype][jtype]*sinv - lj2[itype][jtype]);
	fpair = factor_lj*forcelj/rsq;

	f[i][0] += delx*fpair;
	f[i][1] += dely*fpair;
	f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
	  evdwl = factor_lj*(sinv*(lj3[itype][jtype]*sinv - lj4[itype][jtype])
                  - offset[itype][jtype]);

          if (gridsize && linkflag[itype][jtype] != 0)
            for (k = 0; k < gridsize; k++) {
              sinv = 1.0/(r6 + asqn[itype][jtype][k]);
              evdwlk = factor_lj*(sinv*(lj3n[itype][jtype][k]*sinv -
	               lj4n[itype][jtype][k]) - offn[itype][jtype][k]);
              if (newton_pair || j < nlocal)
                evdwlnode[k] += evdwlk;
              else
                evdwlnode[k] += 0.5*evdwlk;
            }
        }

	if (evflag) ev_tally(i,j,nlocal,newton_pair,evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays 
------------------------------------------------------------------------- */

void PairLJCutSoftcoreOld::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(asq,n+1,n+1,"pair:asq");
  memory->create(offset,n+1,n+1,"pair:offset");

  // all pairs are directly linked to lambda by default:
  memory->create(linkflag,n+1,n+1,"pair:linkflag");
  memory->create(linkedtype,n+1,"pair:linkedtype");
  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++)
      linkflag[i][j] = 1;
    linkedtype[i] = 1;
  }

  memory->create(lj3n,n+1,n+1,gridsize,"pair:lj3n");
  memory->create(lj4n,n+1,n+1,gridsize,"pair:lj4n");
  memory->create(asqn,n+1,n+1,gridsize,"pair:asqn");
  memory->create(offn,n+1,n+1,gridsize,"pair:offn");
}

/* ----------------------------------------------------------------------
   global settings 
------------------------------------------------------------------------- */

void PairLJCutSoftcoreOld::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);

  // reset cutoffs that have been explicitly set
  
  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
	if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJCutSoftcoreOld::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 5) 
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);

  double cut_one = cut_global;
  if (narg == 5) cut_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCutSoftcoreOld::init_style()
{
  // request regular neighbor lists
  neighbor->request(this);

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

  int n = atom->ntypes;
  memory->grow(lj3n,n+1,n+1,gridsize,"pair:lj3n");
  memory->grow(lj4n,n+1,n+1,gridsize,"pair:lj4n");
  memory->grow(asqn,n+1,n+1,gridsize,"pair:asqn");
  memory->grow(offn,n+1,n+1,gridsize,"pair:offn");

  for (int i = 1; i <= n; i++)
    linkedtype[i] = 0;

}

/* ---------------------------------------------------------------------- */

double PairLJCutSoftcoreOld::atanx_x(double x)
{
  double y,z,d,s,t;
  y = -x*x;
  z = d = s = 1.0;
  do {
    z *= y;
    d += 2.0;
    t = z/d;
    s += t;
  } while (t*t > 1.e-32);
  return s;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJCutSoftcoreOld::init_one(int i, int j)
{
  int k;
  double phi,phi0,sig2,sig6,sig12,efactor,rc3,rc6,scinv,TwoPiNiNj;
  double x,x2,y,fe,ge,fw,gw,b6,b12;

  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
			       sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  linkflag[j][i] = linkflag[i][j];
  if (linkflag[i][j] != 0)
    linkedtype[i] = linkedtype[j] = 1;

  rc3 = pow(cut[i][j],3.0);
  rc6 = rc3*rc3;
  sig2 = sigma[i][j]*sigma[i][j];
  sig6 = sig2*sig2*sig2;
  sig12 = sig6*sig6;
  phi0 = (1.0 - linkflag[i][j])*(1.0 + 0.5*linkflag[i][j]);

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;
    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);
    TwoPiNiNj = 2.0*MY_PI*all[0]*all[1];
  }

  phi = phi0 + linkflag[i][j]*lambda;
  efactor = 4.0 * epsilon[i][j] * pow(phi,exponent_n);
  lj3[i][j] = lj3[j][i] = efactor * sig12;
  lj4[i][j] = lj4[j][i] = efactor * sig6;
  lj1[i][j] = lj1[j][i] = 12.0 * lj3[i][j];
  lj2[i][j] = lj2[j][i] =  6.0 * lj4[i][j];
  asq[i][j] = asq[j][i] = alpha*sig6*pow(1.0-phi,exponent_p);
  if (offset_flag) {
    scinv = 1.0/(rc6 + asq[i][j]);
    offset[i][j] = offset[j][i] = scinv*(lj3[i][j]*scinv - lj4[i][j]);
  } else offset[i][j] = offset[j][i] = 0.0;
  if (tail_flag) {
    if (asq[i][j] == 0.0)
      fe = ge = fw = gw = 1.0;
    else {
      x = sqrt(asq[i][j])/rc3;
      x2 = x*x;
      y = 1.0/(1.0 + x2);
      fe = atanx_x( x );
      ge = 1.5*(fe - y)/x2;
      fw = 0.5*(fe + y);
      gw = 0.75*(fw - y*y)/x2;
    }
    b6 = efactor*sig6/(3.0*rc3);
    b12 = b6*sig6/(3.0*rc6);
    etail_ij = TwoPiNiNj*(b12*ge - b6*fe);
    ptail_ij = TwoPiNiNj*(4.0*b12*gw - 2.0*b6*fw);
  }

  if (linkflag[i][j] != 0)
    for (k = 0; k < gridsize; k++) {
      phi = phi0 + linkflag[i][j]*lambdanode[k];
      efactor = 4.0 * epsilon[i][j] * pow(phi,exponent_n);
      lj3n[i][j][k] = lj3n[j][i][k] = efactor * sig12;
      lj4n[i][j][k] = lj4n[j][i][k] = efactor * sig6;
      asqn[i][j][k] = asqn[j][i][k] = alpha*sig6*pow(1.0-phi,exponent_p);
      if (offset_flag) {
        scinv = 1.0/(rc6 + asqn[i][j][k]);
        offn[i][j][k] = offn[j][i][k] = scinv*(lj3n[i][j][k]*scinv - lj4n[i][j][k]);
      } else offn[i][j][k] = offn[j][i][k] = 0.0;
      if (tail_flag) {
        if (asqn[i][j][k] == 0.0)
          fe = ge = 1.0;
        else {
          x = sqrt(asqn[i][j][k])/rc3;
          x2 = x*x;
          y = 1.0/(1.0 + x2);
          fe = atanx_x( x );
          ge = 1.5*(fe - y)/x2;
        }
        b6 = efactor*sig6/(3.0*rc3);
        b12 = b6*sig6/(3.0*rc6);
        if (i == j)
          etailnode[k] += TwoPiNiNj*(b12*ge - b6*fe);
        else
          etailnode[k] += 2.0*TwoPiNiNj*(b12*ge - b6*fe);
      }
    }

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file 
------------------------------------------------------------------------- */

void PairLJCutSoftcoreOld::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j,k;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
	fwrite(&epsilon[i][j],sizeof(double),1,fp);
	fwrite(&sigma[i][j],sizeof(double),1,fp);
	fwrite(&cut[i][j],sizeof(double),1,fp);
      }
      fwrite(&linkflag[i][j],sizeof(int),1,fp);
    }
  fwrite(&gridsize,sizeof(int),1,fp);
  for (k = 0; k < gridsize; k++) {
    fwrite(&lambdanode[k],sizeof(double),1,fp);
//    fwrite(&weight[k],sizeof(double),1,fp);
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutSoftcoreOld::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j,k;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
	if (me == 0) {
	  fread(&epsilon[i][j],sizeof(double),1,fp);
	  fread(&sigma[i][j],sizeof(double),1,fp);
	  fread(&cut[i][j],sizeof(double),1,fp);
	}
	MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
      if (me == 0) fread(&linkflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&linkflag[i][j],1,MPI_INT,0,world);
    }
  if (me == 0) fread(&gridsize,sizeof(int),1,fp);
  MPI_Bcast(&gridsize,1,MPI_INT,0,world);
  memory->grow(lambdanode,gridsize,"pair:lambdanode");
  for (k = 0; k < gridsize; k++) {
    if (me == 0) {
      fread(&lambdanode[k],sizeof(double),1,fp);
//      fread(&weight[k],sizeof(double),1,fp);
    }
    MPI_Bcast(&lambdanode[k],1,MPI_DOUBLE,0,world);
//    MPI_Bcast(&weight[k],1,MPI_DOUBLE,0,world);
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutSoftcoreOld::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&alpha,sizeof(double),1,fp);
  fwrite(&exponent_n,sizeof(double),1,fp);
  fwrite(&exponent_p,sizeof(double),1,fp);
  fwrite(&lambda,sizeof(double),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutSoftcoreOld::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&alpha,sizeof(double),1,fp);
    fread(&exponent_n,sizeof(double),1,fp);
    fread(&exponent_p,sizeof(double),1,fp);
    fread(&lambda,sizeof(double),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&alpha,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&exponent_n,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&exponent_p,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&lambda,1,MPI_DOUBLE,0,world);
}

/* ---------------------------------------------------------------------- */

double PairLJCutSoftcoreOld::single(int i, int j, int itype, int jtype, double rsq,
			 double factor_coul, double factor_lj, double &fforce)
{
 
  double r6,sinv,forcelj,philj;

  r6 = rsq*rsq*rsq;
  sinv = 1.0/(r6 + asq[itype][jtype]);
  forcelj = r6*sinv*sinv*(lj1[itype][jtype]*sinv - lj2[itype][jtype]);
  fforce = factor_lj*forcelj/rsq;

  philj = sinv*(lj3[itype][jtype]*sinv-lj4[itype][jtype]) -
    offset[itype][jtype];
  return factor_lj*philj;
}

/* ---------------------------------------------------------------------- */

void *PairLJCutSoftcoreOld::extract(const char *str, int &dim)
{
  if (strcmp(str,"gridsize") == 0) {
    dim = gridsize;
  }
  else if (strcmp(str,"tail_flag") == 0) {
    dim = tail_flag;
  }
//  else if (strcmp(str,"weight") == 0) {
//    dim = gridsize;
//    return (void *) &weight[0];
//  }
  else if (strcmp(str,"lambdanode") == 0) {
    dim = gridsize;
    return (void *) &lambdanode[0];
  }
  else if (strcmp(str,"etailnode") == 0) {
    dim = gridsize;
    return (void *) &etailnode[0];
  }
  else if (strcmp(str,"energy_grid") == 0) {
    dim = gridsize;
    return (void *) &evdwlnode[0];
  }
  else if (strcmp(str,"lj/cut/softcore") == 0) {
    dim = gridsize;
    return (void *) this;
  }
  return NULL;
}

/* ---------------------------------------------------------------------- */

void PairLJCutSoftcoreOld::modify_params(int narg, char **arg)
{
  if (narg == 0) error->all(FLERR,"Illegal pair_modify command");

  int nkwds = 11;
  char **keyword = new char*[nkwds];
  keyword[0] = "alpha";
  keyword[1] = "n";
  keyword[2] = "p";
  keyword[3] = "lambda";
  keyword[4] = "link";
  keyword[5] = "rev_link";
  keyword[6] = "unlink";
  keyword[7] = "set_grid";
  keyword[8] = "add_node_deprecated";
  keyword[9] = "set_weights";
  keyword[10] = "add_weight";

  int m;
  int iarg = 0;
  while (iarg < narg) {

    // Search for a keyword:
    for (m = 0; m < nkwds; m++)
      if (strcmp(arg[iarg],keyword[m]) == 0)
        break;

    if (m == nkwds) { // no keyword found: call Pair:modify_params and return:

      Pair::modify_params(narg, arg);
      return;

    }
    else if (m < 4) { // alpha, n, p, or lambda:

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
    else if ( m < 7 ) { // link, rev_link, and unlink

      int ilo,ihi,jlo,jhi,flag,count,i,j;
      if (iarg+3 > narg)
        error->all(FLERR,"Illegal pair_modify command");
      if (!allocated) allocate();
      force->bounds(FLERR,arg[iarg+1],atom->ntypes,ilo,ihi);
      force->bounds(FLERR,arg[iarg+2],atom->ntypes,jlo,jhi);
      if (m == 4) flag = +1;
      else if (m == 5) flag = -1;
      else flag =  0;
      for (i = ilo; i <= ihi; i++)
        for (j = MAX(jlo,i); j <= jhi; j++) {
          linkflag[i][j] = linkflag[j][i] = flag;
          count++;
        }
      if (count == 0)
        error->all(FLERR,"Incorrect args for pair coefficients");
      iarg += 3;
     
    }
    else if (m == 7) { // set_grid:

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
    else if (m == 8) { // add_node:

      if (iarg+2 > narg) error->all(FLERR,"Illegal pair_modify command");
      add_node_to_grid(force->numeric(FLERR,arg[iarg+1]));
      iarg += 2;
    
    }
    else if (m == 9) { // set_weights:

      if (gridsize == 0)
        error->all(FLERR,"Softcore lambda grid has not been defined");
      if (iarg+1+gridsize > narg)
        error->all(FLERR,"Illegal pair_modify command");
//      for (int i = 0; i < gridsize; i++)
//        weight[i] = force->numeric(FLERR,arg[iarg+1+i]);
      iarg += 1+gridsize;

    }
    else if (m == 10) { // add_weight:

      if (iarg+3 > narg)
        error->all(FLERR,"Illegal pair_modify command");
      int i = force->numeric(FLERR,arg[iarg+1]);
      if ( (i < 1) || (i > gridsize))
        error->all(FLERR,"Node index out of bounds");
//      weight[i-1] = force->numeric(FLERR,arg[iarg+2]);
      iarg += 3;
    
    }
    else // unknown keyword:
      error->all(FLERR,"Illegal pair_modify command: unknown keyord");
  }
}

/* ---------------------------------------------------------------------- */
