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

   Based on original Lennard-Jones potential code by Paul Crozier (SNL)
------------------------------------------------------------------------- */

/* TODO:
     1) Test write_restart and read_restart routines
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_mie_cut_softcore_london.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairMieCutSoftcoreLondon::PairMieCutSoftcoreLondon(LAMMPS *lmp) : PairAlchemical(lmp)
{
  respa_enable = 1;
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairMieCutSoftcoreLondon::~PairMieCutSoftcoreLondon()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(gamR);
    memory->destroy(Cmie);
    memory->destroy(mie1);
    memory->destroy(mie2);
    memory->destroy(mie3);
    memory->destroy(offset);

    memory->destroy(asq);
  }
}

/* ---------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::compute(int eflag, int vflag)
{
  int i,j,k,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,ratio,sinvc,rgamA,forcemie,factor_mie,sinvcRA;
  int *ilist,*jlist,*numneigh,**firstneigh;

  if (gridflag) for (i = 0; i < gridsize; i++) evdwlnode[i] = 0.0;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_mie = force->special_lj;
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
      factor_mie = special_mie[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {

        rgamA = rsq*rsq*rsq;
        ratio = rgamA / mie1[itype][jtype];
        sinvc = 1.0 / (ratio + asq[itype][jtype]);
        sinvcRA = pow(sinvc,mie3[itype][jtype]);
        forcemie = mie2[itype][jtype] * sinvc * ratio *
            (gamR[itype][jtype]*sinvcRA - 6.0*sinvc);
        fpair = factor_mie*forcemie/rsq;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
           evdwl = factor_mie*(mie2[itype][jtype]*
            (sinvcRA-sinvc)-offset[itype][jtype]);
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);

        if (gridflag)
          for (int k = 0; k < gridsize; k++) {
           ratio = rgamA / mie1n[itype][jtype][k];
           sinvc = 1.0 / (ratio + asqn[itype][jtype][k]);
           evdwl = factor_mie*(mie2n[itype][jtype][k]*
                (pow(sinvc,mie3n[itype][jtype][k])-sinvc)-offsetn[itype][jtype][k]);
            if (newton_pair || j < nlocal)
              evdwlnode[k] += evdwl;
            else
              evdwlnode[k] += 0.5*evdwl;
          }
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();

  uptodate = gridflag;
  gridflag = 0;
}

/* ---------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::compute_inner()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double rsq,ratio,sinvc,rgamA,forcemie,factor_mie,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_mie = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = listinner->inum;
  ilist = listinner->ilist;
  numneigh = listinner->numneigh;
  firstneigh = listinner->firstneigh;

  double cut_out_on = cut_respa[0];
  double cut_out_off = cut_respa[1];

  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_out_on_sq = cut_out_on*cut_out_on;
  double cut_out_off_sq = cut_out_off*cut_out_off;

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
      factor_mie = special_mie[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_out_off_sq) {
        rgamA = rsq*rsq*rsq;
        ratio = rgamA / mie1[itype][jtype]; 
        sinvc = 1.0 / (ratio + asq[itype][jtype]);
        forcemie = mie2[itype][jtype] * sinvc * ratio *
            (gamR[itype][jtype]*pow(sinvc, mie3[itype][jtype]) -
        6.0*sinvc);
        fpair = factor_mie*forcemie/rsq;
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
          fpair *= 1.0 - rsw*rsw*(3.0 - 2.0*rsw);
        }

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::compute_middle()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double rsq,ratio,sinvc,rgamA,forcemie,factor_mie,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_mie = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = listmiddle->inum;
  ilist = listmiddle->ilist;
  numneigh = listmiddle->numneigh;
  firstneigh = listmiddle->firstneigh;

  double cut_in_off = cut_respa[0];
  double cut_in_on = cut_respa[1];
  double cut_out_on = cut_respa[2];
  double cut_out_off = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_in_off_sq = cut_in_off*cut_in_off;
  double cut_in_on_sq = cut_in_on*cut_in_on;
  double cut_out_on_sq = cut_out_on*cut_out_on;
  double cut_out_off_sq = cut_out_off*cut_out_off;

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
      factor_mie = special_mie[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_out_off_sq && rsq > cut_in_off_sq) {
        rgamA = rsq*rsq*rsq;
        ratio = rgamA / mie1[itype][jtype]; 
        sinvc = 1.0 / (ratio + asq[itype][jtype]);
        forcemie = mie2[itype][jtype] * sinvc * ratio *
            (gamR[itype][jtype]*pow(sinvc,mie3[itype][jtype]) -
        6.0*sinvc);
        fpair = factor_mie*forcemie/rsq;
        if (rsq < cut_in_on_sq) {
          rsw = (sqrt(rsq) - cut_in_off)/cut_in_diff;
          fpair *= rsw*rsw*(3.0 - 2.0*rsw);
        }
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
          fpair *= 1.0 + rsw*rsw*(2.0*rsw - 3.0);
        }

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::compute_outer(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,ratio,sinvc,rgamA,forcemie,factor_mie,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  if (gridflag) for (i = 0; i < gridsize; i++) evdwlnode[i] = 0.0;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_mie = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = listouter->inum;
  ilist = listouter->ilist;
  numneigh = listouter->numneigh;
  firstneigh = listouter->firstneigh;

  double cut_in_off = cut_respa[2];
  double cut_in_on = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_in_off_sq = cut_in_off*cut_in_off;
  double cut_in_on_sq = cut_in_on*cut_in_on;

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
      factor_mie = special_mie[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        if (rsq > cut_in_off_sq) {
          rgamA = rsq*rsq*rsq;
          ratio = rgamA / mie1[itype][jtype]; 
          sinvc = 1.0 / (ratio + asq[itype][jtype]);
          forcemie = mie2[itype][jtype] * sinvc * ratio *
          (gamR[itype][jtype]*pow(sinvc,mie3[itype][jtype]) -
          6.0*sinvc);
          fpair = factor_mie*forcemie/rsq;
          if (rsq < cut_in_on_sq) {
            rsw = (sqrt(rsq) - cut_in_off)/cut_in_diff;
            fpair *= rsw*rsw*(3.0 - 2.0*rsw);
          }

          f[i][0] += delx*fpair;
          f[i][1] += dely*fpair;
          f[i][2] += delz*fpair;
          if (newton_pair || j < nlocal) {
            f[j][0] -= delx*fpair;
            f[j][1] -= dely*fpair;
            f[j][2] -= delz*fpair;
          }
        }

        if (eflag) {
          rgamA = rsq*rsq*rsq;
          ratio = rgamA / mie1[itype][jtype];
          sinvc = 1.0 / (ratio + asq[itype][jtype]);
          evdwl = factor_mie*(mie2[itype][jtype]*
                (pow(sinvc,mie3[itype][jtype])-sinvc)-offset[itype][jtype]);
        }

        if (vflag) {
          if (rsq <= cut_in_off_sq) {
            rgamA = rsq*rsq*rsq;
            ratio = rgamA / mie1[itype][jtype]; 
            sinvc = 1.0 / (ratio + asq[itype][jtype]);
            forcemie = mie2[itype][jtype] * sinvc * ratio *
            (gamR[itype][jtype]*pow(sinvc,mie3[itype][jtype]) -
            6.0*sinvc);
            fpair = factor_mie*forcemie/rsq;
          } else if (rsq < cut_in_on_sq)
            fpair = factor_mie*forcemie/rsq;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);

        if (gridflag)
          for (int k = 0; k < gridsize; k++) {
            rgamA = rsq*rsq*rsq;
            ratio = rgamA / mie1n[itype][jtype][k];
            sinvc = 1.0 / (ratio + asqn[itype][jtype][k]);
            evdwl = factor_mie*(mie2n[itype][jtype][k]*
              (pow(sinvc,mie3n[itype][jtype][k])-sinvc)-offsetn[itype][jtype][k]);
            if (newton_pair || j < nlocal)
              evdwlnode[k] += evdwl;
            else
              evdwlnode[k] += 0.5*evdwl;
          }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::allocate()
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
  memory->create(gamR,n+1,n+1,"pair:gamR");
  memory->create(Cmie,n+1,n+1,"pair:Cmie");
  memory->create(mie1,n+1,n+1,"pair:mie1");
  memory->create(mie2,n+1,n+1,"pair:mie2");
  memory->create(mie3,n+1,n+1,"pair:mie3");
  memory->create(offset,n+1,n+1,"pair:offset");

  memory->create(asq,n+1,n+1,"pair:asq");
  memory->create(mie1n,n+1,n+1,gridsize,"pair:mie1n");
  memory->create(mie2n,n+1,n+1,gridsize,"pair:mie2n");
  memory->create(mie3n,n+1,n+1,gridsize,"pair:mie3n");
  memory->create(asqn,n+1,n+1,gridsize,"pair:asqn");
  memory->create(offsetn,n+1,n+1,gridsize,"pair:offsetn");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::coeff(int narg, char **arg)
{
  if (narg < 5 || narg > 6)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);
  double gamR_one = force->numeric(FLERR,arg[4]);

  double cut_one = cut_global;
  if (narg == 6) cut_one = force->numeric(FLERR,arg[5]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      gamR[i][j] = gamR_one;
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

void PairMieCutSoftcoreLondon::init_style()
{
  // request regular or rRESPA neighbor lists

  int irequest;

  if (update->whichflag == 1 && strstr(update->integrate_style,"respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;

    if (respa == 0) irequest = neighbor->request(this,instance_me);
    else if (respa == 1) {
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->respaouter = 1;
    } else {
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 2;
      neighbor->requests[irequest]->respamiddle = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->respaouter = 1;
    }

  } else irequest = neighbor->request(this,instance_me);

  // set rRESPA cutoffs

  if (strstr(update->integrate_style,"respa") &&
      ((Respa *) update->integrate)->level_inner >= 0)
    cut_respa = ((Respa *) update->integrate)->cutoff;
  else cut_respa = NULL;

  PairAlchemical::init_style();

  int n = atom->ntypes;
  memory->grow(mie1n,n+1,n+1,gridsize,"pair:mie1n");
  memory->grow(mie2n,n+1,n+1,gridsize,"pair:mie2n");
  memory->grow(mie3n,n+1,n+1,gridsize,"pair:mie3n");
  memory->grow(asqn,n+1,n+1,gridsize,"pair:asqn");
  memory->grow(offsetn,n+1,n+1,gridsize,"pair:offsetn");

  double save = lambda;
  for (int k = 0; k < gridsize; k++) {
    lambda = lambdanode[k];
    etailnode[k] = 0.0;
    for (int i = 1; i <= n; i++)
      for (int j = i; j <= n; j++)
        if (setflag[i][j] || (setflag[i][i] && setflag[j][j])) {
          init_one(i,j);
          mie1n[i][j][k] = mie1n[j][i][k] = mie1[i][j];
          mie2n[i][j][k] = mie2n[j][i][k] = mie2[i][j];
          mie3n[i][j][k] = mie3n[j][i][k] = mie3[i][j];
          asqn[i][j][k] = asqn[j][i][k] = asq[i][j];
          offsetn[i][j][k] = offsetn[j][i][k] = offset[i][j];
          if (tail_flag) etailnode[k] += (i == j ? 1.0 : 2.0)*etail_ij;
        }
  }
  lambda = save;
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   regular or rRESPA
------------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
  else if (id == 1) listinner = ptr;
  else if (id == 2) listmiddle = ptr;
  else if (id == 3) listouter = ptr;
}

/* ----------------------------------------------------------------------
   Auxiliary function for tail correction calculations
------------------------------------------------------------------------- */

double PairMieCutSoftcoreLondon::atanx_x(double x)
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

double PairMieCutSoftcoreLondon::init_one(int i, int j)
{
  double Cmie,sinvc,ratio,rcA;
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    gamR[i][j] = mix_distance(gamR[i][i],gamR[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  gamR[j][i] = gamR[i][j];
  rcA = cut[i][j]*cut[i][j]*cut[i][j]*cut[i][j]*cut[i][j]*cut[i][j];
  Cmie = (gamR[i][j]/(gamR[i][j]-6.0) *
                pow((gamR[i][j]/6.0),
                    (6.0/(gamR[i][j]-6.0))));

  mie1[i][j] = mie1[j][i] = pow(sigma[i][j],6.0);
  mie2[i][j] = mie2[j][i] = Cmie*epsilon[i][j] * pow(lambda,exponent_n);
  mie3[i][j] = mie3[j][i] = gamR[i][j]/6.0;
  asq[i][j] = asq[j][i] = alpha*pow(1.0-lambda,exponent_p);

  if (offset_flag && (cut[i][j] > 0.0)) {
    ratio = rcA / mie1[i][j]; 
    sinvc = 1.0 / (ratio + asq[i][j]);
    offset[i][j] = offset[j][i] = mie2[i][j] * 
                  (pow(sinvc,mie3[i][j]) - sinvc);
  } else offset[i][j] = 0.0;

  // check interior rRESPA cutoff

  if (cut_respa && cut[i][j] < cut_respa[3])
    error->all(FLERR,"Pair cutoff < Respa interior cutoff");

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::write_restart(FILE *fp)
{
  PairAlchemical::write_restart(fp);
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&gamR[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::read_restart(FILE *fp)
{
  PairAlchemical::read_restart(fp);
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&gamR[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&gamR[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&tail_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g\n",i,epsilon[i][i],sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairMieCutSoftcoreLondon::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g\n",i,j,epsilon[i][j],sigma[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairMieCutSoftcoreLondon::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_mie,
                         double &fforce)
{
  double forcemie,phimie;
  double sinvc,sinvcRA,rgamA,ratio;

    rgamA = rsq*rsq*rsq;
    ratio = rgamA / mie1[itype][jtype]; 
    sinvc = 1.0 / (ratio + asq[itype][jtype]);
    sinvcRA = pow(sinvc,mie3[itype][jtype]);
    forcemie = mie2[itype][jtype] * sinvc * ratio *
      (gamR[itype][jtype]*sinvcRA - 6.0*sinvc);

    fforce = factor_mie*forcemie/rsq;


    phimie = mie2[itype][jtype]*(sinvcRA - sinvc) - offset[itype][jtype];

  return factor_mie*phimie;
}

/* ---------------------------------------------------------------------- */

void *PairMieCutSoftcoreLondon::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  return NULL;
}
