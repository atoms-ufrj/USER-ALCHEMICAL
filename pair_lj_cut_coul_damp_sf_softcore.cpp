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
     1) Include electrostatic part in RESPA routines
     1) Test write_restart and read_restart routines
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_lj_cut_coul_damp_sf_softcore.h"
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

#define sixthroot(X)  cbrt(sqrt(X))

/* ---------------------------------------------------------------------- */

PairLJCutCoulDampSFSoftcore::PairLJCutCoulDampSFSoftcore(LAMMPS *lmp) : PairAlchemical(lmp)
{
  respa_enable = 1;
  writedata = 1;
  self_flag = 0;
}

/* ---------------------------------------------------------------------- */

PairLJCutCoulDampSFSoftcore::~PairLJCutCoulDampSFSoftcore()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut_ljsq);

    memory->destroy(cut_lj);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);

    memory->destroy(asq);
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutCoulDampSFSoftcore::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype,intra;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,vs,fs,evdwl,ecoul,fpair;
  double s,rsq,r2inv,r4,r6,s6,s6inv,forcelj,prefactor,forcecoul,factor_lj,factor_coul;
  int *ilist,*jlist,*numneigh,**firstneigh;

  if (gridflag) for (i = 0; i < gridsize; i++) evdwlnode[i] = ecoulnode[i] = 0.0;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // Compute self energy:
  if (eflag && self_flag)
    for (i = 0; i < nlocal; i++)
      ev_tally(i,i,nlocal,0,0.0,e_self*q[i]*q[i],0.0,0.0,0.0,0.0);

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    qtmp = qqrd2e*q[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      intra = sbmask(j);
      factor_lj = special_lj[intra];
      factor_coul = special_coul[intra];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r4 = rsq*rsq;
        r6 = rsq*r4;
        s6 = r6 + asq[itype][jtype];
        s6inv = 1.0/s6;

        if (rsq < cut_ljsq[itype][jtype])
          forcelj = s6inv*(lj1[itype][jtype]*s6inv - lj2[itype][jtype]);
        else
          forcelj = 0.0;

        if (rsq < cut_coulsq) {
          prefactor = factor_coul*qtmp*q[j];
          if (intra)
            forcecoul = prefactor*sixthroot(s6inv);
          else {
            s = sixthroot(s6);
            unshifted( s, vs, fs );
            forcecoul = prefactor*(fs - f_shift)*s;
          }
        }
        else
          forcecoul = 0.0;

        fpair = efactor*factor_lj*(forcelj + forcecoul)*r4*s6inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          if (rsq < cut_ljsq[itype][jtype])
            evdwl = efactor*factor_lj*(s6inv*(lj3[itype][jtype]*s6inv-lj4[itype][jtype]) -
                    offset[itype][jtype]);
          else
            evdwl = 0.0;

          if (rsq < cut_coulsq)
            ecoul = intra ? forcecoul : efactor*prefactor*(vs + s*f_shift - e_shift);
          else
            ecoul = 0.0;
        }

        if (evflag)
          ev_tally(i,j,nlocal,newton_pair,evdwl,ecoul,fpair,delx,dely,delz);

        if (gridflag)
          for (int k = 0; k < gridsize; k++) {
            s6 = r6 + asqn[itype][jtype][k];
            s6inv = 1.0/s6;

            evdwl = s6inv*(lj3[itype][jtype]*s6inv - lj4[itype][jtype]) -
                    offset[itype][jtype];
            evdwl *= efactorn[k]*factor_lj;

            if (intra)
              ecoul = sixthroot(s6inv);
            else {
              s = sixthroot(s6);
              unshifted( s, vs, fs );
              ecoul = vs + s*f_shift - e_shift;
            }
            ecoul *= efactorn[k]*prefactor;

            if (newton_pair || j < nlocal) {
              evdwlnode[k] += evdwl;
              ecoulnode[k] += ecoul;
            }
            else {
              evdwlnode[k] += 0.5*evdwl;
              ecoulnode[k] += 0.5*ecoul;
            }

          }
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();

  uptodate = gridflag;
  gridflag = 0;
}

/* ---------------------------------------------------------------------- */

void PairLJCutCoulDampSFSoftcore::compute_inner()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double rsq,r6,s6inv,forcelj,factor_lj,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
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
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_out_off_sq) {
        r6 = rsq*rsq*rsq;
        s6inv = 1.0/(r6 + asq[itype][jtype]);
        jtype = type[j];
        forcelj = r6*s6inv*s6inv*(lj1[itype][jtype]*s6inv - lj2[itype][jtype]);
        fpair = factor_lj*forcelj/rsq;
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

void PairLJCutCoulDampSFSoftcore::compute_middle()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double rsq,r6,s6inv,forcelj,factor_lj,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
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
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_out_off_sq && rsq > cut_in_off_sq) {
        r6 = rsq*rsq*rsq;
        s6inv = 1.0/(r6 + asq[itype][jtype]);
        jtype = type[j];
        forcelj = r6*s6inv*s6inv*(lj1[itype][jtype]*s6inv - lj2[itype][jtype]);
        fpair = factor_lj*forcelj/rsq;
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

void PairLJCutCoulDampSFSoftcore::compute_outer(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r6,s6inv,forcelj,factor_lj,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  if (gridflag) for (i = 0; i < gridsize; i++) evdwlnode[i] = 0.0;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
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
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        if (rsq > cut_in_off_sq) {
          r6 = rsq*rsq*rsq;
          s6inv = 1.0/(r6 + asq[itype][jtype]);
          forcelj = r6*s6inv*s6inv*(lj1[itype][jtype]*s6inv - lj2[itype][jtype]);
          fpair = factor_lj*forcelj/rsq;
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
          r6 = rsq*rsq*rsq;
          s6inv = 1.0/(r6 + asq[itype][jtype]);
          evdwl = s6inv*(lj3[itype][jtype]*s6inv-lj4[itype][jtype]) -
            offset[itype][jtype];
          evdwl *= factor_lj;
        }

        if (vflag) {
          if (rsq <= cut_in_off_sq) {
            r6 = rsq*rsq*rsq;
            s6inv = 1.0/(r6 + asq[itype][jtype]);
            forcelj = r6*s6inv*s6inv*(lj1[itype][jtype]*s6inv - lj2[itype][jtype]);
            fpair = factor_lj*forcelj/rsq;
          } else if (rsq < cut_in_on_sq)
            fpair = factor_lj*forcelj/rsq;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);

        if (gridflag)
          for (int k = 0; k < gridsize; k++) {
            s6inv = 1.0/(r6 + asqn[itype][jtype][k]);
            evdwl = s6inv*(lj3[itype][jtype]*s6inv - lj4[itype][jtype]) -
                    offset[itype][jtype];
            evdwl *= efactorn[k]*factor_lj;
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

void PairLJCutCoulDampSFSoftcore::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(cut_ljsq,n+1,n+1,"pair:cut_ljsq");

  memory->create(cut_lj,n+1,n+1,"pair:cut");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");

  memory->create(asq,n+1,n+1,"pair:asq");
  memory->create(efactorn,gridsize,"pair:efactorn");
  memory->create(asqn,n+1,n+1,gridsize,"pair:asqn");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJCutCoulDampSFSoftcore::settings(int narg, char **arg)
{
  if (narg < 3 || narg > 4) error->all(FLERR,"Illegal pair_style command");

  alphaC = force->numeric(FLERR,arg[0]);
  sigmaC = force->numeric(FLERR,arg[1]);
  cut_lj_global = force->numeric(FLERR,arg[2]);

  if (narg == 3)
    cut_coul = cut_lj_global;
  else
    cut_coul = force->numeric(FLERR,arg[3]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j])
          cut_lj[i][j] = cut_lj_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJCutCoulDampSFSoftcore::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 5)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);

  double cut_one = cut_lj_global;
  if (narg == 5) cut_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut_lj[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCutCoulDampSFSoftcore::init_style()
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

  if (!atom->q_flag)
    error->all(FLERR,"Pair style lj/cut/coul/damp/sf/softcore requires atom charges");

  cut_coulsq = cut_coul * cut_coul;
  unshifted( cut_coul, e_shift, f_shift );
  e_shift += f_shift*cut_coul;
  e_self = -(e_shift/2.0 + alphaC/sqrt(MY_PI))*force->qqrd2e;

  int n = atom->ntypes;
  memory->grow(efactorn,gridsize,"pair:efactorn");
  memory->grow(asqn,n+1,n+1,gridsize,"pair:asqn");

  double save = lambda;
  for (int k = 0; k < gridsize; k++) {
    lambda = lambdanode[k];
    efactorn[k] = efactor = pow(lambda, exponent_n);
    etailnode[k] = 0.0;
    for (int i = 1; i <= n; i++)
      for (int j = i; j <= n; j++)
        if (setflag[i][j] || (setflag[i][i] && setflag[j][j])) {
          init_one(i,j);
          asqn[i][j][k] = asqn[j][i][k] = asq[i][j];
          if (tail_flag) etailnode[k] += (i == j ? 1.0 : 2.0)*etail_ij;
        }
  }
  lambda = save;

  PairAlchemical::init_style();
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   regular or rRESPA
------------------------------------------------------------------------- */

void PairLJCutCoulDampSFSoftcore::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
  else if (id == 1) listinner = ptr;
  else if (id == 2) listmiddle = ptr;
  else if (id == 3) listouter = ptr;
}

/* ----------------------------------------------------------------------
   Auxiliary function for tail correction calculations
------------------------------------------------------------------------- */

double PairLJCutCoulDampSFSoftcore::atanx_x(double x)
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

double PairLJCutCoulDampSFSoftcore::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut_lj[i][j] = mix_distance(cut_lj[i][i],cut_lj[j][j]);
  }

  double rc = cut_lj[i][j];
  double rc3 = rc*rc*rc;
  double rc6 = rc3*rc3;
  double sig2 = sigma[i][j]*sigma[i][j];
  double sig6 = sig2*sig2*sig2;
  double sig12 = sig6*sig6;
  double eps4 = 4.0 * epsilon[i][j];

  cut_ljsq[i][j] = cut_ljsq[j][i] = rc*rc;
  lj3[i][j] = lj3[j][i] = eps4 * sig12;
  lj4[i][j] = lj4[j][i] = eps4 * sig6;
  lj1[i][j] = lj1[j][i] = 12.0 * lj3[i][j];
  lj2[i][j] = lj2[j][i] =  6.0 * lj4[i][j];
  asq[i][j] = asq[j][i] = alpha*sig6*pow(1.0 - lambda,exponent_p);

  if (offset_flag && (rc > 0.0)) {
    double rc6inv = 1.0/(rc6 + asq[i][j]);
    offset[i][j] = offset[j][i] = rc6inv*(lj3[i][j]*rc6inv - lj4[i][j]);
  } else offset[i][j] = 0.0;

  // check interior rRESPA cutoff

  if (cut_respa && rc < cut_respa[3])
    error->all(FLERR,"Pair cutoff < Respa interior cutoff");

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

    double TwoPiNiNj = 2.0*MY_PI*all[0]*all[1];
    double fe, ge, fw, gw;
    if (asq[i][j] == 0.0)
      fe = ge = fw = gw = 1.0;
    else {
      double x = sqrt(asq[i][j])/rc3;
      double x2 = x*x;
      double y = 1.0/(1.0 + x2);
      fe = atanx_x( x );
      ge = 1.5*(fe - y)/x2;
      fw = 0.5*(fe + y);
      gw = 0.75*(fw - y*y)/x2;
    }
    double b6 = efactor*eps4*sig6/(3.0*rc3);
    double b12 = b6*sig6/(3.0*rc6);
    etail_ij = TwoPiNiNj*(b12*ge - b6*fe);
    ptail_ij = TwoPiNiNj*(4.0*b12*gw - 2.0*b6*fw);
  }

  return MAX(rc, cut_coul);
}

/* ---------------------------------------------------------------------- */

void PairLJCutCoulDampSFSoftcore::modify_params(int narg, char **arg)
{
  if (narg == 0)
    error->all(FLERR,"Illegal pair_modify command");

  int iarg, ns, skip[narg];
  iarg = ns = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"self") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal pair_modify command");
      if (strcmp(arg[iarg+1],"yes") == 0)
        self_flag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0)
        self_flag = 0;
      else
        error->all(FLERR,"Illegal pair_modify command");
      single_enable = !self_flag;
      iarg += 2;
    }
    else // no keyword found - skip argument:
      skip[ns++] = iarg++;
  }

  // Call parent-class routine with skipped arguments:
  if (ns > 0) {
    for (int i = 0; i < ns; i++)
      arg[i] = arg[skip[i]];
    PairAlchemical::modify_params(ns, arg);
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutCoulDampSFSoftcore::write_restart(FILE *fp)
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
        fwrite(&cut_lj[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutCoulDampSFSoftcore::read_restart(FILE *fp)
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
          fread(&cut_lj[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_lj[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutCoulDampSFSoftcore::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutCoulDampSFSoftcore::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&cut_lj_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&tail_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_lj_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLJCutCoulDampSFSoftcore::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g\n",i,epsilon[i][i],sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLJCutCoulDampSFSoftcore::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g\n",i,j,epsilon[i][j],sigma[i][j],cut_lj[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairLJCutCoulDampSFSoftcore::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
  double r4,s6,s6inv,forcelj,philj;
  double s,vs,fs,prefactor,forcecoul,phicoul;

  r4 = rsq*rsq;
  s6 = rsq*r4 + asq[itype][jtype];
  s6inv = 1.0/s6;
  forcelj = s6inv*(lj1[itype][jtype]*s6inv - lj2[itype][jtype]);

  prefactor = force->qqrd2e * atom->q[i] * atom->q[j];
  s = sixthroot(s6);
  unshifted( s, vs, fs );
  forcecoul = prefactor*(fs - f_shift)*s;

  fforce = efactor*(factor_lj*forcelj + factor_coul*forcecoul)*r4*s6inv;

  philj = s6inv*(lj3[itype][jtype]*s6inv - lj4[itype][jtype]) - offset[itype][jtype];
  phicoul = prefactor*(vs + s*f_shift - e_shift);

  return efactor*(factor_lj*philj + factor_coul*phicoul);
}

/* ---------------------------------------------------------------------- */

void *PairLJCutCoulDampSFSoftcore::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  return NULL;
}

/* ---------------------------------------------------------------------- */

// NOTE: This derivative considers both exponent_n = 1 and exponent_p = 1

double PairLJCutCoulDampSFSoftcore::derivative()
{
  int i,j,k,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r6,s6inv,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  double C = alpha*lambda;
  double dEdl = 0.0;

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
        s6inv = 1.0/(r6 + asq[itype][jtype]);
        evdwl = factor_lj*s6inv*(lj3[itype][jtype]*s6inv - lj4[itype][jtype]);
        if (newton_pair)
          dEdl += evdwl*(1.0 + C*s6inv);
        else
          dEdl += 0.5*evdwl*(1.0 + C*s6inv);
      }
    }
  }

  return dEdl;
}

/* ---------------------------------------------------------------------- */

