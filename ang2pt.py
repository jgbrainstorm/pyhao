# ------------------------------------------------------------------------
# This code calculate the two point angular correlation. 
# using the Landy - Szalay estimator. It has been tested by running
# on the SDSS data and compared with the results from Connolly et al 2002
# and get consistent results.
# Created by: J. Hao, 2011 @ Fermilab
# report bugs to: J. Hao, jghao@fnal.gov
#-------------------------------------------------------------------------- 

import numpy as np
import pyfits as pf
import pylab as pl
import esutil as es
from binplot import *
Da = es.cosmology.Cosmo(h=0.7).Da
Dc = es.cosmology.Cosmo(h=0.7).Dc # comoving distance


def subsetIDX(x,xmin,xmax):
    ind = np.arange(len(x))
    ok = (x >= xmin)*(x <= xmax)
    return ind[ok]


def getAng2Pt(galRA=None,galDEC=None,randRA=None,randDEC=None,galZ=None,randZ=None,srad=1.,zmin=None,zmax=None,zdiff=None,nbins=10,angle=True,binrange=None):
    """
    This function calculate the two point correlation (angular) using the Landy-
    Szalay estimator.
    input: 
        the ra, dec, redshift of galaxy/cluster: galRA, galDEC,galZ
        the ra, dec, redshift of random points: randRA, randDEC, randZ
        the search radius in degree: srad
        the range of redshift to be considered: zmin, zmax
        the maximum difference of the redshift of each data pair: zdiff
        the number of angular bins (in log scale): nbins
        the range of the angular bins: binrange
        if angle = True, it returns results in degree
        if angle = False, it returns results in Mpc
    output:
        angular separation, correlation, error of correlation
    """
    depth=10
    h=es.htm.HTM(depth)
    gok = subsetIDX(galZ,zmin,zmax)
    rok = subsetIDX(randZ,zmin,zmax)
    galRA = galRA[gok]
    galDEC = galDEC[gok]
    galZ = galZ[gok]
    randRA = randRA[rok]
    randDEC = randDEC[rok]
    randZ = randZ[rok]
    #-----gal vs. gal ---
    gin1,gin2,gd12 = h.match(galRA,galDEC,galRA,galDEC,srad,maxmatch=5000)
    ok = np.abs(galZ[gin1]-galZ[gin2])<= zdiff
    gin1 = gin1[ok]
    gin2 = gin2[ok]
    gd12 = gd12[ok] # in angle, deg
    gdr12=np.deg2rad(gd12)*Da(0,(galZ[gin1]+galZ[gin2])/2.)
    #-----rand vs. rand ---
    rin1,rin2,rd12 = h.match(randRA,randDEC,randRA,randDEC,srad,maxmatch=5000)
    ok = np.abs(randZ[rin1]-randZ[rin2])<= zdiff
    rin1 = rin1[ok]
    rin2 = rin2[ok]
    rd12 = rd12[ok] # in angle, deg
    rdr12=np.deg2rad(rd12)*Da(0,(randZ[rin1]+randZ[rin2])/2.)
    #----- gal vs. rand ---
    grin1,grin2,grd12 = h.match(galRA,galDEC,randRA,randDEC,srad,maxmatch=5000)
    ok = np.abs(galZ[grin1]-randZ[grin2])<= zdiff
    grin1 = grin1[ok]
    grin2 = grin2[ok]
    grd12 = grd12[ok] # in angle, deg
    grdr12=np.deg2rad(grd12)*Da(0,(galZ[grin1]+randZ[grin2])/2.)
    # --in terms of physical separation ------
    if angle == False:
        binedge = logbin_edge(nbins=nbins,xrange=binrange)
        gg,edge_gg = np.histogram(gdr12,bins=binedge)
        rr,edge_rr = np.histogram(rdr12,bins=binedge)
        gr,edge_gr = np.histogram(grdr12,bins=binedge)
    else:
        binedge = logbin_edge(nbins=nbins,xrange=binrange)
        gg,edge_gg = np.histogram(gd12,bins=binedge)
        rr,edge_rr = np.histogram(rd12,bins=binedge)
        gr,edge_gr = np.histogram(grd12,bins=binedge)
    gg = gg/2.
    rr = rr/2.
    gr = gr/2.
    Ng = float(len(galRA))
    Nr = float(len(randRA))
    w = (gg*Nr/Ng - 2.*gr*Nr/Ng + rr)/rr
    #w = (gg - 2.*gr + rr)/rr
    wErr = np.sqrt((1+w)/gg) 
    sep = 0.5*(binedge[0:-1]+binedge[1:])
    return sep,w, wErr

if __name__ == "__main__":   
   #--example of using the function---------------
   #------specify the galaxy/cluster catalog and random catalog

   from ang2pt import *
   galfile = '/home/jghao/research/SDSS_DR7_GMBCG/public_catalog/gmbcg_cluster_for_CMB_healpixID.fit'
   randfile = '/home/jghao/research/SDSS_DR7_GMBCG/public_catalog/gmbcg_for_CMB_rand.fit'

   zmin = 0.1
   zmax = 0.25
   zdiff = 0.03
   gal = pf.getdata(galfile)
   rand = pf.getdata(randfile)
   galRA = gal.field('ra')
   galDEC = gal.field('dec')
   randRA = rand.field('ra')
   randDEC = rand.field('dec')
   galZ = gal.field('photoz')
   randZ = rand.field('photoz')

   # --- calculate correlation in degree -----
   sep,w,wErr = getAng2Pt(galRA=galRA,galDEC=galDEC,randRA=randRA,randDEC=randDEC,galZ=galZ,randZ=randZ,srad=15.,zmin=zmin,zmax=zmax,zdiff=zdiff,nbins=10,angle=True,binrange=[0.05,10])

   #----making plot ------

   pl.errorbar(sep,w, yerr=wErr,fmt='bo')
   pl.loglog()
   pl.ylim(0.0001,100)
   pl.xlim(0.01,100)
   pl.xlabel(r'$\theta$' +' [deg]',fontsize=18)
   pl.ylabel(r'$\xi(\theta)$',fontsize=18)
  
   # --- calculate correlation in Mpc ---
   sep,w,wErr = getAng2Pt(galRA=galRA,galDEC=galDEC,randRA=randRA,randDEC=randDEC,galZ=galZ,randZ=randZ,srad=15.,zmin=zmin,zmax=zmax,zdiff=zdiff,nbins=15,angle=False,binrange=[1,50])
   # --- making plot ----
   pl.errorbar(sep,w, yerr=wErr,fmt='bo')
   pl.loglog()
   pl.ylim(0.0001,100)
   pl.xlim(1,100)
   pl.xlabel(r'$r$' +' [Mpc]',fontsize=18)
   pl.ylabel(r'$\xi(r)$',fontsize=18)

