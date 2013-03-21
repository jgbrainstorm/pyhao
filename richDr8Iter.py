"""
This code added the iteration of the redshift based on the ridgeline. Basically, starting with the BCG's redshift and then fit the photoz by the ridgeline and the redo the analysis until it converge. 
"""
import numpy as np
import pyfits as pf
import pylab as pl
import esutil as es
import ecgmmPy as gmm
import rwecgmmPy as rwgmm
import scipy.stats as sts
import glob as gl
import healpy as hp
import os


#--------color model based on mock v3.04-----

def gmrz(gmr):
    z = 1./2.681*gmr - 0.665/2.681
    return z

def rmiz(rmi):
    z = 1./2.063*rmi - (-0.266/2.063)
    return z

def imzz(imz):
    z = 1./1.667*imz - (-0.708/1.667)
    return z

def zmyz(zmy):
    z = 1./0.192 * zmy - (0.223/0.192)
    return z

def sdssradec(ra=None,dec=None):
    print('http://skyserver.sdss3.org/dr9/en/tools/chart/navi.asp?opt=S&ra='+str(ra)+'&dec='+str(dec)+'&scale=0.800000&width=919&height=919&opt=GS&query=')

def mstar(z):
    res = 22.44+3.36*np.log(z)+0.273*np.log(z)**2 - 0.0618*np.log(z)**3 - 0.0227*np.log(z)**4
    return res

def mstar0_2(z):
    """
    see the redmapper paper P5.
    """
    res = mstar(z)+1.75
    return res

Da=es.cosmology.Cosmo(h=0.7).Da
#-----0.4L*----------
def limi(x):
    A=np.exp(3.1638)
    k=0.1428
    lmi=A*x**k
    return(lmi)

#-----0.2L*----------
def limi0_2(x):
    A=np.exp(3.1638)
    k=0.1428
    lmi=A*x**k + 0.6775790888 # this is obtained by comparing with mstar0_2
    return(lmi)


#----trim outliers --------
def trimOutlier(x):
    y=x
    n = len(x)
    y.sort()
    ind_qt1 = round((n+1)/4.)
    ind_qt3 = round((n+1)*3/4.)
    IQR = y[ind_qt3]- y[ind_qt1]
    lowFense = y[ind_qt1] - 1.5*IQR
    highFense = y[ind_qt3] + 1.5*IQR
    ok = (y>lowFense)*(y<highFense)
    return y[ok]

#----define the stack of record array----

def hstack2(arrays):
    return arrays[0].__array_wrap__(np.hstack(arrays))

#-----get the healpix id based on RA and DEC -----

def healpixIDradec(ra,dec,nside = 2**5):
    l,b = es.coords.eq2gal(ra, dec)
    theta=(90-b)/180.*3.14159265  #Galactic coordinate
    phi=l/180.*3.14159265
    hid = hp.ang2pix(nside,theta,phi)
    return hid[0]

def readAllFile(hid,nbids,fileDir):
    if os.path.isfile(fileDir+str(hid)+'.fit'):
        b = pf.getdata(fileDir+str(hid)+'.fit')
    else: 
        return -999
    for nbid in nbids:
        if os.path.isfile(fileDir+str(nbid)+'.fit'):
            bb = pf.getdata(fileDir+str(nbid)+'.fit')
            b = hstack2((b,bb))
    return b


def neighbors(ra,dec,cat,radius,photoz):
    """
    return the ra,dec of neighbors brighter than 0.4 L*
    """
    depth=12
    h=es.htm.HTM(depth)
    imag=cat.field('model_mag')[:,3]
    #ok=imag <= limi(photoz)
    ok=imag <= limi0_2(photoz)
    cat=cat[ok]
    fra=cat.field('ra')
    fdec=cat.field('dec')
    srad=np.rad2deg(radius/Da(0,photoz))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=50000)
    r12=np.deg2rad(d12)*Da(0,photoz)
    return m1,m2,r12

def gmrbgcount(cat,gmr_low,gmr_high,imag_low,imag_high):
    ra=cat.field('ra')
    dec=cat.field('dec')
    num=len(ra)
    area=(max(ra)-min(ra))*(max(dec)-min(dec))
    dsty=num/area
    gmr=cat.field('gmr')
    imag=cat.field('imag')
    gmrvalues=np.c_[gmr,imag]
    gmrkde=sts.gaussian_kde(gmrvalues.T)
    bgct=gmrkde.integrate_box([gmr_low,imag_low],[gmr_high,imag_high]) * dsty
    
def GMRrichness(ra=None,dec=None,photoz=None,cat=None,plot=True,err=True,rw=True,bcg=True):
    fra=cat.field('ra')
    fdec=cat.field('dec')
    imag=cat.field('imag')
    gmr=cat.field('model_mag')[:,1] - cat.field('model_mag')[:,2]
    gmrerr=np.sqrt(cat.field('model_magerr')[:,1]**2 + cat.field('model_magerr')[:,2]**2)
    depth=12
    h=es.htm.HTM(depth)
    srad=np.rad2deg(1./Da(0,photoz))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=50000)
    cimag=imag[m2[0]]
    cgmr=gmr[m2[0]]
    r12=np.deg2rad(d12)*Da(0,photoz)
    if bcg is True:
        indices=(imag[m2]<=limi0_2(photoz))*(imag[m2]>cimag)
    else:
        indices=(imag[m2]<=limi0_2(photoz))
    ntot=len(m2[indices])
    if ntot <= 10:
        return 'not enough galaxy brighter than 0.2 L*'
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(gmr[m2[indices]],per=70),sts.scoreatpercentile(gmr[m2[indices]],per=40)])
    sigma=np.array([0.04,0.3])
    if err is True:
        if rw is False:
            aic2=gmm.aic_ecgmm(gmr[m2[indices]],gmrerr[m2[indices]],alpha,mu,sigma)
            aic1=gmm.wstat(gmr[m2[indices]],gmrerr[m2[indices]])[3] 
        else:
            aic2,alpha,mu,sigma=rwgmm.aic2EM(gmr[m2[indices]],gmrerr[m2[indices]],r12[indices],alpha,mu,sigma)
            aic1=rwgmm.aic1EM(gmr[m2[indices]],gmrerr[m2[indices]],r12[indices])[0]
    else:
        aic2=gmm.aic_ecgmm(gmr[m2[indices]],aalpha=alpha,mmu=mu,ssigma=sigma)
        aic1=gmm.wstat(gmr[m2[indices]])[3]
    srt=np.argsort(sigma)
    alpha=alpha[srt]
    mu=mu[srt]
    sigma=sigma[srt]
    z = gmrz(mu[0])
    if plot==True:
        pl.figure(figsize=(12,6))
        pl.subplot(1,2,1)
        hh=pl.hist(gmr[m2[indices]],bins=50,normed=True,facecolor='green',alpha=0.3,range=[-1,3])
        pl.vlines(cgmr,0,hh[0].max()+0.5,color='red',lw=2,linestyle='dashed')
        pl.grid()
        x=np.arange(-1,3,0.01)
        t=gmm.ecgmmplot(x,alpha,mu,sigma)
        pl.xlabel('g - r')
        pl.figtext(0.61,0.85,'Relative Weights: '+str(np.round(alpha,4)))
        pl.figtext(0.61,0.8,'Mean Colors: '+str(np.round(mu,4)))
        pl.figtext(0.61,0.75,'Mean Color Widths: '+str(np.round(sigma,4)))
        pl.figtext(0.61,0.68,'Richness: '+str(np.round(ntot*alpha[0],2)))
        pl.figtext(0.61,0.61,r'$AIC_1$: '+str(np.round(aic1,3)))
        pl.figtext(0.61,0.54,r'$AIC_2$: '+str(np.round(aic2,3)))
        pl.figtext(0.61,0.47,'Test Photoz: '+str(photoz))
        pl.figtext(0.61,0.4,'Ridgeline Photoz: '+str(round(z,3)))
        pl.title('Total # of galaxies: '+str(ntot))
    return ntot*alpha[0],aic1,aic2,cgmr,alpha,mu,sigma,z

def RMIrichness(ra=None,dec=None,photoz=None,cat=None,plot=True,err=True,rw=True,bcg=True):
    fra=cat.field('ra')
    fdec=cat.field('dec')
    imag=cat.field('imag')
    rmi=cat.field('model_mag')[:,2] - cat.field('model_mag')[:,3]
    rmierr=np.sqrt(cat.field('model_magerr')[:,2]**2 + cat.field('model_magerr')[:,3]**2)
    depth=12
    h=es.htm.HTM(depth)
    srad=np.rad2deg(1./Da(0,photoz))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=50000)
    r12=np.deg2rad(d12)*Da(0,photoz)
    cimag=imag[m2[0]]
    crmi=rmi[m2[0]]
    if bcg is True:
        indices=(imag[m2]<=limi0_2(photoz))*(imag[m2]>cimag)
    else:
        indices=(imag[m2]<=limi0_2(photoz))
    ntot=len(m2[indices])
    if ntot <= 10:
        return 'not enough galaxy brighter than 0.2 L*'
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(rmi[m2[indices]],per=70),sts.scoreatpercentile(rmi[m2[indices]],per=40)])
    sigma=np.array([0.04,0.3])
    if err is True:              
        if rw is False:
            aic2=gmm.aic_ecgmm(rmi[m2[indices]],rmierr[m2[indices]],alpha,mu,sigma)
            aic1=gmm.wstat(rmi[m2[indices]],rmierr[m2[indices]])[3] 
        else:
            aic2,alpha,mu,sigma=rwgmm.aic2EM(rmi[m2[indices]],rmierr[m2[indices]],r12[indices],alpha,mu,sigma)
            aic1=rwgmm.aic1EM(rmi[m2[indices]],rmierr[m2[indices]],r12[indices])[0]
    else:
        aic2=gmm.aic_ecgmm(rmi[m2[indices]],aalpha=alpha,mmu=mu,ssigma=sigma)
        aic1=gmm.wstat(rmi[m2[indices]])[3]
    srt=np.argsort(sigma)
    alpha=alpha[srt]
    mu=mu[srt]
    sigma=sigma[srt]
    z = rmiz(mu[0])
    if plot==True:
        pl.figure(figsize=(12,6))
        pl.subplot(1,2,1)
        hh=pl.hist(rmi[m2[indices]],bins=50,normed=True,facecolor='green',alpha=0.3,range=[-1,3])
        pl.vlines(crmi,0,hh[0].max()+0.5,color='red',lw=2,linestyle='dashed')
        pl.grid()
        x=np.arange(-1,3,0.01)
        t=gmm.ecgmmplot(x,alpha,mu,sigma)
        pl.xlabel('r - i')
        pl.figtext(0.61,0.85,'Relative Weights: '+str(np.round(alpha,4)))
        pl.figtext(0.61,0.8,'Mean Colors: '+str(np.round(mu,4)))
        pl.figtext(0.61,0.75,'Mean Color Widths: '+str(np.round(sigma,4)))
        pl.figtext(0.61,0.68,'Richness: '+str(np.round(ntot*alpha[0],2)))
        pl.figtext(0.61,0.61,r'$AIC_1$: '+str(aic1))
        pl.figtext(0.61,0.54,r'$AIC_2$: '+str(aic2))
        pl.figtext(0.61,0.47,'Test Photoz: '+str(photoz))
        pl.figtext(0.61,0.4,'Ridgeline Photoz: '+str(round(z,3)))
        pl.title('Total # of galaxies: '+str(ntot))
    return ntot*alpha[0],aic1,aic2,crmi,alpha,mu,sigma,z


def getRichness(ra,dec,photoz,err=True,rw=True,bcg=False,plot=True,iter=True):
    sdssradec(ra,dec)
    nside = 2**5
    hid = healpixIDradec(ra,dec,nside)
    nbids = hp.get_all_neighbours(nside,hid)
    dr8=readAllFile(hid,nbids,'/home/jghao/research/data/hpixFileDR8/')
    if dr8 == -999:
        return 'source is not in the sdss dr8 footprint'
    ridgeline_z = 0.
    pl.figure(figsize=(7,6))
    if iter == True:
        for itr in range(5):
            print '---iteration: '+str(itr)+' ----'
            zdiff = abs(photoz - ridgeline_z)
            if zdiff <= 0.01:
                break
            else:
                pl.close()
                if itr != 0:
                    photoz = ridgeline_z
                if photoz < 0.4:
                    res=GMRrichness(ra,dec,photoz,dr8,err=err,rw=rw,bcg=bcg,plot=plot)
                elif photoz >= 0.4 and photoz < 0.75:
                    res=RMIrichness(ra,dec,photoz,dr8,err=err,rw=rw,bcg=bcg,plot=plot)
                if res == 'not enough galaxy brighter than 0.2 L*':
                    return 'not enough galaxy brighter than 0.2 L*'
                else:
                    ridgeline_z = res[-1]
                if ridgeline_z == 0:
                    ridgeline_z = ridgeline_z + 0.01
    else:
        if photoz < 0.4:
            res=GMRrichness(ra,dec,photoz,dr8,err=err,rw=rw,bcg=bcg,plot=plot)
        elif photoz >= 0.4 and photoz < 0.75:
            res=RMIrichness(ra,dec,photoz,dr8,err=err,rw=rw,bcg=bcg,plot=plot)
        #here res=[rich,aic1,aic2,ccolor,alpha,mu,sigma,ridgeline_z]
    return res


