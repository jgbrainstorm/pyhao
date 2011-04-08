import numpy as np
import pyfits as pf
import pylab as pl
import esutil as es
import ecgmmPy as gmm
import scipy.stats as sts
import glob as gl

#-----0.4L*----------
def limi(x):
    A=np.exp(3.1638)
    k=0.1428
    lmi=A*x**k
    return(lmi)

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
    
def GMRrichness(ra,dec,photoz,cat,plot=True,err=True):
    fra=cat.field('ra')
    fdec=cat.field('dec')
    imag=cat.field('model_counts')[:,3]
    gmr=cat.field('gmr')
    gmrerr=cat.field('gmr_err')
    depth=12
    h=es.htm.HTM(depth)
    srad=np.rad2deg(1./es.cosmology.Da(0,photoz,h=0.7))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=5000)
    indices=(imag[m2]<=limi(photoz))
    ntot=len(m2[indices])
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(gmr[m2[indices]],per=70),sts.scoreatpercentile(gmr[m2[indices]],per=40)])
    sigma=np.array([0.04,0.3])
    if err is True:
        aic2=gmm.aic_ecgmm(gmr[m2[indices]],gmrerr[m2[indices]],alpha,mu,sigma)
        aic1=gmm.wstat(gmr[m2[indices]],gmrerr[m2[indices]])[3] 
    else:
        aic2=gmm.aic_ecgmm(gmr[m2[indices]],aalpha=alpha,mmu=mu,ssigma=sigma)
        aic1=gmm.wstat(gmr[m2[indices]])[3] 
    if plot==True:
        pl.hist(gmr[m2[indices]],bins=30,normed=True,histtype='step')
        x=np.arange(-1,5,0.01)
        srt=np.argsort(sigma)
        alpha=alpha[srt]
        mu=mu[srt]
        sigma=sigma[srt]
        t=gmm.ecgmmplot(x,alpha,mu,sigma)
        pl.xlabel('g - r')
        pl.figtext(0.61,0.85,r'$\alpha$: '+str(np.round(alpha,4)))
        pl.figtext(0.61,0.8,r'$\mu$: '+str(np.round(mu,4)))
        pl.figtext(0.61,0.75,r'$\sigma$: '+str(np.round(sigma,4)))
        pl.figtext(0.61,0.68,r'$Ngals$: '+str(np.round(ntot*alpha[0])))
        pl.figtext(0.61,0.61,r'$AIC_1$: '+str(aic1))
        pl.figtext(0.61,0.54,r'$AIC_2$: '+str(aic2))
        pl.figtext(0.61,0.47,'Photoz: '+str(photoz))
        pl.title('Total # of galaxies: '+str(ntot))
    return ntot*alpha[0]

def RMIrichness(ra,dec,photoz,cat,plot=True,err=True):
    fra=cat.field('ra')
    fdec=cat.field('dec')
    imag=cat.field('model_counts')[:,3]
    rmi=cat.field('rmi')      
    rmierr=cat.field('rmi_err')
    depth=12
    h=es.htm.HTM(depth)
    srad=np.rad2deg(1./es.cosmology.Da(0,photoz,h=0.7))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=5000)
    indices=(imag[m2]<=limi(photoz))
    ntot=len(m2[indices])
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(rmi[m2[indices]],per=70),sts.scoreatpercentile(rmi[m2[indices]],per=40)])
    sigma=np.array([0.04,0.3])
    if err is True:                    
        aic2=gmm.aic_ecgmm(rmi[m2[indices]],rmierr[m2[indices]],alpha,mu,sigma)
        aic1=gmm.wstat(rmi[m2[indices]],rmierr[m2[indices]])[3] 
    else:
        aic2=gmm.aic_ecgmm(rmi[m2[indices]],aalpha=alpha,mmu=mu,ssigma=sigma)
        aic1=gmm.wstat(rmi[m2[indices]])[3] 
    if plot==True:
        pl.hist(rmi[m2[indices]],bins=30,normed=True,histtype='step')
        x=np.arange(-1,5,0.01)
        srt=np.argsort(sigma)
        alpha=alpha[srt]
        mu=mu[srt]
        sigma=sigma[srt]
        t=gmm.ecgmmplot(x,alpha,mu,sigma)
        pl.xlabel('r - i')
        pl.figtext(0.61,0.85,r'$\alpha$: '+str(np.round(alpha,4)))
        pl.figtext(0.61,0.8,r'$\mu$: '+str(np.round(mu,4)))
        pl.figtext(0.61,0.75,r'$\sigma$: '+str(np.round(sigma,4)))
        pl.figtext(0.61,0.68,r'$Ngals$: '+str(np.round(ntot*alpha[0])))
        pl.figtext(0.61,0.61,r'$AIC_1$: '+str(aic1))
        pl.figtext(0.61,0.54,r'$AIC_2$: '+str(aic2))
        pl.figtext(0.61,0.47,'Photoz: '+str(photoz))
        pl.title('Total # of galaxies: '+str(ntot))
    return ntot*alpha[0]

def IMZrichness(ra,dec,photoz,cat,plot=True,err=True):
    fra=cat.field('ra')
    fdec=cat.field('dec')
    imag=cat.field('model_counts')[:,3]
    imz=cat.field('imz')
    imzerr=cat.field('imz_err')
    depth=12
    h=es.htm.HTM(depth)
    srad=np.rad2deg(1./es.cosmology.Da(0,photoz,h=0.7))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=5000)
    indices=(imag[m2]<=limi(photoz))
    ntot=len(m2[indices])
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(imz[m2[indices]],per=70),sts.scoreatpercentile(imz[m2[indices]],per=40)])
    sigma=np.array([0.04,0.3])
    if err is True:
        aic2=gmm.aic_ecgmm(imz[m2[indices]],imzerr[m2[indices]],alpha,mu,sigma)
        aic1=gmm.wstat(imz[m2[indices]],imzerr[m2[indices]])[3] 
    else:
        aic2=gmm.aic_ecgmm(imz[m2[indices]],aalpha=alpha,mmu=mu,ssigma=sigma)
        aic1=gmm.wstat(imz[m2[indices]])[3] 
    if plot==True:
        pl.hist(imz[m2[indices]],bins=30,normed=True,histtype='step')
        x=np.arange(-1,5,0.01)
        srt=np.argsort(sigma)
        alpha=alpha[srt]
        mu=mu[srt]
        sigma=sigma[srt]
        t=gmm.ecgmmplot(x,alpha,mu,sigma)
        pl.xlabel('i - z')
        pl.figtext(0.61,0.85,r'$\alpha$: '+str(np.round(alpha,4)))
        pl.figtext(0.61,0.8,r'$\mu$: '+str(np.round(mu,4)))
        pl.figtext(0.61,0.75,r'$\sigma$: '+str(np.round(sigma,4)))
        pl.figtext(0.61,0.68,r'$Ngals$: '+str(np.round(ntot*alpha[0])))
        pl.figtext(0.61,0.61,r'$AIC_1$: '+str(aic1))
        pl.figtext(0.61,0.54,r'$AIC_2$: '+str(aic2))
        pl.figtext(0.61,0.47,'Photoz: '+str(photoz))
        pl.title('Total # of galaxies: '+str(ntot))
    return ntot*alpha[0]

def getRichness(ra,dec,photoz,err=None):
    if ra < 10:
        catid=0
    elif ra > 10 and ra < 20:
        catid=1
    elif ra >20 and ra < 30:
        catid=2
    elif ra >30 and ra < 40:
        catid=3
    elif ra >40 and ra < 50:
        catid=4
    elif ra >50 and ra < 60:
        catid=5
    elif ra >300 and ra < 310:
        catid=6
    elif ra >310 and ra < 320:
        catid=7
    elif ra >320 and ra < 330:
        catid=8
    elif ra >330 and ra < 340:
        catid=9
    elif ra >340 and ra < 350:
        catid=10
    elif ra >350 and ra < 360:
        catid=11
    coadd=pf.getdata('/home/jghao/research/data/coadd10_29_09/gmbcg_input_small_'+str(catid)+'.fit')
    if photoz < 0.4:
        rich=GMRrichness(ra,dec,photoz,coadd,err=err)
    elif photoz >= 0.4 and photoz < 0.75:
        rich=RMIrichness(ra,dec,photoz,coadd,err=err)
    elif photoz >= 0.75:
        rich=IMZrichness(ra,dec,photoz,coadd,err=err)
    return rich

