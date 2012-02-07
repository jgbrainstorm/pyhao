import numpy as np
import pyfits as pf
import pylab as pl
import esutil as es
import ecgmmPy as gmm
import rwecgmmPy as rwgmm
import scipy.stats as sts
import glob as gl


Da=es.cosmology.Cosmo(h=0.7).Da
#-----0.4L*----------
def limi(x):
    A=np.exp(3.1638)
    k=0.1428
    lmi=A*x**k
    return(lmi)

def neighbors(ra,dec,cat,radius,photoz):
    """
    return the ra,dec of neighbors brighter than 0.4 L*
    """
    depth=12
    h=es.htm.HTM(depth)
    imag=cat.field('model_counts')[:,3]
    ok=imag <= limi(photoz)
    cat=cat[ok]
    fra=cat.field('ra')
    fdec=cat.field('dec')
    srad=np.rad2deg(radius/Da(0,photoz))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=5000)
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
    imag=cat.field('mag_z')
    gmr=cat.field('mag_g') - cat.field('mag_r')
    gmrerr=np.sqrt(cat.field('MAGERR_G')**2+cat.field('MAGERR_r')**2)
    depth=10
    h=es.htm.HTM(depth)
    srad=np.rad2deg(1./Da(0,photoz))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=5000)
    cimag=imag[m2[0]]
    cgmr=gmr[m2[0]]
    r12=np.deg2rad(d12)*Da(0,photoz)
    if bcg is True:
        indices=(imag[m2]<=limi(photoz))*(imag[m2]>cimag)
    else:
        indices=(imag[m2]<=limi(photoz))
    ntot=len(m2[indices])
    if ntot <= 10:
        return 0, 0, 0
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(gmr[m2[indices]],per=70),sts.scoreatpercentile(gmr[m2[indices]],per=40)])
    sigma=np.array([0.04,0.3])
    if err is True:
        if rw is False:
            bic2=gmm.bic_ecgmm(gmr[m2[indices]],gmrerr[m2[indices]],alpha,mu,sigma)
            bic1=gmm.wstat(gmr[m2[indices]],gmrerr[m2[indices]])[3] 
        else:
            bic2,alpha,mu,sigma=rwgmm.bic2EM(gmr[m2[indices]],gmrerr[m2[indices]],r12[indices],alpha,mu,sigma)
            bic1=rwgmm.bic1EM(gmr[m2[indices]],gmrerr[m2[indices]],r12[indices])[0]
    else:
        bic2=gmm.bic_ecgmm(gmr[m2[indices]],aalpha=alpha,mmu=mu,ssigma=sigma)
        bic1=gmm.wstat(gmr[m2[indices]])[3] 
    if plot==True:
        pl.hist(gmr[m2[indices]],bins=30,normed=True,facecolor='green',alpha=0.3)
        pl.vlines(cgmr,0,3,color='pink')
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
        pl.figtext(0.61,0.68,r'$Amplitude$: '+str(np.round(ntot*alpha[0],2)))
        pl.figtext(0.61,0.61,r'$BIC_1$: '+str(bic1))
        pl.figtext(0.61,0.54,r'$BIC_2$: '+str(bic2))
        pl.figtext(0.61,0.47,'Photoz: '+str(photoz))
        pl.title('Total # of galaxies: '+str(ntot))
    return ntot*alpha[0],bic1,bic2

def RMIrichness(ra=None,dec=None,photoz=None,cat=None,plot=True,err=True,rw=True,bcg=True):
    fra=cat.field('ra')
    fdec=cat.field('dec')
    imag=cat.field('mag_z')
    rmi=cat.field('mag_r') - cat.field('mag_i')
    rmierr=np.sqrt(cat.field('MAGERR_r')**2+cat.field('MAGERR_i')**2)
    depth=10
    h=es.htm.HTM(depth)
    srad=np.rad2deg(1./Da(0,photoz))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=5000)
    r12=np.deg2rad(d12)*Da(0,photoz)
    cimag=imag[m2[0]]
    crmi=rmi[m2[0]]
    if bcg is True:
        indices=(imag[m2]<=limi(photoz))*(imag[m2]>cimag)
    else:
        indices=(imag[m2]<=limi(photoz))
    ntot=len(m2[indices])
    if ntot <= 10:
        return 0, 0, 0
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(rmi[m2[indices]],per=70),sts.scoreatpercentile(rmi[m2[indices]],per=40)])
    sigma=np.array([0.04,0.3])
    if err is True:              
        if rw is False:
            bic2=gmm.bic_ecgmm(rmi[m2[indices]],rmierr[m2[indices]],alpha,mu,sigma)
            bic1=gmm.wstat(rmi[m2[indices]],rmierr[m2[indices]])[3] 
        else:
            bic2,alpha,mu,sigma=rwgmm.bic2EM(rmi[m2[indices]],rmierr[m2[indices]],r12[indices],alpha,mu,sigma)
            bic1=rwgmm.bic1EM(rmi[m2[indices]],rmierr[m2[indices]],r12[indices])[0]
    else:
        bic2=gmm.bic_ecgmm(rmi[m2[indices]],aalpha=alpha,mmu=mu,ssigma=sigma)
        bic1=gmm.wstat(rmi[m2[indices]])[3] 
    if plot==True:
        pl.hist(rmi[m2[indices]],bins=30,normed=True,facecolor='green',alpha=0.3)
        pl.vlines(crmi,0,3,color='pink')
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
        pl.figtext(0.61,0.68,r'$Amplitude$: '+str(np.round(ntot*alpha[0],2)))
        pl.figtext(0.61,0.61,r'$BIC_1$: '+str(bic1))
        pl.figtext(0.61,0.54,r'$BIC_2$: '+str(bic2))
        pl.figtext(0.61,0.47,'Photoz: '+str(photoz))
        pl.title('Total # of galaxies: '+str(ntot))
    return ntot*alpha[0],bic1,bic2

def IMZrichness(ra=None,dec=None,photoz=None,cat=None,plot=True,err=True,rw=True,bcg=True):
    fra=cat.field('ra')
    fdec=cat.field('dec')
    imag=cat.field('mag_z')
    imz=cat.field('mag_i') - cat.field('mag_z')
    imzerr=np.sqrt(cat.field('MAGERR_i')**2+cat.field('MAGERR_z')**2)
    depth=10
    h=es.htm.HTM(depth)
    srad=np.rad2deg(1./Da(0,photoz))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=5000)
    cimag=imag[m2[0]]
    cimz=imz[m2[0]]
    r12=np.deg2rad(d12)*Da(0,photoz)
    if bcg is True:
        indices=(imag[m2]<=limi(photoz))*(imag[m2]>cimag)
    else:
        indices=(imag[m2]<=limi(photoz))
    ntot=len(m2[indices])
    if ntot <= 10:
        return 0, 0, 0
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(imz[m2[indices]],per=70),sts.scoreatpercentile(imz[m2[indices]],per=40)])
    sigma=np.array([0.04,0.3])
    if err is True:
        if rw is False:
            bic2=gmm.bic_ecgmm(imz[m2[indices]],imzerr[m2[indices]],alpha,mu,sigma)
            bic1=gmm.wstat(imz[m2[indices]],imzerr[m2[indices]])[3] 
        else:
            bic2,alpha,mu,sigma=rwgmm.bic2EM(imz[m2[indices]],imzerr[m2[indices]],r12[indices],alpha,mu,sigma)
            bic1=rwgmm.bic1EM(imz[m2[indices]],imzerr[m2[indices]],r12[indices])[0]
    else:
        bic2=gmm.bic_ecgmm(imz[m2[indices]],aalpha=alpha,mmu=mu,ssigma=sigma)
        bic1=gmm.wstat(imz[m2[indices]])[3] 
    if plot==True:
        pl.hist(imz[m2[indices]],bins=30,normed=True,facecolor='green',alpha=0.3)
        pl.vlines(cimz,0,3,color='pink')
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
        pl.figtext(0.61,0.68,r'$Amplitude$: '+str(np.round(ntot*alpha[0],2)))
        pl.figtext(0.61,0.61,r'$BIC_1$: '+str(bic1))
        pl.figtext(0.61,0.54,r'$BIC_2$: '+str(bic2))
        pl.figtext(0.61,0.47,'Photoz: '+str(photoz))
        pl.title('Total # of galaxies: '+str(ntot))
    return ntot*alpha[0],bic1,bic2


def ZMYrichness(ra=None,dec=None,photoz=None,cat=None,plot=True,err=True,rw=True,bcg=True):
    fra=cat.field('ra')
    fdec=cat.field('dec')
    imag=cat.field('mag_z')
    zmy=cat.field('mag_z') - cat.field('mag_y')
    zmyerr=np.sqrt(cat.field('MAGERR_z')**2+cat.field('MAGERR_y')**2)
    depth=10
    h=es.htm.HTM(depth)
    srad=np.rad2deg(1./Da(0,photoz))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=5000)
    cimag=imag[m2[0]]
    czmy=zmy[m2[0]]
    r12=np.deg2rad(d12)*Da(0,photoz)
    if bcg is True:
        indices=(imag[m2]<=limi(photoz))*(imag[m2]>cimag)
    else:
        indices=(imag[m2]<=limi(photoz))
    ntot=len(m2[indices])
    if ntot <= 10:
        return 0, 0, 0
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(zmy[m2[indices]],per=70),sts.scoreatpercentile(zmy[m2[indices]],per=40)])
    sigma=np.array([0.04,0.3])
    if err is True:
        if rw is False:
            bic2=gmm.bic_ecgmm(zmy[m2[indices]],zmyerr[m2[indices]],alpha,mu,sigma)
            bic1=gmm.wstat(zmy[m2[indices]],zmyerr[m2[indices]])[3] 
        else:
            bic2,alpha,mu,sigma=rwgmm.bic2EM(zmy[m2[indices]],zmyerr[m2[indices]],r12[indices],alpha,mu,sigma)
            bic1=rwgmm.bic1EM(zmy[m2[indices]],zmyerr[m2[indices]],r12[indices])[0]
    else:
        bic2=gmm.bic_ecgmm(zmy[m2[indices]],aalpha=alpha,mmu=mu,ssigma=sigma)
        bic1=gmm.wstat(zmy[m2[indices]])[3] 
    if plot==True:
        pl.hist(zmy[m2[indices]],bins=30,normed=True,facecolor='green',alpha=0.3)
        pl.vlines(czmy,0,3,color='pink')
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
        pl.figtext(0.61,0.68,r'$Amplitude$: '+str(np.round(ntot*alpha[0],2)))
        pl.figtext(0.61,0.61,r'$BIC_1$: '+str(bic1))
        pl.figtext(0.61,0.54,r'$BIC_2$: '+str(bic2))
        pl.figtext(0.61,0.47,'Photoz: '+str(photoz))
        pl.title('Total # of galaxies: '+str(ntot))
    return ntot*alpha[0],bic1,bic2

def getRich(ra,dec,photoz,err=True,rw=True,bcg=True,plot=True):
    mock = pf.getdata('/home/jghao/research/data/des_mock/v3.04/obsCat/DES_Mock_v3.04_Baseline_06.fit')
    if plot == True:
        pl.figure(figsize=(7,6))
    if photoz < 0.4:
        rich,bic1,bic2=GMRrichness(ra,dec,photoz,mock,err=err,rw=rw,bcg=bcg,plot=plot)
    elif photoz >= 0.4 and photoz < 0.75:
        rich,bic1,bic2=RMIrichness(ra,dec,photoz,mock,err=err,rw=rw,bcg=bcg,plot=plot)
    elif photoz >= 0.75:
        rich,bic1,bic2=IMZrichness(ra,dec,photoz,mock,err=err,rw=rw,bcg=bcg,plot=plot)
    elif photoz >= 1.:
        rich,bic1,bic2=ZMYrichness(ra,dec,photoz,mock,err=err,rw=rw,bcg=bcg,plot=plot)
    return rich,bic1,bic2


