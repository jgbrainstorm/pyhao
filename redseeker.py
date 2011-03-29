
import numpy as np
import pyfits as pf
import pylab as pl
import esutil as es
import ecgmmPy as gmm
import scipy.stats as sts

#-----0.4L*----------
def limi(x):
    A=np.exp(3.1638)
    k=0.1428
    lmi=A*x**k
    return(lmi)


def redsker(b,idx,err=True):
    depth=12
    h=es.htm.HTM(depth)
    ra=b.field('ra')
    dec=b.field('dec')
    photoz=b.field('z')
    central=b.field('central')
    gmr=b.field('omag')[:,0]-b.field('omag')[:,1]
    rmi=b.field('omag')[:,1]-b.field('omag')[:,2]
    imz=b.field('omag')[:,2]-b.field('omag')[:,3]
    gmz=b.field('omag')[:,0]-b.field('omag')[:,3]
    rmz=b.field('omag')[:,1]-b.field('omag')[:,3]
    gmi=b.field('omag')[:,0]-b.field('omag')[:,2]
    num=len(ra)
    if err:
        gmrerr=b.field('omagerr')[:,0]-b.field('omagerr')[:,1]
        rmierr=b.field('omagerr')[:,1]-b.field('omagerr')[:,2]
        imzerr=b.field('omagerr')[:,2]-b.field('omagerr')[:,3]
        gmzerr=b.field('omagerr')[:,0]-b.field('omagerr')[:,3]
        rmzerr=b.field('omagerr')[:,1]-b.field('omagerr')[:,3]
        gmierr=b.field('omagerr')[:,0]-b.field('omagerr')[:,2]
    else:
        gmrerr=np.zeros(num)
        rmierr=np.zeros(num)
        imzerr=np.zeros(num)
        gmzerr=np.zeros(num)
        rmzerr=np.zeros(num)
        gmierr=np.zeros(num)
    iamag=b.field('amag')[:,2]
    imag=b.field('omag')[:,2]
    srad=np.rad2deg(1./es.cosmology.Da(0,photoz[idx],h=0.7)/(1+photoz[idx]))
    m1,m2,d12 = h.match(ra[idx],dec[idx],ra,dec,srad,maxmatch=5000)
    indices=(imag[m2]<=limi(photoz[idx]))*(imag[m2]>imag[m1])
    #indices=(iamag[m2]<=-20)*(iamag[m2]>iamag[m1])
    ntot=len(m2[indices])
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(gmr[m2[indices]],per=80),sts.scoreatpercentile(gmr[m2[indices]],per=30)])
    sigma=np.array([0.04,0.3])
    aic2=gmm.aic_ecgmm(gmr[m2[indices]],gmrerr[m2[indices]],alpha,mu,sigma)
    aic1=gmm.wstat(gmr[m2[indices]],gmrerr[m2[indices]])[3]
    fig=pl.figure(figsize=(15,8)) 
    ax=fig.add_subplot(2,3,1)
    pl.hist(gmr[m2[indices]],bins=30,normed=True,histtype='step')
    x=np.arange(-1,5,0.01)
    t=gmm.ecgmmplot(x,alpha,mu,sigma)
    pl.xlabel('g - r')
    pl.title('M200: '+str(b[idx].field('m200')))
    pl.text(0.1,0.85,r'$\alpha$: '+str(np.round(alpha,4)),transform = ax.transAxes)
    pl.text(0.1,0.8,r'$\mu$: '+str(np.round(mu,4)),transform = ax.transAxes)
    pl.text(0.1,0.75,r'$\sigma$: '+str(np.round(sigma,4)),transform = ax.transAxes)
    pl.text(0.1,0.68,r'$Ngals$: '+str(np.round(ntot*alpha[0])),transform = ax.transAxes)
    pl.text(0.1,0.6,r'$AIC$: '+str(np.round(aic1))+', '+str(np.round(aic2)),transform = ax.transAxes)
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(rmi[m2[indices]],per=80),sts.scoreatpercentile(rmi[m2[indices]],per=30)])
    sigma=np.array([0.04,0.3])
    aic2=gmm.aic_ecgmm(rmi[m2[indices]],rmierr[m2[indices]],alpha,mu,sigma)
    aic1=gmm.wstat(rmi[m2[indices]],rmierr[m2[indices]])[3]
    ax=fig.add_subplot(2,3,2)
    pl.hist(rmi[m2[indices]],bins=30,normed=True,histtype='step')
    x=np.arange(-1,5,0.01)
    t=gmm.ecgmmplot(x,alpha,mu,sigma)
    pl.xlabel('r - i')
    pl.title('photoz: '+str(photoz[idx]))
    pl.xlim(-0.2,2.5)
    pl.text(0.1,0.85,r'$\alpha$: '+str(np.round(alpha,4)),transform = ax.transAxes)
    pl.text(0.1,0.8,r'$\mu$: '+str(np.round(mu,4)),transform = ax.transAxes)
    pl.text(0.1,0.75,r'$\sigma$: '+str(np.round(sigma,4)),transform = ax.transAxes)
    pl.text(0.1,0.68,r'$Ngals$: '+str(np.round(ntot*alpha[0])),transform = ax.transAxes)
    pl.text(0.1,0.6,r'$AIC$: '+str(np.round(aic1))+', '+str(np.round(aic2)),transform = ax.transAxes)
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(imz[m2[indices]],per=60),sts.scoreatpercentile(imz[m2[indices]],per=30)])
    sigma=np.array([0.02,0.3])
    aic2=gmm.aic_ecgmm(imz[m2[indices]],imzerr[m2[indices]],alpha,mu,sigma)
    aic1=gmm.wstat(imz[m2[indices]],imzerr[m2[indices]])[3]
    ax=fig.add_subplot(2,3,3)
    pl.hist(imz[m2[indices]],bins=30,normed=True,histtype='step')
    x=np.arange(-1,5,0.01)
    t=gmm.ecgmmplot(x,alpha,mu,sigma)
    pl.xlabel('i - z')
    pl.title('Ntot: '+str(ntot))
    pl.xlim(-0.2,2.5)
    pl.text(0.1,0.85,r'$\alpha$: '+str(np.round(alpha,4)),transform = ax.transAxes)
    pl.text(0.1,0.8,r'$\mu$: '+str(np.round(mu,4)),transform = ax.transAxes)
    pl.text(0.1,0.75,r'$\sigma$: '+str(np.round(sigma,4)),transform = ax.transAxes)
    pl.text(0.1,0.68,r'$Ngals$: '+str(np.round(ntot*alpha[0])),transform = ax.transAxes)
    pl.text(0.1,0.6,r'$AIC$: '+str(np.round(aic1))+', '+str(np.round(aic2)),transform = ax.transAxes)
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(gmz[m2[indices]],per=60),sts.scoreatpercentile(gmz[m2[indices]],per=30)])
    sigma=np.array([0.02,0.3])
    aic2=gmm.aic_ecgmm(gmz[m2[indices]],gmzerr[m2[indices]],alpha,mu,sigma)
    aic1=gmm.wstat(gmz[m2[indices]],gmzerr[m2[indices]])[3]
    ax=fig.add_subplot(2,3,4)
    pl.hist(gmz[m2[indices]],bins=30,normed=True,histtype='step')
    x=np.arange(-1,5,0.01)
    t=gmm.ecgmmplot(x,alpha,mu,sigma)
    pl.xlabel('g - z')
    pl.text(0.1,0.85,r'$\alpha$: '+str(np.round(alpha,4)),transform = ax.transAxes)
    pl.text(0.1,0.8,r'$\mu$: '+str(np.round(mu,4)),transform = ax.transAxes)
    pl.text(0.1,0.75,r'$\sigma$: '+str(np.round(sigma,4)),transform = ax.transAxes)
    pl.text(0.1,0.68,r'$Ngals$: '+str(np.round(ntot*alpha[0])),transform = ax.transAxes)
    pl.text(0.1,0.6,r'$AIC$: '+str(np.round(aic1))+', '+str(np.round(aic2)),transform = ax.transAxes)
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(rmz[m2[indices]],per=60),sts.scoreatpercentile(rmz[m2[indices]],per=30)])
    sigma=np.array([0.02,0.3])
    aic2=gmm.aic_ecgmm(rmz[m2[indices]],rmzerr[m2[indices]],alpha,mu,sigma)
    aic1=gmm.wstat(rmz[m2[indices]],rmzerr[m2[indices]])[3]
    ax=fig.add_subplot(2,3,5)
    pl.hist(rmz[m2[indices]],bins=30,normed=True,histtype='step')
    x=np.arange(-1,5,0.01)
    t=gmm.ecgmmplot(x,alpha,mu,sigma)
    pl.xlabel('r - z')
    pl.text(0.1,0.85,r'$\alpha$: '+str(np.round(alpha,4)),transform = ax.transAxes)
    pl.text(0.1,0.8,r'$\mu$: '+str(np.round(mu,4)),transform = ax.transAxes)
    pl.text(0.1,0.75,r'$\sigma$: '+str(np.round(sigma,4)),transform = ax.transAxes)
    pl.text(0.1,0.68,r'$Ngals$: '+str(np.round(ntot*alpha[0])),transform = ax.transAxes)
    pl.text(0.1,0.6,r'$AIC$: '+str(np.round(aic1))+', '+str(np.round(aic2)),transform = ax.transAxes)
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(gmi[m2[indices]],per=60),sts.scoreatpercentile(gmi[m2[indices]],per=30)])
    sigma=np.array([0.02,0.3])
    aic2=gmm.aic_ecgmm(gmi[m2[indices]],gmierr[m2[indices]],alpha,mu,sigma)
    aic1=gmm.wstat(gmi[m2[indices]],gmierr[m2[indices]])[3]
    ax=fig.add_subplot(2,3,6)
    pl.hist(gmi[m2[indices]],bins=30,normed=True,histtype='step')
    x=np.arange(-1,5,0.01)
    t=gmm.ecgmmplot(x,alpha,mu,sigma)
    pl.xlabel('g - i')
    pl.text(0.1,0.85,r'$\alpha$: '+str(np.round(alpha,4)),transform = ax.transAxes)
    pl.text(0.1,0.8,r'$\mu$: '+str(np.round(mu,4)),transform = ax.transAxes)
    pl.text(0.1,0.75,r'$\sigma$: '+str(np.round(sigma,4)),transform = ax.transAxes)
    pl.text(0.1,0.68,r'$Ngals$: '+str(np.round(ntot*alpha[0])),transform = ax.transAxes)
    pl.text(0.1,0.6,r'$AIC$: '+str(np.round(aic1))+', '+str(np.round(aic2)),transform = ax.transAxes)
    return('Plot is done!')



def GMRrichness(ra,dec,photoz,cat,plot=True):
    fra=cat.field('ra')
    fdec=cat.field('dec')
    imag=cat.field('model_counts')[:,3]
    gmr=cat.field('gmr')
    gmrerr=cat.field('gmr_err')
    depth=12
    h=es.htm.HTM(depth)
    srad=np.rad2deg(1./es.cosmology.Da(0,photoz,h=0.7)/(1+photoz))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=5000)
    indices=(imag[m2]<=limi(photoz))
    ntot=len(m2[indices])
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(gmr[m2[indices]],per=70),sts.scoreatpercentile(gmr[m2[indices]],per=40)])
    sigma=np.array([0.04,0.3])
    aic2=gmm.aic_ecgmm(gmr[m2[indices]],gmrerr[m2[indices]],alpha,mu,sigma)
    aic1=gmm.wstat(gmr[m2[indices]],gmrerr[m2[indices]])[3] 
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

def RMIrichness(ra,dec,photoz,cat,plot=True):
    fra=cat.field('ra')
    fdec=cat.field('dec')
    imag=cat.field('model_counts')[:,3]
    rmi=cat.field('rmi')
    rmierr=cat.field('rmi_err')
    depth=12
    h=es.htm.HTM(depth)
    srad=np.rad2deg(1./es.cosmology.Da(0,photoz,h=0.7)/(1+photoz))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=5000)
    indices=(imag[m2]<=limi(photoz))
    ntot=len(m2[indices])
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(rmi[m2[indices]],per=70),sts.scoreatpercentile(rmi[m2[indices]],per=40)])
    sigma=np.array([0.04,0.3])
    aic2=gmm.aic_ecgmm(rmi[m2[indices]],rmierr[m2[indices]],alpha,mu,sigma)
    aic1=gmm.wstat(rmi[m2[indices]],rmierr[m2[indices]])[3] 
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

def IMZrichness(ra,dec,photoz,cat,plot=True):
    fra=cat.field('ra')
    fdec=cat.field('dec')
    imag=cat.field('model_counts')[:,3]
    imz=cat.field('imz')
    imzerr=cat.field('imz_err')
    depth=12
    h=es.htm.HTM(depth)
    srad=np.rad2deg(1./es.cosmology.Da(0,photoz,h=0.7)/(1+photoz))
    m1,m2,d12 = h.match(ra,dec,fra,fdec,srad,maxmatch=5000)
    indices=(imag[m2]<=limi(photoz))
    ntot=len(m2[indices])
    alpha=np.array([0.5,0.5])
    mu=np.array([sts.scoreatpercentile(imz[m2[indices]],per=70),sts.scoreatpercentile(imz[m2[indices]],per=40)])
    sigma=np.array([0.04,0.3])
    aic2=gmm.aic_ecgmm(imz[m2[indices]],imzerr[m2[indices]],alpha,mu,sigma)
    aic1=gmm.wstat(imz[m2[indices]],imzerr[m2[indices]])[3] 
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



def getRichness(ra,dec,photoz,i):
    """
    estimate the richness by giving a ra,dec and photoz
    """
    if ra[i] < 10:
        catid=0
    elif ra[i] > 10 and ra[i] < 20:
        catid=1
    elif ra[i] >20 and ra[i] < 30:
        catid=2
    elif ra[i] >30 and ra[i] < 40:
        catid=3
    elif ra[i] >40 and ra[i] < 50:
        catid=4
    elif ra[i] >50 and ra[i] < 60:
        catid=5
    elif ra[i] >300 and ra[i] < 310:
        catid=6
    elif ra[i] >310 and ra[i] < 320:
        catid=7
    elif ra[i] >320 and ra[i] < 330:
        catid=8
    elif ra[i] >330 and ra[i] < 340:
        catid=9
    elif ra[i] >340 and ra[i] < 350:
        catid=10
    elif ra[i] >350 and ra[i] < 360:
        catid=11
    coadd=pf.getdata('/home/jghao/research/data/coadd10_29_09/gmbcg_input_small_'+str(catid)+'.fit')
    if photoz < 0.4:
        rich=GMRrichness(ra[i],dec[i],photoz,coadd)
    elif photoz >= 0.4 and photoz < 0.75:
        rich=RMIrichness(ra[i],dec[i],photoz,coadd)
    elif photoz >= 0.75:
        rich=IMZrichness(ra[i],dec[i],photoz,coadd)
    return rich

