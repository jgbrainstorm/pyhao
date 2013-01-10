#! /usr/bin/env python
# This codes is to diagnose the red mapper for stripe 82
import numpy as np
import pyfits as pf
import pylab as pl
import richCoaddIter as rhcoadd
import richDr7Iter as rhdr7
import healpy as hp

bg = pf.getdata('/home/jghao/research/data/sogras/sogras_ra_dec.fits')
N = len(bg)
clusterid = np.arange(N)
rich = np.zeros(N)
aic1 = np.zeros(N)
aic2 = np.zeros(N)
alpha0 = np.zeros(N)
alpha1 = np.zeros(N)
mu0 = np.zeros(N)
mu1 = np.zeros(N)
sigma0 = np.zeros(N)
sigma1 = np.zeros(N)
ccolor = np.zeros(N)
ridgeline_z = np.zeros(N)
ra = np.zeros(N)
dec = np.zeros(N)
photo_z = np.zeros(N)
radius=1.
for i in range(N):
    print i
    if bg[i].field('stripe_82') == True:
        res = rhcoadd.getRichness(float(bg[i].field('ra')),float(bg[i].field('dec')),bg[i].field('photo_z'),bcg=False,plot=True,err=True,radius=radius,rw=True)
        pl.savefig('/home/jghao/research/data/sogras/coadd_'+str(i)+'.png')
        pl.close()
        if len(res)==8:        
            rich[i] = np.round(res[0])
            aic1[i] =res[1]
            aic2[i] = res[2]
            ccolor[i] = res[3]
            alpha0[i] = res[4][0]
            alpha1[i] = res[4][1]
            mu0[i] = res[5][0]
            mu1[i] = res[5][1]
            sigma0[i] = res[6][0]
            sigma1[i] = res[6][1]
            ridgeline_z[i] = res[7]
            ra[i] = bg[i].field('ra')
            dec[i] = bg[i].field('dec')
            photo_z[i] = bg[i].field('photo_z')
data = [clusterid,ra,dec,photo_z,ridgeline_z,rich]
colnames = ['clusterIdx','ra','dec','photo_z','ridgeline_z','richness']
hp.mwrfits('/home/jghao/research/data/sogras/sogras_coadd_new_richness.fit',data,colnames = colnames)

#-----for the dr7 catalog----
bg = pf.getdata('/home/jghao/research/data/sogras/sogras_ra_dec.fits')
idx = (bg.stripe_82 == 84)*bg.photo_z
clusterid = np.arange(len(bg))
stripe82id = clusterid[idx]
N = len(bg)
rich = np.zeros(N)
aic1 = np.zeros(N)
aic2 = np.zeros(N)
alpha0 = np.zeros(N)
alpha1 = np.zeros(N)
mu0 = np.zeros(N)
mu1 = np.zeros(N)
sigma0 = np.zeros(N)
sigma1 = np.zeros(N)
ccolor = np.zeros(N)
ridgeline_z = np.zeros(N)
colnames = ['clusteridx','rich','aic1','aic2','ccolor','alpha0','alpha1','mu0','mu1','sigma0','sigma1','ridgeline_z']
radius=1.
for i in stripe82id:
    print i
    res = rhdr7.getRichness(float(bg[i].field('ra')),float(bg[i].field('dec')),bg[i].field('photo_z'),stripe=82,bcg=False,plot=True,err=True,rw=True,iter=False)
    pl.savefig('/home/jghao/research/data/sogras/dr7_'+str(i)+'.png')
    pl.close()
    if len(res)==8:        
        rich[i] = res[0]
        aic1[i] =res[1]
        aic2[i] = res[2]
        ccolor[i] = res[3]
        alpha0[i] = res[4][0]
        alpha1[i] = res[4][1]
        mu0[i] = res[5][0]
        mu1[i] = res[5][1]
        sigma0[i] = res[6][0]
        sigma1[i] = res[6][1]
        ridgeline_z[i] = res[7]
data = [stripe82id,rich, aic1,aic2,ccolor,alpha0,alpha1,mu0,mu1,sigma0,sigma1,ridgeline_z]
hp.mwrfits('/home/jghao/research/data/sogras/sogras_dr7_new_richness.fit',data,colnames = colnames)


#----measure the dr7 catalog with coadd data in stripe 82

bg = pf.getdata('/home/jghao/homepage/gmbcg_sdss_dr7/gmbcg_sdss_dr7_public_cluster_catalog_stripe.fit')
bg=bg[(bg.field('stripe') == 82)*(bg.field('GM_SCALED_NGALS')>= 15)*(bg.field('photoz')<= 0.43)]
N = len(bg)
clusterid = np.arange(N)
rich = np.zeros(N)
aic1 = np.zeros(N)
aic2 = np.zeros(N)
alpha0 = np.zeros(N)
alpha1 = np.zeros(N)
mu0 = np.zeros(N)
mu1 = np.zeros(N)
sigma0 = np.zeros(N)
sigma1 = np.zeros(N)
ccolor = np.zeros(N)
ridgeline_z = np.zeros(N)
ra = np.zeros(N)
dec = np.zeros(N)
photo_z = np.zeros(N)
dr7rich = np.zeros(N)
radius=1.
for i in range(N):
    print i
    res = rhcoadd.getRichness(float(bg[i].field('ra')),float(bg[i].field('dec')),bg[i].field('photoz'),bcg=False,plot=True,err=True,radius=radius,rw=True)
    pl.savefig('/home/jghao/research/data/sogras/dr7stripe82/coadd_dr7_'+str(i)+'.png')
    pl.close()
    if len(res)==8:        
        rich[i] = np.round(res[0])
        aic1[i] =res[1]
        aic2[i] = res[2]
        ccolor[i] = res[3]
        alpha0[i] = res[4][0]
        alpha1[i] = res[4][1]
        mu0[i] = res[5][0]
        mu1[i] = res[5][1]
        sigma0[i] = res[6][0]
        sigma1[i] = res[6][1]
        ridgeline_z[i] = res[7]
        ra[i] = bg[i].field('ra')
        dec[i] = bg[i].field('dec')
        photo_z[i] = bg[i].field('photoz')
        dr7rich[i] = bg[i].field('GM_SCALED_NGALS')
data = [clusterid,ra,dec,photo_z,ridgeline_z,rich,dr7rich,aic1,aic2]
colnames = ['clusterIdx','ra','dec','photo_z','ridgeline_z','richness','dr7richness','aic1','aic2']
hp.mwrfits('/home/jghao/research/data/sogras/dr7stripe87_coadd_new_richness.fit',data,colnames = colnames)


#---analyze the scaling relation --
import binplot as bp
b=pf.getdata('/home/jghao/research/data/sogras/dr7stripe87_coadd_new_richness.fit')
ok = (b.aic1 -b.aic2 > 10)*(b.ridgeline_z > 0.1)
bb = b[ok]
pl.figure(figsize=(8,8))
pl.plot(bb.dr7richness,bb.richness,'b.',alpha=0.7)
x,y,yerr=bp.bin_scatter_bins(bb.dr7richness,bb.richness,binedge=[10.,20.,30.,40.,60,80.,120.],fmt='ro')
pl.xlabel('DR7 GM_SCALED_Ngals')
pl.ylabel('New Coadd Richness')
pl.xlim(0,150)
pl.ylim(0,150)
pl.grid()
res = linefit(x,y)
pl.plot(np.arange(140),res[0]+res[1]*np.arange(140),'g-',lw=2)
pl.figtext(0.6,0.8,'Slope: '+str(round(res[1],3)))
pl.figtext(0.6,0.77,'Intercept: '+str(round(res[0],3)))
pl.savefig('/home/jghao/research/data/sogras/dr7richness_newcoaddrichness_scaling.png')
