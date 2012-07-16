#! /usr/bin/env python
# This codes is to diagnose the red mapper for stripe 82
import numpy as np
import pyfits as pf
import pylab as pl
import richCoaddIter as rh
import healpy as hp

bg = pf.getdata('/home/jghao/research/data/coadd10_29_09/redmapper/stripe82_bcgs_for_jiangang.fit')
#idx = cat.field('good') == 0
#bg = cat[idx]

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
colnames = ['rich','aic1','aic2','ccolor','alpha0','alpha1','mu0','mu1','sigma0','sigma1']
radius=1.5
for i in range(N):
    print i
    res = rh.getRichness(bg[i].field('ra'),bg[i].field('dec'),bg[i].field('z_lambda'),bcg=True,plot=True,err=True,radius=radius)
    pl.savefig('/home/jghao/research/data/coadd10_29_09/redmapper/fig_aic_rw_err/'+str(i)+'.png')
    pl.close()
    if len(res)==7:        
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

data = [rich, aic1,aic2,ccolor,alpha0,alpha1,mu0,mu1,sigma0,sigma1]
hp.mwrfits('/home/jghao/research/data/coadd10_29_09/redmapper/gmbcg_remeasure_BCG_True_aic_rw_err.fit',data,colnames = colnames)


#------compare------

bg = pf.getdata('/home/jghao/research/data/coadd10_29_09/redmapper/stripe82_bcgs_for_jiangang.fit')
#bcg = pf.getdata('/home/jghao/research/data/coadd10_29_09/redmapper/gmbcg_remeasure_BCG_false_aic_rw_err.fit')
bcg = pf.getdata('/home/jghao/research/data/coadd10_29_09/redmapper/gmbcg_remeasure_BCG_True_aic_rw_err.fit')
idx = ~np.isnan(bcg.mu1)
bg=bg[idx]
bcg=bcg[idx]

ok = (bcg.aic2 -bcg.aic1 < 0)*(bcg.sigma0 <= bcg.sigma1)*(bcg.sigma0 < 0.15)
#bad = bg.field('good') == 0
#good = bg.field('good') == 1

pl.plot(bg.field('lambda'),bcg.rich,'bo',label='Full Sample',alpha = 1)
#pl.plot(bg[ok].field('lambda'),bcg[ok].rich,'r.',label = 'good center lambda')
pl.plot(bg[ok].field('lambda'),bcg[ok].rich,'ro',label = 'Strict Model Definition')


#pl.plot(bg[bad].field('lambda'),bcg[bad].rich,'r.',label = 'bad center lambda')

#pl.plot(bg[ok].field('lambda'),bcg[ok].rich,'r.',label ='aic2 <= aic1',alpha= 1)

pl.xlabel('Lambda')
pl.ylabel('GMBCG Richness')
pl.legend(loc = 'best')
pl.loglog()
pl.ylim(1,200)

#----separation -------
wrongRem = (bcg.ccolor < bcg.mu1)*good
correctRem = (bcg.ccolor < bcg.mu1)*bad
Nwrong=len(bcg[wrongRem])
Nright=len(bcg[correctRem])
Nbad = len(bcg[bad])


pl.hist(bcg[good].ccolor - bcg[good].mu1,bins=20,normed=True,label ='good center',alpha=0.3)
pl.hist(bcg[bad].ccolor - bcg[bad].mu1,bins=20,normed=True,label ='bad center',alpha=0.3)
pl.xlabel('BCG color - Background color')
pl.vlines(0,0,2,linestyle='dashed',color='red')
pl.legend(loc='best')
pl.figtext(0.2,0.8,'correct removal:'+str(Nright)+' out of '+str(Nbad))
pl.figtext(0.2,0.75,'incorrect removal:'+str(Nwrong)+' out of '+str(Nbad))
pl.ylim(0,3)
