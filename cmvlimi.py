#-----------
# this code calculate the magnitude at different redshift to keep the comoving density of galaxy constant. The sample of the galaxy corresponds to an absolute magnitude in i band <= 20.5

import numpy as np
import pyfits as pf
import pylab as pl
import esutil as es
import glob as gl

cosmoV = es.cosmology.Cosmo(h=0.7).V

def limi(x):
    A=np.exp(3.1638)
    k=0.1428
    lmi=A*x**k
    return(lmi)

def comovlim(x):
    polycoeff = np.array([  7.24235186, -29.70849265,  47.82343782, -33.93019772,4.29971516,  10.31565436,  16.54869522])
    lmz=np.polyval(polycoeff,x)
    return(lmz)

#-------------
# the catalog is about 22 deg^2. full sky is 41253 deg^2. So, it is 22/41253 of the full sky. 

truthCat = gl.glob('/home/jghao/research/data/des_mock/v3.04/truthCat/*.fit') 
b=pf.getdata(truthCat[5])

aimag = b.field('amag')[:,2]

ok = aimag <= -20.5
b = b[ok]

z=b.field('z')
zmag = b.field('omag')[:,3]
aimag = b.field('amag')[:,2]

zbin=np.arange(0.05,1.4,0.1)
zmid = (zbin[:-1]+zbin[1:])/2.

n = np.histogram(z,bins = zbin)[0]
comdsty = np.zeros(len(zmid))
limz = np.zeros(len(zmid))


#----target comov density, choose the z = 0.2 - 0.3, abs i mag le -20.5----
zz = (z >= 0.2)*(z<=0.25)*(aimag <= -20.5)

zmagIn = zmag[zz]
targetDsty = len(zmagIn)/cosmoV(0.2,0.3)


for i in range(len(zmid)):
    idx = (z >= zbin[i])*(z <= zbin[i+1])
    zzmag = zmag[idx]
    targetN = round(cosmoV(zbin[i],zbin[i+1])*targetDsty*1.5)
    zzmag = np.sort(zzmag)
    limz[i] = np.median(zzmag[targetN-20:])

b=pf.getdata(truthCat[5])  
pl.plot(b.field('z'),b.field('omag')[:,3],'b,',alpha = 1,label='all galaxies')
pl.plot(z,zmag,'y,',alpha=1,label='galaxy with ABSimag <= -20.5')
pl.plot(z,limi(z),'r,', label='volume limited cut based on color model')
pl.plot(zmid,limz,'b-',label='magnitude cut to keep constant comoving density')
pl.plot(zmid,comovlim(zmid),'r--',label='fitting function for mag cut for const. comoving dsty')
pl.ylim(5,30)
pl.xlabel('redshift')
pl.ylabel('z-band mag')
pl.legend(loc = 'best')
pl.savefig('comov_dsty.png')

#----code used to do the polynomial fit---
res=np.polyfit(zmid,limz,deg=6)
