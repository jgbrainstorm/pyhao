# this codes get the healpix conversion 
import healpy as hp, pyfits as pf, esutil as es, pylab as pl, glob as gl, numpy as np
import os.path

     

#----slice data -----
n = 56560278/6
for i in range(6):
    print i
    hdu = pf.open('/media/jghao/data/sdssdr8/dr8_galaxies_clean_v7.fit')
    hdu[1].data = hdu[1].data[i*n:(i+1)*n]
    hdu.writeto('/media/jghao/data/sdssdr8/dr8_galaxies_clean_v7_patch_'+str(i)+'.fit')
    hdu = 0


#----------make new files ------

f = gl.glob('/media/jghao/data/sdssdr8/patches/*.fit')
nf = len(f)

for i in range(0,nf):
    print i
    ff = pf.getdata(f[i])
    ra=ff.RA
    dec=ff.DEC
    l,b = es.coords.eq2gal(ra, dec)
    theta=(90-b)/180.*3.14159265  #Galactic coordinate
    phi=l/180.*3.14159265
    nside=2**5
    hid=hp.ang2pix(nside,theta,phi)
    unqid = np.unique(hid)
    nid = len(unqid)
    for j in range(nid):
        ok = hid == unqid[j]
        if np.any(ok):
            fname = '/media/jghao/data/sdssdr8/patches/hpixFile/'+str(unqid[j])+'.fit'
            if os.path.isfile(fname):
                fname = fname[:-4]+'_'+str(i)+'.fit'      
            hdu = pf.BinTableHDU(ff[ok])
            hdu.writeto(fname)
            
#---------merge those multiple ones ------------

f = gl.glob('/media/jghao/data/sdssdr8/patches/hpixFile/*_?.fit')
idx = []
for fname in f:
    idx_start=fname.find('hpixFile/')+9
    idx_end = fname.find('_')
    idx.append(int(fname[idx_start:idx_end]))

idx = np.array(idx)
unqidx = np.unique(idx)

for j in unqidx:
    ffo = gl.glob('/media/jghao/data/sdssdr8/patches/hpixFile/'+str(j)+'.fit')[0]
    ffn = gl.glob('/media/jghao/data/sdssdr8/patches/hpixFile/'+str(j)+'_?.fit')
    nfile = len(ffn)
    t1 = pf.open(ffo)
    nrows = []
    nrows.append(t1[1].data.shape[0])
    print j
    for k in range(nfile):
        nrows.append(pf.getdata(ffn[k]).shape[0])
    nrows = np.array(nrows)
    hdu = pf.new_table(t1[1].columns,nrows=nrows.sum())
    for k in range(nfile):
        tb = pf.open(ffn[k])
        for l in range(len(t1[1].columns)):
            hdu.data.field(l)[nrows[k]:nrows[k]+nrows[k+1]] = tb[1].data.field(l)
    hdu.writeto(ffo[:-4]+'.fit',clobber=True)


#------------origanize the data files --------

ff=gl.glob('/media/jghao/data/sdssdr8/patches/hpixFile/*.fit')

idx = []
for fname in ff:
    idx_start=fname.find('hpixFile/')+9
    idx_end = fname.find('.')
    idx.append(int(fname[idx_start:idx_end]))
idx = np.array(idx)

