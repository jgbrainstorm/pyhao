
import pylab as pl
import numpy as np
import scipy.signal as sg
import os
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
import pandas.rpy.common as com

MASS = importr('MASS')

def display():
    pl.savefig('temp.png')
    os.system('display temp.png')
    return 0

def remOutlier(x):
    y=x
    n = len(x)
    y.sort()
    ind_qt1 = round((n+1)/4.)
    ind_qt3 = round((n+1)*3/4.)
    IQR = y[ind_qt3]- y[ind_qt1]
    lowFense = y[ind_qt1] - 1.5*IQR
    highFense = y[ind_qt3] + 1.5*IQR
    ok = (y>lowFense)*(y<highFense)
    indx=np.arange(n)
    return indx[ok]


def robust_mean(x):
    y = x.flatten()
    if len(np.unique(y))==1:
        meany=y[0]
    else:
        n = len(y)
        y.sort()
        ind_qt1 = round((n+1)/4.)
        ind_qt3 = round((n+1)*3/4.)
        IQR = y[ind_qt3]- y[ind_qt1]
        lowFense = y[ind_qt1] - 1.5*IQR
        highFense = y[ind_qt3] + 1.5*IQR
        if lowFense == highFense:
            meany=lowFense
        else:
            ok = (y>lowFense)*(y<highFense)
            yy=y[ok]
            meany=yy.mean(dtype='double')
    return meany


#-------Robust Standard Deviation---

def robust_std(x):
    y = x.flatten()
    if len(np.unique(y))==0:
        stdy=0.
    else:
        n = len(y)
        y.sort()
        ind_qt1 = round((n+1)/4.)
        ind_qt3 = round((n+1)*3/4.)
        IQR = y[ind_qt3]- y[ind_qt1]
        lowFense = y[ind_qt1] - 1.5*IQR
        highFense = y[ind_qt3] + 1.5*IQR
        if lowFense == highFense:
            stdy=lowFense
        else:
            ok = (y>lowFense)*(y<highFense)
            yy=y[ok]
            stdy=yy.std(dtype='double')
    return stdy

#-------Robust Standard Deviation of the mean---

def robust_stdm(x):
    y = x.flatten()
    if len(np.unique(y))==0:
        stdy=0.
    else:
        n = len(y)
        y.sort()
        ind_qt1 = round((n+1)/4.)
        ind_qt3 = round((n+1)*3/4.)
        IQR = y[ind_qt3]- y[ind_qt1]
        lowFense = y[ind_qt1] - 1.5*IQR
        highFense = y[ind_qt3] + 1.5*IQR
        if lowFense == highFense:
            stdym=lowFense
        else:
            ok = (y>lowFense)*(y<highFense)
            yy=y[ok]
            stdym=yy.std(dtype='double')/np.sqrt(len(yy))
    return stdym



def wmean(x,xerr):
    ok=remOutlier(xerr)
    x=x[ok]
    xerr=xerr[ok]
    #ok=remOutlier(x)
    #x=x[ok]
    #xerr=xerr[ok]
    w=1./(xerr**2)
    wm=sum(x*w)/sum(w)
    return(wm)

def wsd(x,xerr):
    ok=remOutlier(xerr)
    x=x[ok]
    xerr=xerr[ok]
    #ok=remOutlier(x)
    #x=x[ok]
    #xerr=xerr[ok]
    w=1./(xerr**2)
    ws=np.sqrt(1./sum(w))
    return(ws)


def wmeano(x,xerr):
    ok=xerr > 0
    x=x[ok]
    xerr=xerr[ok]
    w=1./(xerr**2)
    wm=sum(x*w)/sum(w)
    return(wm)

def wsdo(x,xerr):
    ok=xerr > 0
    x=x[ok]
    xerr=xerr[ok]
    w=1./(xerr**2)
    ws=np.sqrt(1./sum(w))
    return(ws)

def histhao(x,bsize=None,bedge=None):
    if bedge is not None:
        d=np.histogram(x,bins=bedge)
    else:
        nbin=(np.max(x)-np.min(x))/bsize
        d=np.histogram(x,bins=nbin)
    return d

def logbin_edge(x=None,nbins=None,xrange=None):
    if xrange is None:
        rng = np.log10(x.max()) - np.log10(x.min())
        step = rng / float(nbins)
        xedge = 10**np.arange(np.log10(x.min()),np.log10(x.max()),step)
    else:
        rng = np.log10(xrange[1]) - np.log10(xrange[0])
        step = rng / float(nbins)
        xedge = 10**np.arange(np.log10(xrange[0]),np.log10(xrange[1]),step)
    if xedge[-1] == xrange[1]:
        res = xedge
    else:
        res = np.append(xedge,xrange[1])
    return res


def bin_scatter_bins(x,y,yerr=None,binedge=None,fmt=None,label=None,axes=None,alpha=None,plot=True,scatter=False,robust=False):
    h=histhao(x,bedge=binedge) 
    nbin=len(h[0])
    xm=np.zeros(nbin)
    ym=np.zeros(nbin)
    sdym=np.zeros(nbin)
    for i in range(0,len(h[0])):
        ind=(x>=h[1][i])*(x<h[1][i+1])
        tt=x[ind]
        if len(tt) > 2:
            xm[i]=np.mean(x[ind])
            if yerr != None:
                ym[i]=wmeano(y[ind],yerr[ind])
                sdym[i]=wsdo(y[ind],yerr[ind])
            else:
                ym[i]=np.mean(y[ind])
                sdym[i]=np.std(y[ind])/np.sqrt(len(y[ind]))
            if scatter == True:
                if robust == False:
                    sdym[i]=np.std(y[ind])
                else:
                    ym[i]=robust_mean(y[ind])
                    sdym[i]=robust_std(y[ind])
    if plot == True:
        if axes is not None:
            if fmt:
                axes.errorbar(xm,ym,yerr=sdym,fmt=fmt,alpha=alpha)
            else:
                axes.errorbar(xm,ym,yerr=sdym,fmt='ko',alpha=alpha)
            if label:
                axes.errorbar(xm,ym,yerr=sdym,fmt=fmt,label=label,alpha=alpha)
        else:
            if fmt:
                pl.errorbar(xm,ym,yerr=sdym,fmt=fmt,alpha=alpha)
            else:
                pl.errorbar(xm,ym,yerr=sdym,fmt='ko',alpha=alpha)
            if label:
                pl.errorbar(xm,ym,yerr=sdym,fmt=fmt,label=label,alpha=alpha)
    return xm,ym,sdym



def bin_scatter(x,y,yerr=None,binsize=None,nbins=None,fmt=None,label=None,scatter=False,axes=None,plot=True,robust=False):
    if nbins != None:
        binsize = (np.max(x) - np.min(x))/float(nbins)
    h=histhao(x,binsize) 
    nbin=len(h[0])
    xm=np.zeros(nbin)
    ym=np.zeros(nbin)
    sdym=np.zeros(nbin)
    for i in range(0,len(h[0])):
        ind=(x>=h[1][i])*(x<h[1][i+1])
        tt=x[ind]
        if len(tt) > 2:
            xm[i]=np.mean(x[ind])
            if yerr != None:
                ym[i]=wmeano(y[ind],yerr[ind])
                sdym[i]=wsdo(y[ind],yerr[ind])
            else:
                ym[i]=np.mean(y[ind])
                sdym[i]=np.std(y[ind])/np.sqrt(len(y[ind]))
            if scatter == True:
                if robust == False:
                    sdym[i]=np.std(y[ind])
                else:
                    ym[i]=robust_mean(y[ind])
                    sdym[i]=robust_std(y[ind])
    if plot == True:
        if axes is not None:
            if fmt:
                axes.errorbar(xm,ym,yerr=sdym,fmt=fmt)
            else:
                axes.errorbar(xm,ym,yerr=sdym,fmt='ko')
            if label:
                axes.errorbar(xm,ym,yerr=sdym,fmt=fmt,label=label)
        else:
            if fmt:
                pl.errorbar(xm,ym,yerr=sdym,fmt=fmt)
            else:
                pl.errorbar(xm,ym,yerr=sdym,fmt='ko')
            if label:
                pl.errorbar(xm,ym,yerr=sdym,fmt=fmt,label=label)
    return xm,ym,sdym


def bin_hist(x,y,yerr=None,bins=None):
    h=np.histogram(x,bins) 
    xm=np.zeros(bins)
    sdxm=np.zeros(bins)
    ym=np.zeros(bins)
    sdym=np.zeros(bins)

    fig=pl.figure(figsize=(10,10))
    for i in range(0,len(h[0])):
        ind=(x>=h[1][i])*(x<h[1][i+1])
        tt=x[ind]
        if len(tt) > 0:
            if yerr != None:
                ym[i]=wmean(y[ind],yerr[ind])
                sdym[i]=wsd(y[ind],yerr[ind])
            else:
                ym[i]=np.mean(y[ind])
                sdym[i]=np.std(y[ind])/np.sqrt(len(y[ind]))
            ax=fig.add_subplot(3,3,i+1)
            pl.hist(y[ind],bins=30)
            pl.text(0.1,0.9,'#: '+str(len(y[ind])),transform = ax.transAxes)
            pl.text(0.1,0.8,'Mean: '+str(round(ym[i],5)),transform = ax.transAxes)
            pl.text(0.1,0.7,'Sd_M: '+str(round(sdym[i],5)),transform = ax.transAxes)
            pl.title('Bins: '+str(round(h[1][i],2))+'-'+str(round(h[1][i+1],2)))



def bin_hist_logx(x,y,nbins=None,xrange=None):
    if xrange is None:
        xrange=[x.min(),x.max()]
    binedge = logbin_edge(x,nbins,xrange=xrange)
    for i in range(nbins):
        pl.figure()
        ind = (x >= binedge[i])*(x <= binedge[i+1])
        pl.hist(y[ind],bins=30)
        pl.title('bin: '+str(i))
        pl.figtext(0.5,0.8,'Mean: '+str(y[ind].mean()))
        pl.figtext(0.5,0.75,'Std: '+str(y[ind].std()))
    return 0

    

def binboxplot(x,y,binsize=None):
    h=histhao(x,binsize) 
    nbin=len(h[0])
    data=[]
    xm=[]
    for i in range(0,nbin):
        ind=(x>=h[1][i])*(x<h[1][i+1])
        tt=x[ind]
        if len(tt) > 2:
            xm.append(np.median(x[ind]))
            data.append(y[ind])
    pl.boxplot(data)
    pl.xticks(np.arange(1,nbin+1),np.round(xm,2))
    return 0


def dsty(x,y,bins=None,range=None,normed=False,smooth=None,levels=None,format='%.3f'):
    h,xx,yy=np.histogram2d(x,y,bins=bins,range=range,normed=normed)
    xx=(xx[0:-1]+xx[1:])/2.
    yy=(yy[0:-1]+yy[1:])/2.
    pl.contourf(xx,yy,h.T,levels=levels)
    pl.colorbar(format=format)
    return(0)

def kde2d(x, y, bins=10, levels = None, format = '%.3f'):
        dsty = MASS.kde2d(x,y,n=bins,h=np.array([MASS.bandwidth_nrd(x),MASS.bandwidth_nrd(y)]))
        dsty2 = np.array(dsty[2])
        xx = np.array(dsty[0])
        yy = np.array(dsty[1])
        pl.contourf(xx,yy,dsty2.T,levels = None)
        pl.colorbar(format=format)
        return dsty2



def bin_scatter_logx(x,y,yerr=None,nbins=None,xrange=None,fmt=None,label=None,axes=None,alpha=None,plot=True,robust=False):
    binedge = logbin_edge(x,nbins,xrange=xrange)
    xm,ym,sdym=bin_scatter_bins(x,y,yerr=yerr,binedge=binedge,fmt=fmt,label=label,axes=axes,alpha=alpha,plot=plot,robust=robust)
    return xm,ym,sdym

