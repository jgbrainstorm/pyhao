import pyfits as pf
import pylab as pl
import numpy as np

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
    w=1./(xerr**2)
    wm=sum(x*w)/sum(w)
    return(wm)

def wsdo(x,xerr):
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


def bin_scatter_bins(x,y,yerr=None,binedge=None,fmt=None,label=None):
    h=histhao(x,bedge=binedge) 
    nbin=len(h[0])
    xm=np.zeros(nbin)
    ym=np.zeros(nbin)
    sdym=np.zeros(nbin)
    for i in range(0,len(h[0])):
        ind=(x>=h[1][i])*(x<h[1][i+1])
        tt=x[ind]
        if len(tt) > 0:
            xm[i]=np.mean(x[ind])
            if yerr != None:
                ym[i]=wmean(y[ind],yerr[ind])
                sdym[i]=wsd(y[ind],yerr[ind])
            else:
                ym[i]=np.mean(y[ind])
                sdym[i]=np.std(y[ind])/np.sqrt(len(y[ind]))
    if fmt:
        pl.errorbar(xm,ym,yerr=sdym,fmt=fmt)
    else:
        pl.errorbar(xm,ym,yerr=sdym,fmt='ko')
    if label:
        pl.errorbar(xm,ym,yerr=sdym,fmt=fmt,label=label)



def bin_scatter(x,y,yerr=None,binsize=None,fmt=None,label=None,scatter=False):
    h=histhao(x,binsize) 
    nbin=len(h[0])
    xm=np.zeros(nbin)
    ym=np.zeros(nbin)
    sdym=np.zeros(nbin)
    for i in range(0,len(h[0])):
        ind=(x>=h[1][i])*(x<h[1][i+1])
        tt=x[ind]
        if len(tt) > 10:
            xm[i]=np.mean(x[ind])
            if yerr != None:
                ym[i]=wmean(y[ind],yerr[ind])
                sdym[i]=wsd(y[ind],yerr[ind])
            else:
                ym[i]=np.mean(y[ind])
                sdym[i]=np.std(y[ind])/np.sqrt(len(y[ind]))
            if scatter == True:
                sdym[i]=np.std(y[ind])
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
            if len(yerr)>0:
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