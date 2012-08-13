#! /usr/bin/env python

import glob as gl
import os,sys
import pyfits as pf
import numpy as np
import esutil as es

Da=es.cosmology.Cosmo(h=0.7).Da

def circle_string(ra=None,dec=None,radius=None,z=None):
    ang_radius=np.rad2deg(radius/Da(0.,z))
    ndivide = 361
    ang = np.arange(ndivide)
    ang = np.deg2rad(ang)
    circle_str=str(ra+ang_radius*np.cos(ang[0])-180)+','+str(dec+ang_radius*np.sin(ang[0]))+','+'0 '
    for i in range(1,ndivide):
        seg_str = str(ra+ang_radius*np.cos(ang[i])-180)+','+str(dec+ang_radius*np.sin(ang[i]))+','+'0 '
        circle_str = circle_str+seg_str
    return circle_str



def makeKML(fileName=None,clusterID=None,ra=None, dec=None,z=None,richness = None):
    num=len(ra)
    file=open(fileName,'w')
    file.write('<kml xmlns="http://earth.google.com/kml/2.2" hint="target=sky">\n')
    file.write('<Document>\n')
    file.write('<Style id="GalaxyCluster">\n')
    file.write('<BalloonStyle>\n')
    file.write('<text><center><b>$[name]</b></center><br/>$[description]</text>\n')
    file.write('</BalloonStyle>\n')
    file.write('</Style>\n')
    for i in range(num):
        file.write('<Folder>\n')
        file.write('<Placemark> \n')
        #file.write('<name> Cluster:'+str(z[i])+'</name>\n')
        file.write('<description>\n')
        file.write('<![CDATA[ \n')
        file.write('<ul>\n')
        if clusterID != None:
            file.write('<li> <b> ClusterID: '+str(clusterID[i])+'</b>\n')
        if richness != None:
            file.write('<li> <b> Richness: '+str(richness[i])+'</b>\n')
        #file.write('<li> <b> <a href="http://cas.sdss.org/dr7/en/tools/chart/navi.asp?ra='str(ra[i])+'&dec='+str(dec[i])+'&scale=0.800000&width=919&height=919&opt=GS&query=" target="frame2"> SDSS Image </a></b>\n')  
        file.write('</ul>\n')
        file.write('KMZ by: J. Hao @ Fermilab, 2012 ]]> \n')
        file.write('</description> \n')
        file.write(' <LookAt> \n')
        file.write(' <longitude>'+str(ra[i]-180.)+'</longitude> \n')
        file.write(' <latitude>'+str(dec[i])+'</latitude> \n')
        file.write(' <altitude>0</altitude> \n')
        file.write(' <range>0</range> \n')
        file.write(' <tilt>0</tilt> \n')
        file.write(' <heading>0</heading> \n')
        file.write(' </LookAt> \n')
        file.write(' <Point> \n')
        file.write(' <coordinates>'+str(ra[i]-180.)+','+str(dec[i])+','+'0'+'</coordinates> \n')
        file.write(' </Point> \n')
        file.write(' </Placemark>\n')
        file.write(' <Placemark>\n')
        file.write('<name> Scaled Radius Circle</name>\n')
        file.write(' <LookAt>\n')
        file.write(' <longitude>'+str(ra[i]-180.)+'</longitude>\n')
        file.write(' <latitude>'+str(dec[i])+'</latitude>\n')
        file.write(' <altitude>0</altitude>\n')
        file.write(' <range>0</range>\n')
        file.write(' <tilt>0</tilt>\n')
        file.write(' <heading>0</heading>\n')
        file.write(' </LookAt>\n')
        file.write(' <LinearRing>\n')
        file.write('<altitudeMode>clampToGround</altitudeMode> \n')
        file.write(' <coordinates>\n')
        file.write(circle_string(ra[i],dec[i],1.,z[i])+'\n')
        file.write('</coordinates>\n')
        file.write('</LinearRing>\n')
        file.write('</Placemark>\n')
        file.write(' </Folder>\n')
    file.write('</Document> \n')
    file.write('</kml> \n')
    file.close()
    return 0
        
if __name__ == '__main__':
    #------code ------------
    """
    bg = pf.getdata('/home/jghao/research/data/coadd10_29_09/redmapper/stripe82_bcgs_for_jiangang.fit')

    ra = bg.field('ra')
    dec = bg.field('dec')
    z = bg.field('z_lambda')
    richness = bg.field('lambda')
    filename ='coadd_lambda.kml'
    clusterid = np.arange(len(ra))
    ast.makeKML(fileName=filename,clusterID=clusterid,ra=ra, dec=dec,z=z,richness = richness)
    """
