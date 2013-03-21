#---this script measure the bimodal feature around the verita sources. 

import richDr8Iter as rh


def handscanDR8():
    b=rh.pf.getdata('/home/jghao/research/data/redmapper_dr8/dr8_run_redmapper_v5.2_lgt20_catalog.fit')
    for i in range(30,len(b)):
        try:
            rh.getRichness(b.RA[i],b.DEC[i],b.ZRED[i])
        except:
            print '--- Not available ! --'
            continue
        else:
            print 'Lambda_Chisq = '+str(b.LAMBDA_CHISQ[i])
            print 'Zred = '+str(b.ZRED[i])
            raw_input("Press Enter for next cluster...")
            rh.pl.close()

