#---this script measure the bimodal feature around the verita sources. 

import pandas as pd
import richDr8Iter as rh


def handscanDR8():
    b=pd.read_csv('/home/jghao/research/data/veritas/veritas_unique.csv')
    ra = b['ra']
    dec = b['dec']
    ID = b['ID']
    for i in range(0,len(b)):
        print '-----'+str(i)+'-----'
        try:
            t=rh.getRichness(ra[i],dec[i],0.1)
        except:
            print '--- Not available ! --'
            continue
        else:
            if len(t) > 30 or len(t) == 3:
                continue
            else:
                print 'VERITAS ID: '+ID[i]
                rh.pl.figtext(0.61,0.92,'VERITAS ID: '+ID[i])
                raw_input("Press Enter for next cluster...")
                rh.pl.close()

