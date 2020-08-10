# Written by Metison Wood <wubqmail(at)163.com> under DWYW license
# To analyze the Chemical Short Range Order

# Python standard libraries
# {{{
from __future__ import (absolute_import, print_function, division,
        unicode_literals)
import os,sys
import numpy as np
import pandas as pd
# }}}

def csro(datas=None, style='CS', outfile=True, outpath=''):
# {{{
    """A function to analyze the CSRO.
    
    Parameters
    ----------
    datas: pandas.DataFrame
        atom_id(int), atype(int), Neibs(str)
    style: str, optional.
        'WC': Warren-Cowley
        'CS': Cargill-Spaepen, (default)
        ref: Y.Q. Cheng, Progress in Material Science 56 (2011) 379-473
    outfile: bool, optional
        whether output the results to file.
    outpath: str, optional
        the path to output, default(current workdir)

    Returns
    -------
    pandas.DataFrame
        atom_id, atype, csro(A), csro(B)
    files:
         1   2
      1 1-1 1-2
      2 2-1 2-2

    Note: only binary is supported.
    """
    # Get the initial paras
    CNs = []
    for i in datas['Neibs']:
        neibs = [int(i) for i in i.split(' ')]
        cnall = len(neibs)
        tmp_types = datas['atype'].iloc[np.array(neibs, dtype=np.int32)-1].value_counts()
        if len(tmp_types) == 1:
            if 1 in tmp_types.keys():
                cnA = tmp_types[1]
                cnB = 0
            elif 2 in tmp_types.keys():
                cnA = 0
                cnB = tmp_types[2]
        elif len(tmp_types) == 2:
            cnA = tmp_types[1]
            cnB = tmp_types[2]
        CNs.append([cnall, cnA, cnB])
    CNs = np.array(CNs, dtype=np.int32)
    ZA = np.mean(CNs[:,0][datas['atype'] == 1])
    ZB = np.mean(CNs[:,0][datas['atype'] == 2])
    xA = len(CNs[:,0][datas['atype'] == 1])/len(CNs)
    xB = 1-xA
    Z = ZA*xA + ZB*xB
    # For CS style
    ZAA = xA*ZA*ZA/Z
    ZAB = xB*ZA*ZB/Z
    ZBA = xA*ZB*ZA/Z
    ZBB = xB*ZB*ZB/Z
    nmax = xB*ZB/(xA*ZA) if xB*ZB < xA*ZA else xA*ZA/(xB*ZB)
    # Analyze the csro
    csro = []
    for i in zip(datas['atype'], CNs):
        if i[0] == 1:
            if style == 'WC':
                csroaa = 1 - i[1][1]/(xA*ZA)
                csroab = 1 - i[1][2]/(xB*ZA)
                csro.append([csroaa, csroab])
            if style == 'CS':
                csroaa = i[1][1]/ZAA-1
                csroab = (i[1][2]/ZAB-1)/nmax
                csro.append([csroaa, csroab])
        if i[0] == 2:
            if style == 'WC':
                csroba = 1 - i[1][1]/(xA*ZB)
                csrobb = 1 - i[1][2]/(xB*ZB)
                csro.append([csroba, csrobb])
            if style == 'CS':
                csroba = (i[1][1]/ZBA-1)/nmax
                csrobb = i[1][2]/ZBB-1
                csro.append([csroba, csrobb])
    csro = np.array(csro)
    datas['csroA'] = csro[:, 0]
    datas['csroB'] = csro[:, 1]
    results = np.zeros((2, 2))
    if outfile:
        results[0, 0] = np.mean(datas['csroA'][datas['atype'] == 1])
        results[0, 1] = np.mean(datas['csroB'][datas['atype'] == 1])
        results[1, 0] = np.mean(datas['csroA'][datas['atype'] == 2])
        results[1, 1] = np.mean(datas['csroB'][datas['atype'] == 2])
        np.savetxt(os.path.join(outpath, 'csro.dat'), results, fmt='%.6f')
    return datas[['atom_id', 'atype', 'csroA', 'csroB']]
# }}}

if __name__ == "__main__":
    import mdAna as ma
    frame = ma.Frame('test.data')
    timestep, systems = frame.read_single()
    voro = ma.Voronoi(systems=systems, symbols=['Cu'])
    res = voro.revoro()
    datas = pd.DataFrame({'atom_id':res.atoms.atom_id,
        'atype':res.atoms.atype, 'Neibs':res.atoms.voroNeib})
    results = csro(datas=datas, style='CS')
    #print(results)
