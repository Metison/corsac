# Written by Metison Wood <wubqmail(at)163.com> under DWYW license
# For creating a color list of RGB

# Standard Python libraries
# {{{
import os, sys
import math
# }}}

def colorgrad2(colorA=[0, 0, 0], colorB=[255, 255, 255], num=10):
# {{{
    """Creating a gradient from two colors.
    
    Parameters
    ----------
    colorA: a list of int
        RGB values (0~255).
    colorB: a list of int
        RGB values (0~255).
    num: int
        the number of datas.
    
    Returns
    -------
    results: a list of RGB values.
    """
    # Check args
    for i in colorA:
        if not isinstance(i, int) or i > 255 or i < 0:
            raise ValueError('colorA value is not available')
            sys.exit(1)
    for i in colorB:
        if not isinstance(i, int) or i > 255 or i < 0:
            raise ValueError('colorA value is not available')
            sys.exit(1)
    results = []
    for i in range(num):
        R = colorA[0] + (colorB[0] - colorA[0])*i/(num-1)
        R = math.ceil(R)
        G = colorA[1] + (colorB[1] - colorA[1])*i/(num-1)
        G = math.ceil(G)
        B = colorA[2] + (colorB[2] - colorA[2])*i/(num-1)
        B = math.ceil(B)
        results.append([R, G, B])
    return results
# }}}
def colorrainbow(num=6):
# {{{
    """Creating a color gradient of rainbow.
    
    Parameters
    ----------
    num: int
        the number of datas.
    
    Returns
    -------
    results: a list of RGB values.
    """
    results = []
    numinc = num/6
    for i in range(num+1):
        if i <= numinc:
            R = 255
            G = math.ceil(0 + 152*i/numinc)
            B = 0
        elif i > numinc and i <= 2*numinc:
            R = 255
            G = math.ceil(152 + 103*(i-1*numinc)/numinc)
            B = 0
        elif i > 2*numinc and i <= 3*numinc:
            R = math.ceil(255 - 255*(i-2*numinc)/numinc)
            G = 255
            B = 0
        elif i > 3*numinc and i <= 4*numinc:
            R = 0
            G = 255
            B = math.ceil(0 + 255*(i-3*numinc)/numinc)
        elif i > 4*numinc and i <= 5*numinc:
            R = 0
            G = math.ceil(255 - 255*(i-4*numinc)/numinc)
            B = 255
        else:
            R = math.ceil(0 + 150*(i-5*numinc)/numinc)
            G = 0
            B = 255
        results.append([R, G, B])
    return results
# }}}

if __name__ == "__main__":
    results = colorgrad2([1, 1, 0], [255, 254, 253])
    #results = colorrainbow(num=24)
    print(len(results))
    print(results)
