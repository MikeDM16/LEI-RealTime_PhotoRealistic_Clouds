import random
import math
from PIL import Image
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import time
import sys
import struct

perm = list(range(256))
random.shuffle(perm)
perm += perm
dirs = [(math.cos(a * 2.0 * math.pi / 256),
         math.sin(a * 2.0 * math.pi / 256),
         math.tan(a * 2.0 * math.pi / 256))
         for a in range(256)]

def noise(x, y, z, per):
    def surflet(gridX, gridY, gridZ):
        distX, distY, distZ = abs(x-gridX), abs(y-gridY), abs(z-gridZ),
        polyX = 1 - 6*distX**5 + 15*distX**4 - 10*distX**3
        polyY = 1 - 6*distY**5 + 15*distY**4 - 10*distY**3
        polyZ = 1 - 6*distZ**5 + 15*distZ**4 - 10*distZ**3
        hashed = perm[perm[perm[int(gridX)%per] + int(gridY)%per] + int(gridZ)%per]
        grad = (x-gridX)*dirs[hashed][0] + (y-gridY)*dirs[hashed][1] + (z-gridZ)*dirs[hashed][2]
        return polyX * polyY * polyZ * grad

    intX, intY, intZ = int(x), int(y), int(z)
    return (surflet(intX+0, intY+0, intZ+0) + surflet(intX+1, intY+0, intZ+0) +
            surflet(intX+0, intY+1, intZ+0) + surflet(intX+1, intY+1, intZ+0) +
            surflet(intX+0, intY+0, intZ+1) + surflet(intX+1, intY+0, intZ+1) +
            surflet(intX+0, intY+1, intZ+1) + surflet(intX+1, intY+1, intZ+1))

def fBm(x, y, z, per, octs):
    val = 0
    for o in range(octs):
        val += 0.5**o * noise(x*2**o, y*2**o, z*2**o, per*2**o)
    return val

size, freq, octs = 128, 1/64.0, 8
data = np.zeros((size, size, size))

with open("vtk_values", "w") as mfile:
    for z in range(size):
        sys.stdout.write("\r%d%%" % (z/size*100))
        sys.stdout.flush()
        for y in range(size):
            for x in range(size):
                mfile.write(str(fBm(x*freq, y*freq, z*freq, int(size*freq), octs))+'\n')
    mfile.close()

#itk_image = sitk.GetImageFromArray(data, isVector=False)
#print("pixel id: {0} ({2})".format(test.GetPixelID(), test.GetPixelIDTypeAsString()))
#test = sitk.Cast(test, sitk.stikFloat32)
#sitk.WriteImage(itk_image, "output.nii", True )
#img = nib.load("output.nii")
#im = Image.new("L", (size, size, size))
#im.putdata(data, 128, 128, 128)
#im.save("noise.png")