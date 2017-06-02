import os
import numpy as np
import Image
from scipy import misc
import glob

dsDir = '/media/vplab/Samik_Work/Samik/Datasets/FR_SURV_VID_pr/FR_SURV_VID_Faces/'

r = []
rLabel = []
t = []
tLabel = []

count = 1

for directories in os.listdir(dsDir):
    dir = os.path.join(dsDir,directories)
    trdir = os.path.join(dir,'Indoor/')
    tsdir = os.path.join(dir, 'Outdoor/')
    for shots in os.listdir(trdir):
        shotDir = os.path.join(trdir,shots)
        for files in os.listdir(shotDir):
            filepath = os.path.join(shotDir,files)
            img = misc.imread(filepath)
            img = misc.imresize(img, [100,100,3],'bicubic')
            r.append(img)
            rLabel.append(count)
            trImages = np.asarray(r)
            trLabel = np.asarray(rLabel)
    for shots in os.listdir(tsdir):
        shotDir = os.path.join(tsdir,shots)
        for files in os.listdir(shotDir):
            filepath = os.path.join(shotDir,files)
            img = misc.imread(filepath)
            img = misc.imresize(img, [100,100,3],'bicubic')
            t.append(img)
            tLabel.append(count)
            tsImages = np.asarray(t)
            tsLabel = np.asarray(tLabel)
    count = count+1
np.save('trainImagesIITM',trImages)
np.save('trainLabelIITM',trLabel)
np.save('testImagesIITM',tsImages)
np.save('testLabelIITM',tsLabel)

