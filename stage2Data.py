import numpy as np

trImages = np.load('trainImagesIITM.npy')
trLabel = np.load('trainLabelIITM.npy')
tsImages = np.load('testImagesIITM.npy')
tsLabel = np.load('testLabelIITM.npy')

gxTr = []
gyTr = []
pxTr = []

nTs = np.size(tsImages,0)
n_samples = int(0.2*nTs)
np.random.choice(nTs, n_samples, replace=False)
tgImages = tsImages[n_samples,:,:,:]
tgLabel = tsLabel[n_samples,:]

for i in range(np.size(trImages,0)):
    for j in range(np.size(tgImages,0)):
        tempGx = trImages[i,:,:,:]
        tempGy = trLabel[i]
        tempPx = tgImages[j,:,:,:]
        tempPy = tgLabel[j]
        if trLabel[i] == tgLabel[j]:
            gxTr.append(tempGx)
            pxTr.append(tempPx)
gxTrain = np.asarray(gxTr)
pxTrain = np.asarray(pxTr)


np.save('ae_gxTrain',gxTrain)
np.save('ae_pxTrain',pxTrain)

np.save('targetImagesIITM',tgImages)
np.save('targetLabelIITM',tgLabel)