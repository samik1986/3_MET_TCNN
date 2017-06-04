import numpy as np

trImages = np.load('trainImagesIITM.npy')
trLabel = np.load('trainLabelIITM.npy')
tsImages = np.load('testImagesIITM.npy')
tsLabel = np.load('testLabelIITM.npy')

gxTr = []
gyTr = []
pxTr = []

nTs = np.size(tsImages,0)
n_samples = int(0.05*nTs)
indx = np.random.choice(nTs, n_samples, replace=False)
tgImages = tsImages[indx,:,:,:]
tgLabel = tsLabel[indx,:]
# np.save('targetImagesIITM',tgImages)
# np.save('targetLabelIITM',tgLabel)
print len(trImages)
print len(tgLabel)

count = 0

for i in range(np.size(trImages,0)):
    for j in range(np.size(tgImages,0)):
        tempGx = trImages[i,:,:,:]
        tempGy = trLabel[i,:]
        tempPx = tgImages[j,:,:,:]
        tempPy = tgLabel[j,:]
        if (tempGy == tempPy).all():
            print count
            count = count +1
            gxTr.append(tempGx)
            pxTr.append(tempPx)

            # print gxTr.type()

# np.save('ae_gxTr',gxTr)
# np.save('ae_pxTr',pxTr)

gxTrain = np.asarray(gxTr)
pxTrain = np.asarray(pxTr)
np.save('ae_gxTrain',gxTrain)
np.save('ae_pxTrain',pxTrain)
    # print i, j, count

# np.save('targetImagesIITM',tgImages)
# np.save('targetLabelIITM',tgLabel)