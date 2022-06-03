import torch
import numpy as np
import matplotlib.pyplot as plt
import ConvolutionPerfusionPy as cPerfusion

def IniNetwork(inputChan, frameNum, dataNum):
    print('Ini')
    return cPerfusion.TrainMyNetwork(inputChan, frameNum, dataNum)

def CalcBF(myNet, timeFunc, LSF, target):
    print('Train')

    tFunc = torch.Tensor(timeFunc)
    trainLSF = torch.Tensor(LSF)
    trainTARGET = torch.Tensor(target)

    tFunc = tFunc.to(myNet.model.device)
    trainLSF = trainLSF.to(myNet.model.device)
    trainTARGET = trainTARGET.to(myNet.model.device)

    tFunc = tFunc.repeat((myNet.inputChan, myNet.dataNum, 1))

    impulseResponse = myNet.CalcImpulseResponse(trainLSF, trainTARGET)

    BF, BV, MTT = myNet.CalcBloodFlow(tFunc, impulseResponse)

    bf  = BF.to('cpu').detach().numpy().copy()
    bv  = BV.to('cpu').detach().numpy().copy()
    mtt = MTT.to('cpu').detach().numpy().copy()
    IR = impulseResponse.to('cpu').detach().numpy().copy()

    IR = np.ravel(IR)
    return IR, bf, bv, mtt



def sampleCalFromImg():

    import cv2
    import glob

    files = glob.glob("sampleData/*.tif")

    img0 = cv2.imread(files[0], cv2.IMREAD_ANYDEPTH)


    lsfData = np.loadtxt('sampleData/LSFs.csv', delimiter=',')
    lsfData = lsfData.T

    timeFunc = lsfData[0]
    LSF = lsfData[1:]
    LSF = np.expand_dims(LSF, 0)
    inputChan = LSF.shape[1]
    frameNum = LSF.shape[2]
    dataNum = img0.size

    myNet = IniNetwork(inputChan, frameNum, dataNum)

    target = np.zeros((len(files), dataNum))

    for i, file in enumerate(files):
        img = cv2.imread(file, cv2.IMREAD_ANYDEPTH)

        img = cv2.boxFilter(img, 0, (3,3))

        arr = np.asarray(img)
        target[i] = np.ravel(arr)

    target = target.T
    target = np.expand_dims(target, 0)

    impulseResponse, bf, bv, mtt = CalcBF(myNet, timeFunc, LSF, target)

    torch.save(myNet.model.state_dict(), './model.pth')
    
    train_history = myNet.train_history
    plt.plot(range(len(train_history)), train_history, marker='.', label='loss (Training data)')
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.set_yscale('log')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss.png')

    if 1:
        numBFprms = 3 # bf, bv, mtt
        for i in range(inputChan):
            bfImg = bf[i,:]
            bfImg = bfImg.reshape(img0.shape)
            bfImg = bfImg.astype(np.uint8)
            plt.subplot(numBFprms, inputChan, 0 * inputChan + i + 1)
            plt.imshow(bfImg, cmap = "jet")
            plt.colorbar()

            bvImg = bv[i,:]
            bvImg = bvImg.reshape(img0.shape)
            bvImg = bvImg.astype(np.uint8)
            plt.subplot(numBFprms, inputChan, 1 * inputChan + i + 1)
            plt.imshow(bvImg, cmap = "jet")
            plt.colorbar()

            mttImg = mtt[i,:]
            mttImg = mttImg.reshape(img0.shape)
            mttImg = mttImg.astype(np.uint8)
            plt.subplot(numBFprms, inputChan, 2 * inputChan + i + 1)
            plt.imshow(mttImg, cmap = "jet")
            plt.colorbar()
        plt.show()


    if 1:
        trainLSF = torch.Tensor(LSF)
        trainLSF = trainLSF.to(myNet.model.device)
        trainLSF = torch.sub(trainLSF, torch.unsqueeze(torch.min(trainLSF, 2)[0], 2))

        tgt = torch.Tensor(target)
        tgt = tgt.to(myNet.model.device)
        tgt = torch.sub(tgt, torch.unsqueeze(torch.min(tgt, 2)[0], 2))

        trainLSF = trainLSF.repeat(1, dataNum, 1)
        predTARGET = myNet.model(trainLSF)
        predTARGET = predTARGET[:, :, :frameNum] 

        tLSF = trainLSF.to('cpu').detach().numpy().copy()

        x = 50
        y = 70
        n = img0.shape[0] * y + x
        predT = predTARGET.to('cpu').detach().numpy().copy()
        tg = tgt.to('cpu').detach().numpy().copy()
        plt.plot(timeFunc, tg[0][n], marker="o",linestyle='None')
        plt.plot(timeFunc, predT[0][n])

        for i in range(inputChan):
            pr = impulseResponse[n * inputChan * frameNum + i * frameNum : n * inputChan * frameNum + (i + 1) * frameNum]
            con = np.convolve(pr, tLSF[0][i])
            con = con[:frameNum]
            plt.plot(timeFunc, con, linestyle="dotted")
        plt.grid(True)
        plt.show()

        for i in range(inputChan):
            pr = impulseResponse[n * inputChan * frameNum + i * frameNum : n * inputChan * frameNum + (i + 1) * frameNum]
            plt.plot(timeFunc, pr)
        plt.grid(True)
        plt.show()



if __name__ == "__main__":

    sampleCalFromImg()
