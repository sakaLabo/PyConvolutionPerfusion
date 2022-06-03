EPOCHS = 500
BATCH_SIZE = 1
LEARNING_RATE = 0.001
REGULARIZATION = 0.01

ACCEPT_NEGATIVE = 0         # 0:正数のみ
DATA_SUB = 1

USE_CPU = 0

import torch
import time

class ConvolutionPerfusionModel(torch.nn.Module):

  def __init__(self, inputChan, frameNum, dataNum):
    super().__init__()
    self.outputChan = 1
    self.inputChan = inputChan
    self.frameNum =  frameNum
    self.dataNum = dataNum
    if USE_CPU == 1 :
        self.device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            print(f'CUDAデバイス名： {torch.cuda.get_device_name(0)}')
            self.device = torch.device("cuda")
        else:
            print('CPUで実行')
            self.device = torch.device("cpu") 

    self.conv = torch.nn.Conv1d(in_channels = inputChan * dataNum, out_channels = self.outputChan * dataNum, kernel_size = frameNum, groups = dataNum, padding = frameNum - 1, stride = 1, bias = False, padding_mode = 'zeros')

  def forward(self, LSF):
    conved = self.conv(LSF)
    return conved


class TrainMyNetwork():

    def __init__(self, inputChan, frameNum, dataNum):
        self.inputChan = inputChan
        self.frameNum =  frameNum
        self.dataNum = dataNum
        self.model = ConvolutionPerfusionModel(inputChan, self.frameNum, self.dataNum)

        if 0:
            self.model.load_state_dict(torch.load(TRAIN_MODEL_PATH))

        else:
            iniConv = torch.nn.Conv1d(in_channels = self.model.inputChan, out_channels = self.model.outputChan, kernel_size = frameNum, groups = 1, padding = frameNum - 1, stride = 1, bias = False, padding_mode = 'zeros')
            torch.nn.init.kaiming_uniform_(iniConv.weight, mode="fan_in")
            tmpW = iniConv.weight.repeat( self.dataNum, 1, 1)
            self.model.conv.weight = torch.nn.Parameter(tmpW)


        self.model = self.model.to(self.model.device)
        self.lossFunc = torch.nn.L1Loss()
 
        self.optimizer = torch.optim.Adam( self.model.parameters(), lr = LEARNING_RATE, weight_decay = REGULARIZATION)

        self.mask = torch.zeros((self.model.outputChan, self.model.inputChan, frameNum), device = self.model.device)

    def MakeMask(self, LSF):
        peakIDX = torch.argmax( LSF, 2 )
        maskIDX = torch.zeros( self.inputChan, device = self.model.device )
        for i in range(peakIDX.shape[1] - 1):
            maskIDX[i] = peakIDX[0][i+1] - peakIDX[0][i]
        maskIDX[peakIDX.shape[1] - 1] = self.frameNum - peakIDX[0][peakIDX.shape[1] - 1]
        for h in range(self.inputChan):
            for t in range(int(maskIDX[h])):
               self.mask[0][h][self.frameNum - t - 1] = 1
        return

    def TrainStep(self, LSF, target):
        self.model.train()
               
        self.predTARGET = self.model(LSF) 
        self.predTARGET = self.predTARGET[:, :, :self.frameNum] 

        self.optimizer.zero_grad()
        loss = self.lossFunc(self.predTARGET, target)   
        
        loss.backward()
        self.optimizer.step()
       
        if ACCEPT_NEGATIVE == 0:
            for p in self.model.parameters():
               p.data.clamp_(0.0) 
            p.data = p.data * self.mask

        return loss.item()

    def CalcImpulseResponse(self, LSF, target):
        if self.model.device == torch.device("cuda"):
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        self.MakeMask(LSF)

        if DATA_SUB == 1:
            LSF = torch.sub( LSF, torch.unsqueeze( torch.min( LSF, 2 )[0], 2 ) )
            target = torch.sub( target, torch.unsqueeze( torch.min( target, 2 )[0], 2 ) )

        LSF = LSF.repeat(1, self.dataNum, 1)


        self.train_history = []        
        start = time.time() 

        for epoch in range(EPOCHS):

            loss = self.TrainStep(LSF, target)
            avgLoss = loss / self.dataNum

            self.train_history.append(avgLoss)            

        print ("elapsed_time:{0}".format(time.time() - start) + "[sec]")
        print("avgLoss:{0}".format(avgLoss))
             
        for p in self.model.parameters():
            prmList=p.data
        prmList=torch.flip(prmList, [2])

        return prmList

    def CalcBloodFlow(self, srcTimeFunc, srcIR):
        frameStep = srcTimeFunc[0][0][1] - srcTimeFunc[0][0][0];
        srcIR = srcIR.permute((1, 0, 2))
        srcIR = torch.div( srcIR, frameStep );
        tmpBf = torch.max( srcIR, 2 )

        dstBF = tmpBf[0]
        dstBV = torch.trapz( srcIR, srcTimeFunc )
        dstMTT = torch.div( dstBV, dstBF )
        
        coefBF = 100.0 * 60.0
        dstBF = torch.mul( dstBF, coefBF )
        coefBV = 100.0
        dstBV = torch.mul( dstBV, coefBV )
        dstMTT[dstMTT == float('inf')] = 0

        return dstBF, dstBV, dstMTT