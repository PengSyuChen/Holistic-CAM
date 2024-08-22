import torch
import cv2
import numpy as np
import torch.nn.functional as F
class HolisticCAM():
    def __init__(self, obj):
        self._copy_(obj)

    # due to the difference forward/backward procress of diffenent CNNs,
    # we choose isolate these procresses into single .py class
    def _copy_(self, BaseCam):  # forward and backward
        self.inputResolutions = BaseCam.inputResolutions
        self.featureDict = BaseCam.featureDict
        self.gradientsDict = BaseCam.gradientsDict
        self.classDict = BaseCam.classDict

    # Positive Gradient Enhancement
    def PGE(self, activations, grads):
        grad_1 = grads
        sum_acitvations = torch.sum(activations[0], dim=(1, 2))
        eps = 0.00001
        A_s = sum_acitvations[..., None, None]
        grad_t_2 = grad_1 * grad_1
        grad_t_3 = grad_t_2 * grad_1
        a_kc = grad_t_2 / (2 * grad_t_2 + A_s * grad_t_3 + eps) # ref to GradCAM++
        a_kc = torch.where(grads != 0, a_kc, 0)
        a_kc = torch.sum(a_kc, dim=0)
        W_kc = grads * a_kc # element-wise weighting, Eq.8
        return W_kc

    # Fundamental Scale Denoising
    def FSD(self, PHA, bMask, blurSize=51, ksize=(91, 91)):
        # fix value filtering
        sMin, sMax = bMask.min(), bMask.max()
        bMask = (bMask - sMin) / (sMax - sMin) # to 0~1
        bMask = torch.where(bMask > 0.5, 0.5, bMask) # fix out the high-frequency information
        sMin, sMax = bMask.min(), bMask.max()
        bMask = (bMask - sMin) / (sMax - sMin) # to 0~1
        bMask = bMask * 255
        # 𝑚𝑒𝑑𝑖𝑎𝑛_𝑏𝑙𝑢𝑟2𝑑
        bMask = cv2.medianBlur(cv2.convertScaleAbs(bMask.cpu().numpy()), blurSize)
        # 𝑚𝑒𝑎𝑛_𝑏𝑙𝑢𝑟2𝑑
        LPW = cv2.blur(bMask, ksize)
        LPW = torch.from_numpy(LPW).cuda()
        return PHA * LPW

    # High Resolution Attribution Generation
    def PHA_Generation(self, classOfInterest=None, gradient_enhance=True):
        fundamentalScale = self.inputResolutions[0]
        groundTruthClass = self.classDict[fundamentalScale]
        count = 0
        fusedWeights = None
        fusedAcitvationMaps = None
        fundamentalScaleMask = None
        for resolution in self.inputResolutions:
            if groundTruthClass == self.classDict[resolution] or self.classDict[resolution] == classOfInterest:
                count += 1
                # excuse PGE
                if gradient_enhance:
                    positiveGradient_t = F.relu(self.gradientsDict[resolution])
                    weight_t = self.PGE(self.featureDict[resolution], positiveGradient_t)
                else:
                    weight_t = self.gradientsDict[resolution]
                weight_t = F.interpolate(weight_t.cuda(), (fundamentalScale, fundamentalScale), mode='bilinear', align_corners=False)
                feature_t = F.interpolate(self.featureDict[resolution].cuda(), (fundamentalScale, fundamentalScale), mode='bilinear', align_corners=False)  # bilinear
                if count == 1: # fundamentalScaleMask
                    fusedWeights = weight_t
                    fusedAcitvationMaps = feature_t
                    fundamentalScaleMask = (fusedWeights * fusedAcitvationMaps)[0].sum(dim=0, keepdim=False) # dim of channel
                else:
                    fusedWeights += weight_t
                    fusedAcitvationMaps += feature_t
        fusedWeights = fusedWeights / count
        fusedAcitvationMaps = fusedAcitvationMaps / count # torch.Size([1, 512, 224, 224])
        primaryAttributionMap = (fusedWeights * fusedAcitvationMaps)[0].sum(dim=0, keepdim=False)
        return primaryAttributionMap, fundamentalScaleMask

    def holistic_CAM(self, classOfInterest=None, gradient_enhance=True, denosing=False, blurSize=51, ksize=(91,91)):
        primaryAttributionMap, fundamentalScaleMask = self.PHA_Generation(classOfInterest, gradient_enhance)
        if denosing:
            holisticCAM = self.FSD(primaryAttributionMap, fundamentalScaleMask, blurSize, ksize)
        else:
            holisticCAM = primaryAttributionMap
        # generate cam mask
        cam = holisticCAM.float().data.cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam
