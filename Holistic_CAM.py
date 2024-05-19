class Holistic_CAM():
    def __init__(self, obj):
       self._copy_(obj) 
    def _copy_(self, BaseCam):
        self.inputResolutions = BaseCam.inputResolutions
        self.featureDict = BaseCam.featureDict
        self.gradientsDict = BaseCam.gradientsDict
        self.classDict = BaseCam.classDict
        self.device = BaseCam.device
    def positiveGradientEnhancement(self, activations, grads, select=0, derivatives=1):
        global res
        grad_1 = grads
        grad_t = grads
        a_kc_s = []
        sum_acitvations = torch.sum(activations[0], dim=(1, 2))
        eps = 0.000001
        # A_s = activations
        A_s = sum_acitvations[..., None, None]
        for t in range(1, derivatives + 1):
            grad_t_p_1 = grad_t * grad_1
            grad_t_p_2 = grad_t_p_1 * grad_1
            a_kc = grad_t_p_1 / ((t + 1) * grad_t_p_1 + A_s * grad_t_p_2 + eps) # [2048,7,7]
            a_kc = torch.where(grads != 0, a_kc, 0)
            a_kc_s.append(a_kc)
        res = torch.stack(a_kc_s[select: derivatives])
        a_kc = torch.sum(res, dim=0)
        W_kc = grads * a_kc
        return W_kc

    def fundamentalScaleDenoising(self, baseScaleMask, fusedMask, middleBlurSize, meanBlurSize):
        basicScaleMap = baseScaleMask.to(self.device)
        saliencyMap = fusedMask[0].sum(dim=0, keepdim=False)
        basicScaleMap = basicScaleMap[0].sum(dim=0, keepdim=False)
        sMin, sMax = basicScaleMap.min(), basicScaleMap.max()
        basicScaleMap = (basicScaleMap - sMin) / (sMax - sMin)
        basicScaleMap = torch.where(basicScaleMap > 0.5, 0.5, basicScaleMap)
        sMin, sMax = basicScaleMap.min(), basicScaleMap.max()
        basicScaleMap = (basicScaleMap - sMin) / (sMax - sMin)
        basicScaleMap = basicScaleMap * 255
        basicScaleMap = basicScaleMap.cpu().numpy()
        basicScaleMap = cv2.medianBlur(cv2.convertScaleAbs(basicScaleMap), middleBlurSize)
        basicScaleMap = cv2.blur(basicScaleMap, meanBlurSize)
        basicScaleMap = torch.from_numpy(basicScaleMap).to(self.device)

        sMin, sMax = basicScaleMap.min(), basicScaleMap.max()
        basicScaleMap = (basicScaleMap - sMin) / (sMax - sMin)
        saliencyMap = saliencyMap * basicScaleMap
        return saliencyMap

    def _estimateSaliencyMap(self, selects, derivatives, classOfInterest=None, blur=False, blurerKernelSize=None, blurer_2_size=None, blurWorkType='all'):
        self.blur = blur
        self.blurerKernelSize = blurerKernelSize
        self.blurer_2_size = blurer_2_size
        self.blurWorkType = blurWorkType

        saveResolution = self.inputResolutions[0]
        groundTruthClass = self.classDict[saveResolution]

        count = 0
        basicScaleMap = None
        meanScaledFeatures = None
        meanScaledPPweight = None
        saveFeatures = None
        savePpweights = None
        for resolution in self.inputResolutions:
             if groundTruthClass == self.classDict[resolution] or self.classDict[resolution] == classOfInterest:
                 count += 1
                 A = self.featureDict[resolution]
                 positive_gradients = F.relu(self.gradientsDict[resolution])
                 ppweight = self.positiveGradientEnhancement(A, positive_gradients, selects, derivatives)
                 ppweight = F.interpolate(ppweight.to(self.device), (saveResolution, saveResolution), mode='bilinear', align_corners=False)
                 upSampledFeatures = F.interpolate(self.featureDict[resolution].to(self.device), (saveResolution, saveResolution), mode='bilinear', align_corners=False) # 双线性插值法
                 if meanScaledFeatures is None:
                    meanScaledFeatures = upSampledFeatures
                 else:
                    meanScaledFeatures += upSampledFeatures

                 if meanScaledPPweight is None:
                    meanScaledPPweight = ppweight
                 else:
                    meanScaledPPweight += ppweight

             if count == 1:
                 savePpweights = ppweight
                 saveFeatures = upSampledFeatures
        ppweight = meanScaledPPweight / count
        feature = meanScaledFeatures / count
        ppweight = (ppweight-ppweight.min())/(ppweight.max() - ppweight.min())
        saliencyMap = (ppweight * feature).to(self.device)
        basicScaleMap = (savePpweights * saveFeatures).to(self.device)
        cam = self.fundamentalScaleDenoising(saliencyMap, basicScaleMap, self.blurerKernelSize, self.blurer_2_size)
        cam = cam.float().data.cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam
