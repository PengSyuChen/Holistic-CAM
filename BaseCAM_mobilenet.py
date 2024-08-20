import numpy as np
import torch
import torch.nn.functional as F
import copy

class MultiScalBaseCAM:
    def __init__(self, model, feature_module=None, get_bottleneck=False, target_bottleneck=None, get_layer=False, target_layer=None, get_conv=False, target_conv=None, inputResolutions=None):
        self.model = model
        self.inputResolutions = inputResolutions
        self.target_bottleneck = target_bottleneck
        self.get_layer = get_layer
        self.target_layer = target_layer
        self.get_conv = get_conv
        self.target_conv = target_conv

        if self.inputResolutions is None:
            self.inputResolutions = list(range(224, 1000, 100))
            # [224, 324, 424, 524, 624, 724, 824, 924]

        self.classDict = {}
        self.probsDict = {}
        self.featureDict = {}
        self.gradientsDict = {}

    def _recordActivationsAndGradients(self, inputResolution, image, classOfInterest=None):
        def forward_hook(module, input, output):
            self.featureDict[inputResolution] = (copy.deepcopy(output.clone().detach().cpu()))

        def backward_hook(module, grad_input, grad_output):
            self.gradientsDict[inputResolution] = (copy.deepcopy(grad_output[0].clone().detach().cpu()))
        '''
        (17): InvertedResidual(
              (conv): Sequential(
                      (0): Conv2dNormActivation(
                           (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
                           (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                           (2): ReLU6(inplace=True)
                      )
                      (1): Conv2dNormActivation(
                          (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
                          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                          (2): ReLU6(inplace=True)
                      )
                     (2): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
                     (3): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
        )
        '''
        for module_name, module in self.model._modules.items():
            if module_name == 'features':  # features
                for block_name, block in module._modules.items():
                    if block_name == self.target_bottleneck:  # (17) InvertedResidual
                        if self.get_layer == True:
                            for layer_name, layer in block._modules.items(): # (conv)
                                if self.get_conv == True:
                                    for conv_name, conv in layer._modules.items(): # (0) Conv2dNormActivation 0 or 2
                                        if conv_name == self.target_conv:  # (0): Conv2d
                                            forwardHandle = conv.register_forward_hook(forward_hook)
                                            backwardHandle = conv.register_backward_hook(backward_hook)
                                            break
                                else:
                                    forwardHandle = layer.register_forward_hook(forward_hook)
                                    backwardHandle = layer.register_backward_hook(backward_hook)
                                    break
                        else:
                            forwardHandle = block.register_forward_hook(forward_hook)
                            backwardHandle = block.register_backward_hook(backward_hook)
                            break

        logits = self.model(image)
        softMaxScore = F.softmax(logits, dim=1)
        probs, classes = softMaxScore.sort(dim=1, descending=True)
        if classOfInterest is None:
            ids = classes[:, [0]]
        else:
            ids = torch.tensor(classOfInterest).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
        score = logits[0][ids[0][0]]
        self.score = score
        self.classDict[inputResolution] = ids.clone().detach().item()
        self.probsDict[inputResolution] = probs[0, 0].clone().detach().item()
        # self.scoresDict[inputResolution] = score.clone().detach().cpu()
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, ids, 1.0)
        logits.backward(gradient=one_hot,
                        retain_graph=False)
        forwardHandle.remove()
        backwardHandle.remove()
        del forward_hook
        del backward_hook
        return logits

    def run(self, image, classOfInterest=None):
        for index, inputResolution in enumerate(self.inputResolutions):
            if index == 0:
                upSampledImage = image.cuda()
                logits = self._recordActivationsAndGradients(inputResolution, upSampledImage, classOfInterest=classOfInterest)
            else:
                upSampledImage = F.interpolate(image, (inputResolution, inputResolution), mode='bicubic',
                                               align_corners=False).cuda()
                self._recordActivationsAndGradients(inputResolution, upSampledImage, classOfInterest=classOfInterest)
        return logits