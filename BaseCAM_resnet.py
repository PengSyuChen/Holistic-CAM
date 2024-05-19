import numpy as np
import torch
import torch.nn.functional as F
import copy
import cv2

class CAM:
    def __init__(self, model, feature_module, get_bottleneck=False, target_bottleneck=None, get_conv=False, target_conv=None, device='cuda'):
        self.model = model  # 定义模型
        self.feature_module = feature_module  # 第几个layer
        self.get_bottleneck = get_bottleneck
        self.target_bottleneck = target_bottleneck
        self.get_conv = get_conv
        self.target_conv = target_conv
        self.device = device
        self.model = self.model.to(self.device)
        self.input = None
        self.output = None
        self.class_ = None
        self.probs_ = None
        self.activaion_ = None
        self.weight_ = None


    def _record_activations_and_weight(self, input, index=None):
        self.input = input

        def forward_hook(module, input, output):
            self.activaion_ = (copy.deepcopy(output.clone().detach().cpu()))

        # 提前注入钩子
        for module_name, module in self.model._modules.items():
            if module == self.feature_module:
                if self.get_bottleneck == True:
                    for bottleneck_name, bottleneck in module._modules.items():
                        if bottleneck_name == self.target_bottleneck:  # 神经网络模型块取出来的name是str
                            if self.get_conv == True:
                                for conv_name, conv in bottleneck._modules.items():
                                    if conv_name == self.target_conv:
                                        forwardHandle = conv.register_forward_hook(forward_hook)
                                        break
                            else:
                                forwardHandle = bottleneck.register_forward_hook(forward_hook)
                                break
                else:
                    forwardHandle = module.register_forward_hook(forward_hook)
                    break

        logits = self.model(input)
        softMaxScore = F.softmax(logits, dim=1)
        # 带着序号排序
        probs, classes = softMaxScore.sort(dim=1, descending=True)
        ids = classes[:, [0]]
        if index != None:
            ids = torch.tensor([[index]])
        ids = ids.to(self.device)
        self.class_ = ids.clone().detach().item()
        self.probs_ = probs[0, 0].clone().detach().item()
        self.weight_ = list(self.model.fc.parameters())[0]  # w
        forwardHandle.remove()
        del forward_hook  # 删除
        return logits

    def _compute_cam(self):
        weight = self.weight_[self.class_].cpu().data.numpy()
        activaion = self.activaion_[0].cpu().data.numpy()
        cam = np.zeros(activaion.shape[1:], dtype=np.float32)
        for i in range(0, activaion.shape[0]):
            cam += weight[i] * activaion[i]
        # cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, [self.input.shape[3], self.input.shape[2]])
        cam_min, cam_max = np.min(cam), np.max(cam)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        # cam = torch.from_numpy(cam).cuda()
        return cam
class BaseCAM:
    def __init__(self, model, feature_module, get_bottleneck=False, target_bottleneck=None, get_conv=False, target_conv=None, device='cuda'):
        self.model = model  # 定义模型
        self.feature_module = feature_module  # 第几个layer
        self.get_bottleneck = get_bottleneck
        self.target_bottleneck = target_bottleneck
        self.get_conv = get_conv
        self.target_conv = target_conv
        self.device = device
        # self.model = self.model.to(self.device)
        self.input = None
        self.output = None

        self.class_ = None
        self.probs_ = None
        self.activaion_ = None
        self.gradients_ = None

    def _record_activations_and_gradients(self, input, index=None):
        self.input = input

        def forward_hook(module, input, output):
            self.activaion_ = (copy.deepcopy(output.clone().detach().cpu()))

        def backward_hook(module, grad_input, grad_output):
            self.gradients_ = (copy.deepcopy(grad_output[0].clone().detach().cpu()))

        for module_name, module in self.model._modules.items():
            if module == self.feature_module:
                if self.get_bottleneck == True:
                    for bottleneck_name, bottleneck in module._modules.items():
                        if bottleneck_name == self.target_bottleneck:  # 神经网络模型块取出来的name是str
                            if self.get_conv == True:
                                for conv_name, conv in bottleneck._modules.items():
                                    if conv_name == self.target_conv:
                                        forwardHandle = conv.register_forward_hook(forward_hook)
                                        backwardHandle = conv.register_backward_hook(backward_hook)
                                        break
                            else:
                                forwardHandle = bottleneck.register_forward_hook(forward_hook)
                                backwardHandle = bottleneck.register_backward_hook(backward_hook)
                                break
                else:
                    forwardHandle = module.register_forward_hook(forward_hook)
                    backwardHandle = module.register_backward_hook(backward_hook)
                    break

        logits = self.model(input)
        softMaxScore = F.softmax(logits, dim=1)
        # 带着序号排序
        probs, classes = softMaxScore.sort(dim=1, descending=True)
        ids = classes[:, [0]]
        if index != None:
            ids = torch.tensor([[index]])
        ids = ids.to(self.device)
        self.class_ = ids.clone().detach().item()
        self.probs_ = probs[0, 0].clone().detach().item()

        one_hot = torch.zeros_like(logits)
        one_hot = one_hot.scatter_(1, ids, 1.0).to(self.device)
        # self.model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=False)
        forwardHandle.remove()
        backwardHandle.remove()
        del forward_hook  # 删除
        del backward_hook
        # print('前向传播&反向传播完毕')
        return logits
class MultiScalBaseCAM:
    def __init__(self, model, feature_module, get_bottleneck=False, target_bottleneck=None, get_conv=False, target_conv=None, inputResolutions=None, device='cuda'):
        self.model = model
        self.inputResolutions = inputResolutions
        self.feature_module = feature_module
        self.get_bottleneck = get_bottleneck
        self.target_bottleneck = target_bottleneck
        self.get_conv = get_conv
        self.device = device
        self.target_conv = target_conv

        if self.inputResolutions is None:
            self.inputResolutions = list(range(224, 1000, 100))  # 从224到1000输出每个数加100
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
        # 提前注入钩子
        for module_name, module in self.model._modules.items():
            if module == self.feature_module:
                if self.get_bottleneck == True:
                    for bottleneck_name, bottleneck in module._modules.items():
                        if bottleneck_name == self.target_bottleneck:  # 神经网络模型块取出来的name是str
                            if self.get_conv == True:
                                for conv_name, conv in bottleneck._modules.items():
                                    if conv_name == self.target_conv:
                                        forwardHandle = conv.register_forward_hook(forward_hook)
                                        backwardHandle = conv.register_backward_hook(backward_hook)
                                        break
                            else:
                                forwardHandle = bottleneck.register_forward_hook(forward_hook)
                                backwardHandle = bottleneck.register_backward_hook(backward_hook)
                                break
                else:
                    forwardHandle = module.register_forward_hook(forward_hook)
                    backwardHandle = module.register_backward_hook(backward_hook)
                    break

        logits = self.model(image)
        softMaxScore = F.softmax(logits, dim=1)
        # 带着序号排序
        probs, classes = softMaxScore.sort(dim=1, descending=True)
        if classOfInterest is None:
            ids = classes[:, [0]]
        else:
            ids = torch.tensor(classOfInterest).unsqueeze(dim=0).unsqueeze(dim=0)
        ids = ids.to(self.device)
        # score = logits[0][ids[0][0]]
        # self.score = score
        self.classDict[inputResolution] = ids.clone().detach().item()
        self.probsDict[inputResolution] = probs[0, 0].clone().detach().item()
        # self.scoresDict[inputResolution] = score.clone().detach().cpu()
        one_hot = torch.zeros_like(logits)
        one_hot = one_hot.scatter_(1, ids, 1.0).to(self.device)
        logits.backward(gradient=one_hot,
                        retain_graph=False)  # 根据one_hot锁定目标计算梯度 https://blog.csdn.net/baidu_38797690/article/details/122180655
        forwardHandle.remove()
        backwardHandle.remove()
        del forward_hook  # 删除
        del backward_hook
        return logits

    def run(self, image, classOfInterest=None, device='cuda'):
        for index, inputResolution in enumerate(self.inputResolutions):
            if index == 0:
                upSampledImage = image.to(device)
                logits = self._recordActivationsAndGradients(inputResolution, upSampledImage,
                                                                   classOfInterest=classOfInterest)  # 用于evaluation
            else:
                upSampledImage = F.interpolate(image, (inputResolution, inputResolution), mode='bicubic',
                                               align_corners=False).to(device)
                self._recordActivationsAndGradients(inputResolution, upSampledImage, classOfInterest=classOfInterest)
        return logits