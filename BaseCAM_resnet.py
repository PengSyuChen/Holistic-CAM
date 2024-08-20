import torch
import torch.nn.functional as F
import copy

class MultiScalBaseCAM:
    def __init__(self, model, feature_module, get_bottleneck = False, target_bottleneck=None, get_layer=None, target_layer=None, get_conv=False, target_conv=None, inputResolutions=None):
        self.model = model
        self.inputResolutions = inputResolutions
        self.feature_module = feature_module
        self.get_bottleneck = get_bottleneck
        self.target_bottleneck = target_bottleneck
        self.get_layer = get_layer # Useless, to be consistent with mobilenet. The Value is always none.
        self.target_layer = target_layer # Useless too.
        self.get_conv = get_conv
        self.target_conv = target_conv

        if self.inputResolutions is None:
            self.inputResolutions = list(range(224, 1000, 100))

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.classDict = {}
        self.probsDict = {}
        self.featureDict = {}
        self.gradientsDict = {}

    def _recordActivationsAndGradients(self, inputResolution, image, classOfInterest=None):
        def forward_hook(module, input, output):
            self.featureDict[inputResolution] = (copy.deepcopy(output.clone().detach().cpu()))
        def backward_hook(module, grad_input, grad_output):
            self.gradientsDict[inputResolution] = (copy.deepcopy(grad_output[0].clone().detach().cpu()))
        for module_name, module in self.model._modules.items():
            if module == self.feature_module:
                if self.get_bottleneck == True:
                    for bottleneck_name, bottleneck in module._modules.items():
                        if bottleneck_name == self.target_bottleneck:
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
        probs, classes = softMaxScore.sort(dim=1, descending=True)
        if classOfInterest is None:
            ids = classes[:, [0]]
        else:
            ids = torch.tensor(classOfInterest).unsqueeze(dim=0).unsqueeze(dim=0).to(self.device)
        score = logits[0][ids[0][0]]
        self.score = score
        self.classDict[inputResolution] = ids.clone().detach().item()
        self.probsDict[inputResolution] = probs[0, 0].clone().detach().item()
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
                upSampledImage = image.to(self.device)
                logits = self._recordActivationsAndGradients(inputResolution, upSampledImage, classOfInterest=classOfInterest)
            else:
                upSampledImage = F.interpolate(image, (inputResolution, inputResolution), mode='bicubic', align_corners=False).to(self.device)
                self._recordActivationsAndGradients(inputResolution, upSampledImage, classOfInterest=classOfInterest)
        return logits