import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')
import torchvision.models as models
import cv2
import gc
import matplotlib.cm as cm
import BaseCAM_resnet
import BaseCAM_mobilenet

def load_model(model_name):
    global model, BaseCAMs
    assert model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                          'vgg16', 'vgg19', 'densenet121', 'denesnet169', 'inception_v3',
                          'mobilenetV2', 'mobilenetV3s', 'mobilenetV3l',] , \
        'Current available model: resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg19, densenet121, denesnet169, inception_v3, mobilenetV2, mobilenetV3s, mobilenetV3l'

    if 'resnet' in model_name:
        BaseCAMs = BaseCAM_resnet
        if model_name == 'resnet18':
            model = models.resnet18(True)
        elif model_name == 'resnet34':
            model = models.resnet34(True)
        elif model_name == 'resnet50':
            model = models.resnet50(True)
        elif model_name == 'resnet101':
            model = models.resnet101(True)
        elif model_name == 'resnet152':
            model = models.resnet152(True)
    elif 'vgg' in model_name:
        BaseCAMs = BaseCAM_vgg
        if model_name == 'vgg16':
            model = models.vgg16(True)
        elif model_name == 'vgg19':
            model = models.vgg19(True)
    elif 'mobilenet' in model_name:
        BaseCAMs = BaseCAM_mobilenet
        if model_name == 'mobilenetV3s':
            model = models.mobilenet_v3_small(True)
        elif model_name == 'mobilenetV3l':
            model = models.mobilenet_v3_large(True)
        elif model_name == 'mobilenetV2':
            model = models.mobilenet_v2(True)
    else:
        print('coming soon...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model = model.to(device)
    model = model.eval()
    return model, BaseCAMs

def preprocess_image(img):
    image = img.copy()
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    image = image[np.newaxis, ...]
    return torch.tensor(image, requires_grad=True)
def normalize(Ac):
    Ac_shape = Ac.shape
    AA = Ac.view(Ac.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    scaled_ac = AA.view(Ac_shape)
    return scaled_ac
def tensor2image(x, i=0):
    x = normalize(x)
    x = x[i].detach().cpu().numpy()
    x = cv2.resize(np.transpose(x, (1, 2, 0)), (224, 224))
    return x
def threshold(x):
    mean_ = x.mean()
    std_ = x.std()
    thresh = mean_ + std_
    x = (x > thresh)
    return x
def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)
def norm_image(image):
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)
def to0_1(mask):
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    return mask
def read_img(file_name, device='cuda'):
        img = cv2.imread(file_name)
        img = img[:, :, ::-1]
        img = np.float32(cv2.resize(img, (224,224))) / 255
        input = preprocess_image(img)
        input = input.to(device)
        img_show = img
        del img
        return input, img_show
# ------------------------- Mask visualization------------------------- #
def show_mask_GBR(mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    maskPic = np.float32(heatmap)
    maskPic = maskPic / np.max(maskPic)
    maskPic = np.uint8(255 * maskPic)
    maskPic = maskPic[:, :, ::-1]
    return maskPic
def show_mask_Seismic(mask):
    mask = cm.seismic(mask)
    mask = np.uint8(mask * 255)
    maskPic = cv2.cvtColor(mask, cv2.COLOR_RGBA2BGR)  #
    maskPic = maskPic[:, :, ::-1]
    return maskPic
def visualizing_CAM(img, mask, work_type='GBR'):
    if work_type == 'GBR':
        return show_cam_on_img_GBR(img, mask)
    elif work_type == 'Seismic':
        return show_cam_on_img_Seismic(img, mask)
    else:
        print('Error')
def show_cam_on_img_GBR(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    camImg = np.uint8(255 * cam)
    camImg = camImg[:, :, ::-1]
    return camImg
def show_cam_on_img_Seismic(img, mask):
    heatmap = cm.seismic(mask)
    heatmap = np.uint8(heatmap * 255)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGBA2BGR)
    cam = 1.0 * np.float32(heatmap / 255) + 1 * np.float32(img)
    cam = cam / np.max(cam)
    camImg = np.uint8(255 * cam)
    camImg = camImg[:, :, ::-1]
    return camImg
