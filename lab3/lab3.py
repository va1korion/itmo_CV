import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.models import alexnet, resnet50, vgg16, AlexNet_Weights, ResNet50_Weights, VGG16_Weights

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_pred(prediction, name, weights):
    val, index = torch.topk(prediction, 5)
    print(name)
    for i in range(5):
        category_name = weights.meta["categories"][index[i]]
        print(f"{category_name}: {100 * val[i]:.1f}%")


def read_image(i):
    from torchvision.io import read_image
    return read_image(f"{i}")


def create_res_net50_model():
    from torchvision.models import resnet50, ResNet50_Weights
    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    return model


def create_model(weights, model):
    modell = model(weights=weights)
    return modell.eval()


def create_alex_net_model(weights, model):
    from torchvision.models import alexnet, AlexNet_Weights

    # Step 1: Initialize model with the best available weights
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    return model


def create_predict(weights, img, model):
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    return model(batch).squeeze(0).softmax(0)


def create_vgg16_model():
    from torchvision.models import vgg16, VGG16_Weights

    # Step 1: Initialize model with the best available weights
    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    return model


if __name__ == '__main__':
    model_alex_net = create_model(AlexNet_Weights.DEFAULT, alexnet)
    model_res_net = create_model(ResNet50_Weights.DEFAULT, resnet50)
    model_vgg_16 = create_model(VGG16_Weights.DEFAULT, vgg16)
    from os import listdir
    from os.path import isfile, join

    MYPATH = 'data/'
    data_images = [f for f in listdir(MYPATH) if isfile(join(MYPATH, f))]

    from os import listdir
    from os.path import isfile, join

    MYPATH = 'data/'
    lables = [f for f in listdir(MYPATH) if not isfile(join(MYPATH, f))]
    lables

    data_images = []
    for l in lables:
        for d in listdir(join(MYPATH, l)):
            data_images.append(join(MYPATH, l, d))

    acc_5 = [0, 0, 0]
    acc_1 = [0, 0, 0]
    for data in data_images:
        img = read_image(data)
        print("=========================")
        print(f"exp{data}")
        print("=========================")
        pred = create_predict(AlexNet_Weights.DEFAULT, img, model_alex_net)
        act = data.split('/')[1].split('\\')[0]
        top5_prob, top5_catid = torch.topk(pred, 5)
        top5_catid_temp = []
        for temp in top5_catid:
            top5_catid_temp.append(AlexNet_Weights.DEFAULT.meta["categories"][temp].lower())
        top1_prob, top1_catid = torch.topk(pred, 1)
        if AlexNet_Weights.DEFAULT.meta["categories"][top1_catid].lower() == act.lower():
            acc_1[0] += 1.0
        if act.lower() in top5_catid_temp:
            acc_5[0] += 1.0

        show_pred(pred, "alex_net", AlexNet_Weights.DEFAULT)
        pred = create_predict(ResNet50_Weights.DEFAULT, img, model_res_net)
        top5_prob, top5_catid = torch.topk(pred, 5)
        top5_catid_temp = []
        for temp in top5_catid:
            top5_catid_temp.append(AlexNet_Weights.DEFAULT.meta["categories"][temp].lower())
        top1_prob, top1_catid = torch.topk(pred, 1)
        if (AlexNet_Weights.DEFAULT.meta["categories"][top1_catid].lower() == act.lower()):
            acc_1[1] += 1.0
        if (act.lower() in top5_catid_temp):
            acc_5[1] += 1.0
        show_pred(pred, "res net 50", ResNet50_Weights.DEFAULT)

        pred = create_predict(VGG16_Weights.DEFAULT, img, model_vgg_16)
        top5_prob, top5_catid = torch.topk(pred, 5)
        top5_catid_temp = []
        for temp in top5_catid:
            top5_catid_temp.append(AlexNet_Weights.DEFAULT.meta["categories"][temp].lower())
        top1_prob, top1_catid = torch.topk(pred, 1)
        if (AlexNet_Weights.DEFAULT.meta["categories"][top1_catid].lower() == act.lower()):
            acc_1[2] += 1.0
        if (act.lower() in top5_catid_temp):
            acc_5[2] += 1.0
        show_pred(pred, "vgg 16", VGG16_Weights.DEFAULT)

    model_names = ['alex_net', 'res net 50', 'vgg16']
    i = 0
    for n in model_names:
        print(n + 'top 1 accuracy = ' + str(acc_1[i] / len(data_images)))
        print(n + 'top 5 accuracy = ' + str(acc_5[i] / len(data_images)))
        i += 1