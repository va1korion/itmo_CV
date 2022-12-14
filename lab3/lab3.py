import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.models import inception_v3, swin_s, vit_l_32, Inception_V3_Weights, Swin_S_Weights, ViT_L_32_Weights

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


def create_model(weights, model):
    modell = model(weights=weights)
    return modell.eval()


def create_predict(weights, img, model):
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    return model(batch).squeeze(0).softmax(0)


def predict(weights, model, j, act):
    pred = create_predict(weights.DEFAULT, img, model)
    top5_prob, top5_catid = torch.topk(pred, 5)
    top5_catid_temp = []
    for temp in top5_catid:
        top5_catid_temp.append(weights.DEFAULT.meta["categories"][temp].lower())

    top1_prob, top1_catid = torch.topk(pred, 1)
    if weights.DEFAULT.meta["categories"][top1_catid].lower() == act.lower():
        acc_1[j] += 1.0
    if act.lower() in top5_catid_temp:
        acc_5[j] += 1.0
    show_pred(pred, "alex_net", weights.DEFAULT)


if __name__ == '__main__':
    model_inception = create_model(Inception_V3_Weights.DEFAULT, inception_v3)  # classic inception
    model_swin = create_model(Swin_S_Weights.DEFAULT, swin_s)   # smaller transformer
    model_vit = create_model(ViT_L_32_Weights.DEFAULT, vit_l_32)   # bigger transformer

    from os import listdir
    from os.path import isfile, join

    MYPATH = 'data/'
    data_images = [f for f in listdir(MYPATH) if isfile(join(MYPATH, f))]

    from os import listdir
    from os.path import isfile, join

    MYPATH = 'data/'
    lables = [f for f in listdir(MYPATH) if not isfile(join(MYPATH, f))]

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
        act = data.split('/')[1].split('\\')[0]

        predict(Inception_V3_Weights, model_inception, 0, act)
        predict(Swin_S_Weights, model_swin, 0, act)
        predict(ViT_L_32_Weights, model_vit, 0, act)

    model_names = ['inception', 'SWIN', 'ViT']
    i = 0
    for n in model_names:
        print(n + 'top 1 accuracy = ' + str(acc_1[i] / len(data_images)))
        print(n + 'top 5 accuracy = ' + str(acc_5[i] / len(data_images)))
        i += 1
