import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil


class Bilinear_Pooling(nn.Module):
    def __init__(self,  **kwargs):
        super(Bilinear_Pooling, self).__init__()

    def forward(self, feature_map1, feature_map2):
        N, D1, H, W = feature_map1.size()
        feature_map1 = torch.reshape(feature_map1, (N, D1, H * W))
        N, D2, H, W = feature_map2.size()
        feature_map2 = torch.reshape(feature_map2, (N, D2, H * W))
        X = torch.bmm(feature_map1, torch.transpose(feature_map2, 1, 2)) / (H * W)
        X = torch.reshape(X, (N, D1 * D2))
        X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        bilinear_features = 100 * torch.nn.functional.normalize(X)
        return bilinear_features


def attention_erase(attention_maps, input_image):
    B,N,W,H = input_image.shape
    input = input_image
    batch_size, num_parts, height, width = attention_maps.shape
    attention_maps = torch.nn.functional.interpolate(attention_maps.detach(),size=(W,H),mode='bilinear')
    weights = F.avg_pool2d(attention_maps,(W,H)).reshape(batch_size,-1)
    weights = torch.add(torch.sqrt(weights),1e-12)
    weights = torch.div(weights,torch.sum(weights,dim=1).unsqueeze(1)).cpu().numpy()

    masks = []
    for i in range(batch_size):
        attention_map = attention_maps[i].detach()
        weight = weights[i]
        index = np.random.choice(np.arange(0, num_parts), 1, p=weight)[0]
        mask = attention_map[index:index + 1, :, :]
        threshold = random.uniform(0.2, 0.5) * mask.max()
        mask = (mask < threshold).float()
        masks.append(mask)
    masks = torch.stack(masks)
    erase_img = input*masks
    return erase_img


def Center_Loss(features, centers, target, alpha=0.95):
    features = features.reshape(features.shape[0], -1)
    target_centers = centers[target]
    target_centers = torch.nn.functional.normalize(target_centers, dim=-1)
    center_offset = (1-alpha)*(features.detach() - target_centers)
    distance = torch.pow(features - target_centers, 2)
    distance = torch.sum(distance, dim=-1)
    center_loss = torch.mean(distance)

    return center_loss, center_offset


def save_checkpoint(state, is_best, path='checkpoint', filename='checkpoint.pth.tar'):
    full_path = os.path.join(path, filename)
    torch.save(state, full_path)
    if is_best:
        shutil.copyfile(full_path, os.path.join(path, 'model_best.pth.tar'))
        print("Save best model at %s==" %
              os.path.join(path, 'model_best.pth.tar'))


def co_att(feature1, feature2):
    B, N, W, H = feature1.shape
    x1 = feature1.reshape(B, N, W*H)
    x2 = feature2.reshape(B, N, W*H)
    I = -torch.bmm(x1, x2.permute(0,2,1))
    I = F.softmax(I,2)
    Y = torch.bmm(I, feature1.reshape(B, N, W*H)).reshape(B, N, W, H)
    Y = Y + feature1
    return Y





