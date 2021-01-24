import torch
import numpy as np
from utils import *


def train(train_loader, model, criterion, optimizer, center):
    train_loss = 0
    correct = 0
    total = 0
    idx = 0
    model.train()
    for i, data in enumerate(train_loader):
        idx = i
        img, positive_img, label = data
        input = img.cuda()
        positive_input = positive_img.cuda()
        label = torch.Tensor(label).type(torch.int64).cuda()

        out1, attention_maps1, bilinear_features, output1 = model(input)
        erase_img = attention_erase(attention_maps1, input)
        out2, _, _, output2 = model(erase_img)

        out3, attention_maps3, _, output3 = model(positive_input)
        erase_img_positive = attention_erase(attention_maps3, positive_input)
        out4, _, _, output4 = model(erase_img_positive)

        fuse_map1 = co_att(attention_maps1, attention_maps3)
        fuse_map2 = co_att(attention_maps3, attention_maps1)
        bilinear_pooling = Bilinear_Pooling()
        pooling1 = torch.flatten(bilinear_pooling(out1, fuse_map1), 1)
        pooling2 = torch.flatten(bilinear_pooling(out2, fuse_map2), 1)
        output5 = model.module.classifier(pooling1)
        output6 = model.module.classifier(pooling2)

        loss1 = criterion(output1, label)
        loss2 = criterion(output2, label)
        loss3 = criterion(output3, label)
        loss4 = criterion(output4, label)
        loss5 = criterion(output5, label)
        loss6 = criterion(output6, label)

        features = bilinear_features.reshape(bilinear_features.shape[0], -1)/100
        center_loss, center_diff = Center_Loss(features, center, label)
        center[label] += center_diff

        loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6 + center_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(output1.data, 1)
        total += label.size(0)
        correct += predicted.eq(label.data).cpu().sum()
        if i % 50 == 0:

            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                i, train_loss / (i + 1), 100. * float(correct) / total, correct, total))

    train_acc = 100. * float(correct) / total
    train_loss = train_loss / (idx + 1)

    return train_acc, train_loss


def test(test_loader, model, criterion, center):
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    model.eval()
    for i, data in enumerate(test_loader):
        idx = i
        img, label = data
        input = img.cuda()
        label = label.cuda()

        _, attention_maps, bilinear_features, output1 = model(input)

        erase_img = attention_erase(attention_maps, input)
        _, _, _, output2 = model(erase_img)

        loss1 = criterion(output1, label)
        loss2 = criterion(output2, label)

        features = bilinear_features.reshape(bilinear_features.shape[0], -1)/100
        center_loss, _ = Center_Loss(features, center, label)

        loss = (loss1 + loss2)/2 + center_loss
        test_loss += loss.item()

        _, predicted = torch.max(output1.data, 1)
        total += label.size(0)
        correct += predicted.eq(label.data).cpu().sum()

        if i % 50 == 0:
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            i, test_loss / (i + 1), 100. * float(correct) / total, correct, total))

    test_acc = 100. * float(correct) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_loss
