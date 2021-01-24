import logging
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset import config, Dataset, collate_fn
from utils import *
from train import train, test
from model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
cudnn.benchmark = False
def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='PCA-Net parameters')
    parser.add_argument('--dataset', metavar='DIR', default='bird', help='bird car aircraft')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--model-name', default='resnet50', type=str, help='model name')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--decay-step', default=2, type=int, metavar='N',
                        help='learning rate decay step')
    parser.add_argument('--gamma', default=0.9, type=float, metavar='M',
                        help='gamma')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--checkpoint-path', default='./checkpoint_bird', type=str, metavar='checkpoint_path',
                        help='path to save checkpoint')

    args = parser.parse_args()
    return args


args = parse_args()
print(args)
init_seeds(seed=0)
best_acc1 = 0.

try:
    os.stat(args.checkpoint_path)
except:
    os.makedirs(args.checkpoint_path)

logging.info("OPENING " + args.checkpoint_path + '/results_train.csv')
logging.info("OPENING " + args.checkpoint_path + '/results_test.csv')

results_train_file = open(args.checkpoint_path + '/results_train.csv', 'w')
results_train_file.write('epoch, train_acc,train_loss\n')
results_train_file.flush()

results_test_file = open(args.checkpoint_path + '/results_test.csv', 'w')
results_test_file.write('epoch, test_acc,test_loss\n')
results_test_file.flush()

# dataset
train_root, test_root, train_pd, test_pd, cls_num = config(data=args.dataset)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomCrop((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}
train_dataset = Dataset(train_root, train_pd, train=True, transform=data_transforms['train'], num_positive=1)
test_dataset = Dataset(test_root, test_pd, train=False, transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


model = resnet50(pretrained=True, use_bp=True)
in_features = model.classifier.in_features
model.classifier = torch.nn.Linear(in_features=in_features, out_features=cls_num)

model = model.cuda()
model = torch.nn.DataParallel(model)

# feature center
feature_len = 512
center_dict = {'center': torch.zeros(cls_num, feature_len * 32)}
center = center_dict['center'].cuda()

criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.cuda()

optimizer = torch.optim.SGD(
    model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
cudnn.benchmark = True

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

for epoch in range(args.start_epoch, args.epochs):
    scheduler.step()
    for param_group in optimizer.param_groups:
        lr_val = float(param_group['lr'])
    print("Start epoch %d, lr=%f" % (epoch, lr_val))

    train_acc, train_loss = train(train_loader, model, criterion, optimizer, center)
    logging.info('Iteration %d, train_acc = %.4f,train_loss = %.4f' % (epoch, train_acc, train_loss))
    results_train_file.write('%d, %.4f,%.4f\n' % (epoch, train_acc, train_loss))
    results_train_file.flush()

    val_acc, val_loss = test(test_loader, model, criterion, center)
    is_best = val_acc > best_acc1
    best_acc1 = max(val_acc, best_acc1)
    logging.info('Iteration %d, test_acc = %.4f,test_loss = %.4f' % (epoch, val_acc, val_loss))
    results_test_file.write('%d, %.4f,%.4f\n' % (epoch, val_acc, val_loss))
    results_test_file.flush()

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
        'center': center
    }, is_best, args.checkpoint_path)






