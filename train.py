import time
import logging
import configparser
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GreedyViG
from dataset import ImageNetteDataset


def load_dataset(path, batch_size):
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]
    # is the order of tranfoms important? Is this the best order?
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means,
                             std=pretrained_stds),
        transforms.RandomErasing(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means,
                             std=pretrained_stds),
    ])

    # cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    # mixup = v2.MixUp(num_classes=NUM_CLASSES)
    # cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    train_dataset = ImageNetteDataset(
        path, split='train', transform=transform_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
    val_dataset = ImageNetteDataset(path, split='val', transform=transform_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
    return train_dataloader, val_dataloader


def train_step(model, dataloader, optimizer, criterion, device, epoch=None, mix_aug=None):
    running_loss, correct, total = [], 0, 0
    model.train()

    train_bar = tqdm(dataloader)
    for x, y in train_bar:
        x, y = x.to(device), y.to(device).long()

        if mix_aug is not None:
            x, y = mix_aug(x, y)

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        # predicted_class = pred.argmax(dim=1, keepdim=False)

        # total += y.numel()
        # correct += (predicted_class == y).sum().item()

        running_loss.append(loss.item())
        train_bar.set_description(
            f'Epoch: [{epoch}] Loss: {round(sum(running_loss) / len(running_loss), 6)}')
    # acc = correct / total
    acc = None
    return sum(running_loss) / len(running_loss), acc


def validation_step(model, dataloader, device):
    correct, total = 0, 0
    model.eval()

    validation_bar = tqdm(dataloader)
    with torch.no_grad():
        for x, y in validation_bar:
            x, y = x.to(device), y.to(device)

            pred = model(x)

            predicted_class = pred.argmax(dim=1, keepdim=False)

            total += y.numel()
            correct += (predicted_class == y).sum().item()

            acc = correct / total
            validation_bar.set_description(
                f'accuracy is {round(acc * 100, 2)}% until now.')
    return acc


def train(conf, device):
    save_dir = Path(conf['TRAIN']['SAVE_DIR']) / \
               datetime.now().strftime('%Y%m%d_%H%M')
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(save_dir / 'train.log'),
        logging.StreamHandler()
    ])
    model = GreedyViG(blocks=[[2,2], [2,2], [6,2], [2,2]],
              channels=[48, 96, 192, 384],
              dropout=0.,
              drop_path=0.1,
              emb_dims=768,
              K=[8, 4, 2, 1],
              num_classes=10)
    model.to(device)
    logging.info('Model loaded')
    logging.info({section: dict(conf[section]) for section in conf.sections()})

    train_dataloader, val_dataloader = load_dataset(
        conf['DATASET']['PATH'], conf['TRAIN'].getint('BATCH_SIZE'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=conf['TRAIN'].getfloat('LR'))

    loss_history, train_acc_hist, val_acc_hist = [], [], []
    max_val_acc = 0

    since = time.time()
    for epoch in range(1, conf['TRAIN'].getint('EPOCHS') + 1):

        loss, train_acc = train_step(
            model, train_dataloader, optimizer, criterion, device, epoch)
        val_acc = validation_step(model, val_dataloader, device)

        loss_history.append(loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        logging.info(f'Epoch: {epoch}, Loss: {loss}, Val acc: {val_acc * 100}')

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(model.state_dict(), save_dir /
                       f'best_model.pth')

    logging.info('Training Finished.')
    logging.info(f'Max validation accuracy is {round(max_val_acc * 100, 2)}%')
    logging.info(f'elapsed time is {time.time() - since}')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf = configparser.ConfigParser()
    conf.read('confs/main.ini')
    train(conf, device)
