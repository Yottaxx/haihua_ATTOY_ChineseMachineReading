import json
from collections import OrderedDict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import time
from sklearn.model_selection import *
from transformers import *
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse

from utils.MyDataset import MyDataset
from utils.tools import seed_everything

parser = argparse.ArgumentParser()
# 注意这个参数，必须要以这种形式指定，即使代码中不使用。因为 launch 工具默认传递该参数
parser.add_argument("--local_rank", type=int)

CFG = {  # 训练的参数配置
    'fold_num': 6,  # 五折交叉验证
    'seed': 42,
    'model': 'hfl/chinese-macbert-large',  # 预训练模型
    'max_len': 256,  # 文本截断的最大长度
    'epochs': 8,
    'train_bs': 2,  # batch_size，可根据自己的显存调整
    'valid_bs': 2,
    'lr': 2e-5,  # 学习率
    'lrSelf': 1e-4,  # 学习率
    'num_workers': 8,
    'accum_iter': 8,  # 梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4,  # 权重衰减，防止过拟合
    'device': 0,
    'adv_lr': 0.01,
    'adv_norm_type': 'l2',
    'adv_init_mag': 0.03,
    'adv_max_norm': 1.0,
    'ip': 2,
    'gpuNum': 4
}


seed_everything(CFG['seed'])  # 固定随机种子
args = parser.parse_args()
# torch.cuda.set_device(CFG['device'])
device = torch.device('cuda', args.local_rank)

train_df = pd.read_csv('./utils/train.csv')
test_df = pd.read_csv('./utils/test.csv')
train_df['label'] = train_df['Answer'].apply(lambda x: ['A', 'B', 'C', 'D'].index(x))  # 将标签从ABCD转成0123
test_df['label'] = 0
tokenizer = BertTokenizer.from_pretrained(CFG['model'])  # 加载bert的分词器


def collate_fn(data):  # 将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch, n_choices, max_len)
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(x[1], text_pair=x[0], padding='max_length', truncation=True, max_length=CFG['max_len'],
                         return_tensors='pt')
        input_ids.append(text['input_ids'].tolist())
        attention_mask.append(text['attention_mask'].tolist())
        token_type_ids.append(text['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([x[-1] for x in data])
    return input_ids, attention_mask, token_type_ids, label


class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def getDelta(attention_mask, embeds_init):
    delta = None
    batch = embeds_init.shape[0]
    length = embeds_init.shape[-2]
    dim = embeds_init.shape[-1]

    attention_mask = attention_mask.view(-1, length)
    embeds_init = embeds_init.view(-1, length, dim)
    if CFG['adv_init_mag'] > 0:  # 影响attack首步是基于原始梯度(delta=0)，还是对抗梯度(delta!=0)
        input_mask = attention_mask.to(embeds_init)
        input_lengths = torch.sum(input_mask, 1)
        if CFG['adv_norm_type'] == "l2":
            delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
            dims = input_lengths * embeds_init.size(-1)
            mag = CFG['adv_init_mag'] / torch.sqrt(dims)
            delta = (delta * mag.view(-1, 1, 1)).detach()
        elif CFG['adv_norm_type'] == "linf":
            delta = torch.zeros_like(embeds_init).uniform_(-CFG['adv_init_mag'], CFG['adv_init_mag'])
            delta = delta * input_mask.unsqueeze(2)
    else:
        delta = torch.zeros_like(embeds_init)  # 扰动初始化

    return delta.view(batch, -1, length, dim)


def updateDelta(delta, delta_grad, embeds_init):
    batch = delta.shape[0]
    length = delta.shape[-2]
    dim = delta.shape[-1]
    delta = delta.view(-1, length, dim)
    delta_grad = delta_grad.view(-1, length, dim)

    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
    denorm = torch.clamp(denorm, min=1e-8)
    delta = (delta + CFG['adv_lr'] * delta_grad / denorm).detach()
    if CFG['adv_max_norm'] > 0:
        delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
        exceed_mask = (delta_norm > CFG['adv_max_norm']).to(embeds_init)
        reweights = (CFG['adv_max_norm'] / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
        #        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        delta = (delta * reweights).detach()

    return delta.view(batch, -1, length, dim)


def train_model(model, train_loader):  # 训练一个epoch
    model.train()
    # print(model)
    losses = AverageMeter()
    accs = AverageMeter()

    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    for step, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
        input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
            device), token_type_ids.to(device), y.to(device).long()

        with autocast():  # 使用半精度训练
            with torch.no_grad():
                kd1 = F.softmax(modelT1(input_ids, attention_mask, token_type_ids)[0], dim=1)
                kd2 = F.softmax(modelT2(input_ids, attention_mask, token_type_ids)[0], dim=1)
                kd3 = F.softmax(modelT3(input_ids, attention_mask, token_type_ids)[0], dim=1)
                kd = (kd1 + kd2 + kd3) / 2

            if isinstance(model, torch.nn.DataParallel) or isinstance(model, DDP):
                embeds_init = model.module.bert.embeddings.word_embeddings(input_ids)
            else:
                embeds_init = model.bert.embeddings.word_embeddings(input_ids)

            # prepare random unit tensor
            d = getDelta(attention_mask=attention_mask, embeds_init=embeds_init)

            ip = CFG['ip']
            for i in range(ip):
                d.requires_grad_()
                embeds_init = embeds_init + d
                output = model(
                    inputs_embeds=embeds_init,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)[
                    0]

                loss = criterion(output, y) - (kd * torch.log(F.softmax(output))).sum(dim=1).mean()
                loss = loss / ip

                if CFG['gpuNum'] > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if CFG['accum_iter'] > 1:
                    loss = loss / CFG['accum_iter']

                scaler.scale(loss).backward()

                acc = (output.argmax(1) == y).sum().item() / y.size(0)

                losses.update(loss.item() * CFG['accum_iter'], y.size(0))
                accs.update(acc, y.size(0))

                tk.set_postfix(loss=losses.avg, acc=accs.avg)

                delta_grad = d.grad.clone().detach()
                d = updateDelta(d, delta_grad, embeds_init)
                if isinstance(model, torch.nn.DataParallel) or isinstance(model, DDP):
                    embeds_init = model.module.bert.embeddings.word_embeddings(input_ids)
                else:
                    embeds_init = model.bert.embeddings.word_embeddings(input_ids)

            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

    return losses.avg, accs.avg


def test_model(model, val_loader):  # 验证
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    y_truth, y_pred = [], []

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), y.to(device).long()

            output = model(input_ids, attention_mask, token_type_ids)[0]

            y_truth.extend(y.cpu().numpy())
            y_pred.extend(output.argmax(1).cpu().numpy())

            loss = criterion(output, y)

            acc = (output.argmax(1) == y).sum().item() / y.size(0)

            losses.update(loss.item(), y.size(0))
            accs.update(acc, y.size(0))

            tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg


dist.init_process_group(backend='nccl')

folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']) \
    .split(np.arange(train_df.shape[0]), train_df.label.values)  # 五折交叉验证

cv = []  # 保存每折的最佳准确率

for fold, (trn_idx, val_idx) in enumerate(folds):

    train = train_df.loc[trn_idx]
    val = train_df.loc[val_idx]

    train_set = MyDataset(train)
    val_set = MyDataset(val)

    train_sampler = DistributedSampler(train_set, num_replicas=dist.get_world_size(), rank=args.local_rank)
    dev_sampler = DistributedSampler(val_set, num_replicas=dist.get_world_size(), rank=args.local_rank)

    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=CFG['train_bs'], collate_fn=collate_fn,
                              num_workers=0)
    val_loader = DataLoader(val_set, sampler=dev_sampler, batch_size=CFG['valid_bs'], collate_fn=collate_fn,
                            shuffle=False,
                            num_workers=0)

    best_acc = 0

    model = BertForMultipleChoice.from_pretrained(CFG['model']).to(device)  # 模型
    new_layer = ["bert"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)],
         "lr": CFG['lrSelf']},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], }
    ]

    scaler = GradScaler()
    optimizer = AdamW(optimizer_grouped_parameters, lr=CFG['lr'], weight_decay=CFG['weight_decay'])  # AdamW优化器
    criterion = nn.CrossEntropyLoss()
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // CFG['accum_iter'],
                                                CFG['epochs'] * len(train_loader) // CFG['accum_iter'])
    # get_cosine_schedule_with_warmup策略，学习率先warmup一个epoch，然后cos式下降
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    modelT1 = BertForMultipleChoice.from_pretrained(CFG['model']).to(device)
    modelT2 = BertForMultipleChoice.from_pretrained(CFG['model']).to(device)
    modelT3 = BertForMultipleChoice.from_pretrained(CFG['model']).to(device)

    modelT1.eval()
    modelT2.eval()
    modelT3.eval()

    predictions = []

    # load params

    for item in range(3):  # 把训练后的五个模型挨个进行预测
        new_state_dict = OrderedDict()
        state_dict = torch.load('LargeMacFinal_{}_fold_{}.pt'.format(CFG['model'].split('/')[1], item + 1),
                                map_location=device)
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v

        if item == 0:
            print("loading----fold:1------")
            modelT1.load_state_dict(
                new_state_dict)
        if item == 1:
            print("loading----fold:2------")
            modelT2.load_state_dict(
                new_state_dict)
        if item == 2:
            print("loading----fold:3------")
            modelT3.load_state_dict(
                new_state_dict)

    modelT1 = DDP(modelT1, device_ids=[args.local_rank], output_device=args.local_rank)
    modelT2 = DDP(modelT2, device_ids=[args.local_rank], output_device=args.local_rank)
    modelT3 = DDP(modelT3, device_ids=[args.local_rank], output_device=args.local_rank)

    print(device)
    # print(model)

    for epoch in range(CFG['epochs']):

        if epoch == 6:
            break

        print('epoch:', epoch)
        time.sleep(0.2)
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_model(model, train_loader)
        val_loss, val_acc = test_model(model, val_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'LargeMacKD256sub_{}_fold_{}.pt'.format(CFG['model'].split('/')[1], fold))

    cv.append(best_acc)
    with open("resultKD.txt", "a+") as f:
        f.write(str(args.local_rank) + ' ' + CFG['model'] + " " + str(best_acc) + '\n')

test_set = MyDataset(test_df)
test_loader = DataLoader(test_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                         num_workers=CFG['num_workers'])
model = BertForMultipleChoice.from_pretrained(CFG['model']).to(device)
model = nn.DataParallel(model)

predictions = []

for fold in range(int(CFG['fold_num'])):  # 把训练后的五个模型挨个进行预测
    y_pred = []
    model.load_state_dict(torch.load('LargeMacKD256sub_{}_fold_{}.pt'.format(CFG['model'].split('/')[1], fold)))

    with torch.no_grad():
        tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), y.to(device).long()

            output = model(input_ids, attention_mask, token_type_ids)[0].cpu().numpy()

            y_pred.extend(output)

    predictions += [y_pred]

predictions = np.mean(predictions, 0).argmax(1)
sub = pd.DataFrame(columns=['id', 'label'])  # 提交
sub['label'] = predictions
sub['label'] = sub['label'].apply(lambda x: ['A', 'B', 'C', 'D'][x])

sub.to_csv('sub01.csv', index=False)
# np.mean(cv)
