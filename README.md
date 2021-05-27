# 海华中文阅读理解比赛
​                                                                                     **队名：ATTOY** 
​                                                                                     **排名：第三名** 

## 赛题背景
https://www.biendata.xyz/competition/haihua_2021

文字是人类用以记录和表达的最基本工具，也是信息传播的重要媒介。透过文字与符号，我们可以追寻人类文明的起源，可以传播知识与经验，读懂文字是认识与了解的第一步。对于人工智能而言，它的核心问题之一就是认知，而认知的核心则是语义理解。
 
机器阅读理解(Machine Reading Comprehension)是自然语言处理和人工智能领域的前沿课题，对于使机器拥有认知能力、提升机器智能水平具有重要价值，拥有广阔的应用前景。机器的阅读理解是让机器阅读文本，然后回答与阅读内容相关的问题，体现的是人工智能对文本信息获取、理解和挖掘的能力，在对话、搜索、问答、同声传译等领域，机器阅读理解可以产生的现实价值正在日益凸显，长远的目标则是能够为各行各业提供解决方案。
 
《2021海华AI挑战赛·中文阅读理解》大赛由中关村海华信息技术前沿研究院与清华大学交叉信息研究院联合主办，腾讯云计算协办。共设置题库16000条数据，总奖金池30万元，且腾讯云计算为中学组赛道提供独家算力资源支持。
 
本次比赛的数据来自小学/中高考语文阅读理解题库（其中，技术组的数据主要为中高考语文试题，中学组的数据主要来自小学语文试题）。相较于英文，中文阅读理解有着更多的歧义性和多义性，然而璀璨的中华文明得以绵延数千年，离不开每一个时代里努力钻研、坚守传承的人，这也正是本次大赛的魅力与挑战，让机器读懂文字，让机器学习文明。秉承着人才培养的初心，我们继续保留针对中学组以及技术组的两条平行赛道，科技创新，时代有我，期待你们的回响。
 
##比赛任务

本次比赛技术组的数据来自中高考语文阅读理解题库。每条数据都包括一篇文章，至少一个问题和多个候选选项。参赛选手需要搭建模型，从候选选项中选出正确的一个。
 
## 2021海华AI挑战赛·中文阅读理解·技术组 第三名（ATTOY团队）解决方案

## 算法方案
### 1.预训练模型：MacBERT-Large
### 2.对抗训练
FreeLB [ICLR 2020]
### 3.知识蒸馏
Born Agai nNeural Networks [ICML 2018]

[ICLR 2020]: https://openreview.net/forum?id=BygzbyHFvB
[ICML 2018]: https://openreview.net/forum?id=H1EwisW_-r

## 环境要求
tqdm==4.50.2
numpy==1.19.2
pandas==1.1.3
transformers==3.5.1
torch==1.7.0+cu110
scikit_learn==0.24.2

## 运行方法
bash bash.sh

## 超参数
### FreeLB训练参数配置
    'fold_num': 4, 
    'seed': 42,
    'model': 'hfl/chinese-macbert-large', 
    'max_len': 512, 
    'epochs': 12,
    'train_bs': 4, 
    'valid_bs': 4,
    'lr': 2e-5,  
    'lrSelf': 1e-4,  
    'accum_iter': 8, 
    'weight_decay': 1e-4, 
    'adv_lr': 0.01,
    'adv_norm_type': 'l2',
    'adv_init_mag': 0.03,
    'adv_max_norm': 1.0,
    'ip': 2
### EKD训练参数配置
    'fold_num': 4, 
    'seed': 42,
    'model': 'hfl/chinese-macbert-large', 
    'max_len': 256, 
    'epochs': 12,
    'train_bs': 4, 
    'valid_bs': 4,
    'lr': 2e-5,  
    'lrSelf': 1e-4,  
    'accum_iter': 8, 
    'weight_decay': 1e-4, 
    'adv_lr': 0.01,
    'adv_norm_type': 'l2',
    'adv_init_mag': 0.03,
    'adv_max_norm': 1.0,
    'ip': 2
