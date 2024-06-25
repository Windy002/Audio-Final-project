import torch
import tensorboard
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from glob import glob
import numpy as np
from mynet import *
from python_speech_features import mfcc, delta # mfcc音频特征提取工具包
import scipy.io.wavfile as wav

data_path = 'FSDD'
waves = glob("{}/*.wav".format(data_path),recursive=True)
print("总数据数量: ",len(waves))

#打乱列表
random.shuffle(waves)
#划分数据集 train：8 test：2
train_waves = waves[:int(len(waves)*0.8)]
test_waves = waves[int(len(waves)*0.8):]
print("训练集数量: ",len(train_waves),"\n测试集数量: ",len(test_waves))


#MFCC特征提取
def get_mfcc(data, fs):
    # MFCC特征提取
    wav_feature = mfcc(data, fs)

    # 特征一阶差分
    d_mfcc_feat = delta(wav_feature, 1)
    # 特征二阶差分
    d_mfcc_feat2 = delta(wav_feature, 2)
    # 特征拼接
    feature = np.concatenate(
        [wav_feature.reshape(1, -1, 13), d_mfcc_feat.reshape(1, -1, 13), d_mfcc_feat2.reshape(1, -1, 13)], 0)

    # 对数据进行截取或者填充
    if feature.shape[1] > 64:
        feature = feature[:, :64, :]
    else:
        feature = np.pad(feature, ((0, 0), (0, 64 - feature.shape[1]), (0, 0)), 'constant')
    # 通道转置(HWC->CHW) 处理成类似图片的形式 方便后续用pytorch处理
    feature = feature.transpose((2, 0, 1))

    return feature


# 读取音频样例
# fs, signal = wav.read('FSDD/0_george_0.wav')
# # 特征提取
# feature = get_mfcc(signal, fs)
# print('特征形状(CHW):', feature.shape, type(feature))


#提取制作标签 对音频进行mfcc处理
def preproess(waves):
    datalist=[]
    lablelist=[]
    for w in tqdm(waves):
        lablelist.append([int(w[5])])
        fs, signal = wav.read(w)
        f = get_mfcc(signal, fs)
        datalist.append(f)
    return np.array(datalist),np.array(lablelist)

train_data,train_lable=preproess(train_waves)
test_data,test_lable=preproess(test_waves)

#组装数据集
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,audio,text):
        super(MyDataset, self).__init__()
        self.text = text
        self.audio = audio

    def __getitem__(self, index):
        return self.audio[index],self.text[index]

    def __len__(self):
        return self.audio.shape[0]


#加载数据集
train_dataset = MyDataset(train_data,train_lable)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)

test_dataset = MyDataset(test_data,test_lable)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True,drop_last=True)



#打印一批数据信息看看
# for data in train_loader:
#     audio, text = data
#     text = text.view(-1)
#     print(audio.shape)
#     print(text)
#     print(text.shape)
#     break




#初始化网络 利用GPU训练
model = Mynet()
model = model.cuda()

#设置模型训练参数
epochs = 200

#添加tensorboard
writer = SummaryWriter("logs_train")

#记录训练次数
total_train_step = 0
total_test_step = 0

#学习速率
lr_rate = 1e-3

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
optimizer = optim.AdamW(model.parameters(), lr=lr_rate)

#训练模型
for i in range(epochs):
    print("------------第{}轮训练开始----------".format(i + 1))

    # 训练步骤开始
    model.train()
    for data in train_loader:
        audio, text = data
        #调整数据以符合输入
        audio = audio.cuda()
        text = text.cuda()
        audio = audio.type(torch.float32)
        text = text.view(-1)
        text = text.type(torch.int64)

        #前向传播
        output = model(audio)
        loss = loss_fn(output, text)

        # 反向传播 调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 10 ==0 :
            print("训练次数:  {}, Loss:  {}".format(total_train_step, loss.item()))

            # 写入tensorboard
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            audio, text = data
            audio = audio.cuda()
            text = text.cuda()
            # 调整数据以符合输入
            audio = audio.type(torch.float32)
            text = text.view(-1)
            text = text.type(torch.int64)

            output = model(audio)

            loss = loss_fn(output, text)
            total_test_loss = total_test_loss + loss.item()


            accuray = (output.argmax(1) == text).sum()
            accuray = accuray.item()
            total_accuracy = total_accuracy + accuray

        print("整体测试集上的Loss: {}".format(total_test_loss))

        #写入tensorboard
        writer.add_scalar("test_loss",total_test_loss,total_test_step)

        print("整体测试集上的准确率: {}".format(total_accuracy / len(test_waves)))

        # 写入tensorboard
        writer.add_scalar("test_accuracy",total_accuracy / len(test_waves),total_test_step)

        total_test_step += 1

        # torch.save(model,"model_path/model_{}.path".format(i+1))
        # print("模型已保存")

writer.close()


















# for data in train_loader:
#     audio, text = data
#     audio = audio.type(torch.float32)
#     # 逐列归一化
#     audio_normalized = (audio - audio.min(dim=0)[0]) / (audio.max(dim=0)[0] - audio.min(dim=0)[0])
#
#     # 检查归一化结果
#     # print(audio_normalized)
#     # print(audio_normalized.min(), audio_normalized.max())
#     text = text.view(-1)
#     text = text.type(torch.int64)
#
#     # print('audio:',audio)
#     # print('audio.shape:',audio.shape)
#     # print('text:',text)
#     # print('text.shape',text.shape)
#     out = model(audio)
#     # print('out',out)
#     break


