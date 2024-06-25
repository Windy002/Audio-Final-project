import torch
from torch import nn
import torch.nn.functional as F

#模型1
# class Mynet(nn.Module):
#     def __init__(self):
#         super(Mynet,self).__init__()
#
#         self.module1 = nn.Sequential(
#                 nn.Conv2d(in_channels=13, out_channels=16, kernel_size=(3,3), stride=1, padding=1),
#                 nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,2), stride=(1,2), padding=(1,0)),
#                 nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
#                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,2), stride=(1,2), padding=(1,0)),
#                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
#                 nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,2), stride=2, padding=0),
#                 nn.Flatten(),
#                 nn.Linear(in_features=512, out_features=128),
#                 nn.Linear(in_features=128, out_features=10)
#         )
#
#     def forward(self,x):
#         x = self.module1(x)
#         return x


#模型2
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()

        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.module1(x)
        return x








# mynet = Mynet()
# #print(mynet)
#
# # 验证
# input = torch.ones((64,13,3,64))
# output = mynet(input)
# print(output.shape)