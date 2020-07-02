import torch
import torch.nn as nn

in_dim_1 =15949

class module_hid(torch.nn.Module):
    def __init__(self):
        super(module_hid,self).__init__()
        input_dim = in_dim_1
        layer1 = 15
        layer2 = 15
        layer_out = 3
        self.layer1 = torch.nn.Sequential(
            nn.Linear(input_dim,layer1),nn.BatchNorm1d(layer1)
        )
        self.layer_h = torch.nn.Sequential(
            nn.Linear(layer1,layer2), nn.BatchNorm1d(layer1),nn.ReLU(True)
        )
        self.layer_out = torch.nn.Sequential(
            nn.Linear(layer1, layer_out),nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer_out(x)
        return x

class module_sim(torch.nn.Module):
    def __init__(self):
        super(module_sim,self).__init__()
        input_dim = in_dim_1
        l1_num = 300
        l2_num = 30
        out_num = 3
        self.layer_in = torch.nn.Sequential(
            nn.Linear(input_dim,l1_num),nn.BatchNorm1d(l1_num),nn.ReLU(True)
        )
        self.layer_hide_1 = torch.nn.Sequential(
            nn.Linear(l1_num,l2_num),nn.Dropout(p=0.5),nn.ReLU(True)
        )
        self.layer_out = torch.nn.Sequential(
            nn.Linear(l2_num,out_num),nn.ReLU(True)
        )
        self.lo = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.layer_in(x)
        x = self.layer_hide_1(x)
        x = self.layer_out(x)
        x = self.lo(x)
        return x

class module_mid(torch.nn.Module):
    def __init__(self):
        super(module_mid,self).__init__()
        input_dim = in_dim_1
        layer1 = 300
        layer2 = 60
        layer3 = 60
        layer4 = 60
        layer5 = 15
        layer6 = 15
        layer7 = 15
        layer8 = 3
        self.layer1 = torch.nn.Sequential(
            nn.Linear(input_dim,layer1),nn.BatchNorm1d(layer1),nn.ReLU(True)
        )
        self.layer2 = torch.nn.Sequential(
            nn.Linear( layer1,layer2),nn.BatchNorm1d(layer2),nn.ReLU(True)
        )
        self.layer3 = torch.nn.Sequential(
            nn.Linear(layer2, layer3), nn.BatchNorm1d(layer3),nn.ReLU(True)
        )
        self.layer4 = torch.nn.Sequential(
            nn.Linear(layer3, layer4),nn.BatchNorm1d(layer4), nn.ReLU(True)
        )
        self.layer5 = torch.nn.Sequential(
            nn.Linear(layer4, layer5),nn.BatchNorm1d(layer5), nn.ReLU(True)
        )
        self.layer6 = torch.nn.Sequential(
            nn.Linear(layer5, layer6), nn.BatchNorm1d(layer6),nn.ReLU(True)
        )
        self.layer7 = torch.nn.Sequential(
            nn.Linear(layer6, layer7), nn.BatchNorm1d(layer7), nn.ReLU(True)
        )
        self.layer8 = torch.nn.Sequential(
            nn.Linear(layer7, layer8), nn.ReLU(True), nn.Softmax(dim=1)
        )


    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        # x = self.layer_out(x)
        return x
class module_expend(torch.nn.Module):
    def __init__(self):
        super(module_expend,self).__init__()
        input_dim = in_dim_1
        layer1 = 1500
        layer2 = 150
        layer3 = 15
        layer4 = 15
        layer5 = 3
        self.layer1 = torch.nn.Sequential(
            nn.Linear(input_dim,layer1),nn.BatchNorm1d(layer1),nn.ReLU(True)
        )
        self.layer2 = torch.nn.Sequential(
            nn.Linear( layer1,layer2),nn.BatchNorm1d(layer2),nn.ReLU(True)
        )
        self.layer3 = torch.nn.Sequential(
            nn.Linear(layer2, layer3), nn.BatchNorm1d(layer3),nn.ReLU(True)
        )
        self.layer4 = torch.nn.Sequential(
            nn.Linear(layer3, layer4),nn.BatchNorm1d(layer4), nn.ReLU(True)
        )
        self.layer5 = torch.nn.Sequential(
            nn.Linear(layer4, layer5),nn.BatchNorm1d(layer5), nn.ReLU(True)
        )
        self.layer_out = torch.nn.Sequential(
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer_out(x)
        return x
