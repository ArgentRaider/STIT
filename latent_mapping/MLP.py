import torch
from torch import nn
from training.networks import FullyConnectedLayer

class mlp_18(nn.Module):
    def __init__(self):
        super(mlp_18, self).__init__()

        self.activation = 'lrelu'
        self.channel_num = 18
        # self.channel_fc_dim = [512, 128, 32, 8]
        self.channel_fc_dim = [512, 128, 64, 32, 8]

        # self.fc_dim = [ self.channel_num*self.channel_fc_dim[-1], 72, 50]
        self.fc_dim = [ self.channel_num*self.channel_fc_dim[-1], 72, 50, 50]
        self.output_dim = 50

        for ci in range(self.channel_num):
            for fi in range(len(self.channel_fc_dim)-1):
                fc_layer = FullyConnectedLayer(self.channel_fc_dim[fi], self.channel_fc_dim[fi+1], activation=self.activation)
                setattr(self, f'fc_{fi}_channel_{ci}', fc_layer)
        
        for fi in range( len(self.fc_dim)-1 ):
            fc_layer = FullyConnectedLayer(self.fc_dim[fi], self.fc_dim[fi+1], activation=self.activation)
            setattr(self, f'fc_{fi}', fc_layer)
        
        fc_layer = FullyConnectedLayer(self.fc_dim[-1], self.output_dim)
        setattr(self, 'output_fc', fc_layer)
        
    def forward(self, w):
        x = []
        for ci in range(self.channel_num):
            x_ci = w[:, ci]
            for fi in range(len(self.channel_fc_dim)-1):
                fc_layer = getattr(self, f'fc_{fi}_channel_{ci}')
                x_ci = fc_layer(x_ci)
            x.append(x_ci)
        x = torch.cat(x, dim=1)

        for fi in range(len(self.fc_dim)-1):
            fc_layer = getattr(self, f'fc_{fi}')
            x = fc_layer(x)
        
        output_fc_layer = getattr(self, 'output_fc')
        x = output_fc_layer(x)

        return x
