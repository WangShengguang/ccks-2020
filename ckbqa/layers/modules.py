from torch import nn

import torch


class TextCNN(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super(TextCNN).__init__()
        input_chanel = 1
        kernel_num = 1
        filter_sizes = [2, 3, 4]
        self.convs = [nn.Conv2d(input_chanel, kernel_num, (kernel_size, embed_dim))
                      for kernel_size in filter_sizes]
        # self.drop_out = nn.Dropout(0.8)
        self.fc = nn.Linear(len(filter_sizes) * kernel_num, out_dim)

    def forward(self, x):
        xs = []
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            xs.append(x)
        x = torch.cat(xs, dim=1)
        x = self.fc(x)
        return x
