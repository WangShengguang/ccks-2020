import torch


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.relu(self.margin - euclidean_distance), 2)) / 2

        return loss_contrastive

