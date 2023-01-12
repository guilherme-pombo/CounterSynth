import torch.nn as nn
import torch


def weights_init_normal(model):
    '''
    More of a stable init for this problem than the default Pytorch init
    :param model: the model to initialise
    :return:
    '''
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)


def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
    if maxpool is True:
        layer = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
            nn.BatchNorm3d(out_channel),
            nn.MaxPool3d(2, stride=maxpool_stride),
            nn.ReLU(),
        )
    else:
        layer = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
            nn.BatchNorm3d(out_channel),
            nn.ReLU()
        )
    return layer


class SFCNDiscriminator(nn.Module):

    def __init__(self, in_channels, channel_number=[32, 64, 128, 256, 64]):
        '''
        This is the model definition for the discriminator. It borrows a lot from the model of the paper
        "Accurate brain age prediction with lightweight deep neural networks"
        :param in_channels: The number of modalities that are input into the discriminator
        :param channel_number: Convolutional kernels to use throughout the network
        '''

        super(SFCNDiscriminator, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = in_channels
            else:
                in_channel = channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(f'conv_{i}',
                                                  conv_layer(in_channel,
                                                             out_channel,
                                                             maxpool=True,
                                                             kernel_size=3,
                                                             padding=1))
            else:
                self.feature_extractor.add_module(f'conv_{i}',
                                                  conv_layer(in_channel,
                                                             out_channel,
                                                             maxpool=False,
                                                             kernel_size=1,
                                                             padding=0))

        in_channel = channel_number[-1]
        # Output for classification task - Sex
        self.classifier_cls = nn.Sequential()
        self.classifier_cls.add_module(f'conv_{n_layer}',
                                       nn.Conv3d(in_channel, 1, kernel_size=2, padding=0, bias=False))

        # Output for regression task - Age
        self.classifier_reg = nn.Sequential()
        self.classifier_reg.add_module(f'conv_{n_layer + 1}',
                                       nn.Conv3d(in_channel, 1, kernel_size=2, padding=0, bias=False))

        # Output for adversarial task
        self.classifier_adv = nn.Sequential()
        self.classifier_adv.add_module(f'conv_{n_layer + 2}',
                                       nn.Conv3d(in_channel, 1, kernel_size=2, padding=0, bias=False))

    def forward(self, x):
        encoded_features = self.feature_extractor(x)

        # Features are shared for all tasks
        class_preds = self.classifier_cls(encoded_features)
        reg_preds = self.classifier_reg(encoded_features)
        adversarial_preds = self.classifier_adv(encoded_features)

        class_preds = class_preds.view(x.size(0), 1)
        reg_preds = reg_preds.view(x.size(0), 1)
        adversarial_preds = adversarial_preds.view(x.size(0), 1)
        return adversarial_preds, class_preds, reg_preds
