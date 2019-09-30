import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block_2D(nn.Module):
    """Block of 2D convolutions with kernel size 1x1 and ReLU activations
    """

    def __init__(self, input_channels, output_channels, layers):
        super().__init__()

        conv_block = []
        conv_block.append(nn.Conv2d(input_channels, output_channels,
                                    kernel_size=1))
        conv_block.append(nn.ReLU())

        for i in range(layers-1):
            conv_block.append(nn.Conv2d(output_channels,
                                        output_channels,
                                        kernel_size=1))
            conv_block.append(nn.ReLU())

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class pred_network(nn.Module):
    """Block of 1D convolutions with kernel size 1 and ReLU activations. Last
       convolution reduces the number of channels to 1 for predictions.
    """

    def __init__(self, channels, layers):
        super().__init__()

        conv_block = []
        for i in range(layers-1):
            conv_block.append(nn.Conv1d(channels, channels, kernel_size=1))
            conv_block.append(nn.ReLU())

        conv_block.append(nn.Conv1d(channels, 1, kernel_size=1))

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class pairs_of_pairs(nn.Module):
    """This block generates a pair matrix, concatenates previous networks
       outputs and processes the matrix with a conv_block_2D
    """

    def __init__(self, output_channels, layers, device):
        super().__init__()

        self.device = device
        self.conv_block = conv_block_2D(output_channels,
                                        output_channels, layers)

    def forward(self, x, x_concat, batch_size, size):

        input_channels = x.shape[1]
        x = x.unsqueeze(2).repeat(1, 1, size-1, 1)

        # create empty tensors
        pair_matrix = torch.empty(batch_size, 2*input_channels,
                             size-1, size).to(self.device)
        roll = torch.empty(batch_size, input_channels,
                           size-1, size, dtype=torch.long).to(self.device)

        # create roll matrix
        for d in range(size-1):
            roll[:, :, d, :] = torch.arange(size).roll(d+1)

        # use roll matrix to create pair matrix
        pair_matrix[:, :input_channels, :, :] = x
        pair_matrix[:, input_channels:, :, :] = torch.gather(x, 3, roll)

        # concatenate previous network outputs
        pairs_cat = torch.cat((*x_concat, pair_matrix), dim=1)

        # convolutional block
        pairs_cat = self.conv_block(pairs_cat)

        return pairs_cat


class Network(nn.Module):
    """Model to process single and pair features to predicted scc. The number
       pair blocks, base channels and layers can be selected.
    """

    def __init__(self, pair_blocks, base_channels, layers, device):

        super().__init__()

        self.device = device
        self.pair_blocks = pair_blocks

        # first pair block
        self.pair_block_1 = conv_block_2D(22, base_channels, layers)

        # all other pair blocks
        channels = [22, base_channels]
        for index in range(2, pair_blocks+1):
            channels.append(2*channels[-1] + sum(channels[:-1]))
            setattr(self, 'pair_block_{:d}'.format(index),
                    pairs_of_pairs(channels[-1], layers, device))

        # scc predictions
        self.pred = pred_network(sum(channels[1:])+9, layers)

        self.apply(init_params)
        self.to(device)

    def forward(self, single, pairs, batch_size, size):

        # first pair block
        pair_list = [pairs, self.pair_block_1(pairs)]

        # all other pair blocks
        for index in range(2, self.pair_blocks+1):
            pair_block = getattr(self, 'pair_block_{:d}'.format(index))
            pair_list.append(pair_block(pair_list[-1].sum(dim=2),
                                        pair_list[:-1], batch_size, size))

        # scc predictions
        cat_list = [single]
        cat_list.extend([pair_list[index].sum(dim=2)
                        for index in range(1, self.pair_blocks+1)])
        cat = torch.cat(cat_list, dim=1)
        pred = self.pred(cat)

        return pred.view(batch_size, size)


def init_params(layer):

    if type(layer) == nn.Conv1d or type(layer) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
