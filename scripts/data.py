import os
import math
import numpy as np
import random
import pandas as pd
import torch
from torch.distributions.categorical import Categorical
import data_prep

class Loader:
    """Base class for all loaders
    """

    def __init__(self, N_max, n_per_conn, path, device):
        """Initializes the train loader
        """

        # store parameters
        self.N_max = N_max
        self.n_per_conn = n_per_conn
        self.path = path
        self.device = device

    def single_features(self, data, device='cpu'):
        """Calculates distance and returns Tensor containing types and
           distances
        """

        data = data.to(device)
        distances = (data[:, 8:11, :] - data[:, 11:14, :])**2
        distances = distances.sum(dim=1, keepdim=True).sqrt()

        return torch.cat((data[:, :8, :], distances), dim=1)

    def pair_features(self, data, batch_size, size, device='cpu'):
        """Computes pair feature matrix of distances and types
        """

        # move to device
        data = data.to(device)

        # get positions and types
        pos0 = data[:, 8:11, :].unsqueeze(2).repeat(1, 1, size-1, 1)
        pos1 = data[:, 11:14, :].unsqueeze(2).repeat(1, 1, size-1, 1)
        types = data[:, :8, :].unsqueeze(2).repeat(1, 1, size-1, 1)

        # create empty tensors
        pairs = torch.empty(batch_size, 22, size-1, size, device=device)
        roll = torch.empty(batch_size, 1, size-1, size,
                           dtype=torch.long, device=device)

        # create roll matrix
        for d in range(size-1):
            roll[:, :, d, :] = torch.arange(size).roll(d+1)

        # types
        pairs[:, :8, :, :] = types
        pairs[:, 8:16, :, :] = torch.gather(types, 3, roll.repeat(1,8,1,1))

        roll = roll.repeat(1,3,1,1)
        # connection distances
        pairs[:, 16, :, :] = ((pos0 - pos1)**2).sum(dim=1).sqrt()
        pairs[:, 17, :, :] = ((torch.gather(pos0, 3, roll)
                            -torch.gather(pos1, 3, roll))**2).sum(dim=1).sqrt()

        # distances inbetween connections
        pairs[:, 18, :, :] = ((
            (pos0-torch.gather(pos0, 3, roll))**2).sum(dim=1).sqrt())
        pairs[:, 19, :, :] = ((
            (pos0 - torch.gather(pos1, 3, roll))**2).sum(dim=1).sqrt())
        pairs[:, 20, :, :] = ((
            (pos1 - torch.gather(pos0, 3, roll))**2).sum(dim=1).sqrt())
        pairs[:, 21, :, :] = ((
            (pos1 - torch.gather(pos1, 3, roll))**2).sum(dim=1).sqrt())

        return pairs


class Train_loader(Loader):
    """Loads one molecule at a time and shuffles data from epoch to epoch
    """

    def __init__(self, *args):
        super().__init__(*args)

        self.iteration = 0

    def set_iterations(self, iterations):
        self.max_iteration = iterations + self.iteration

    def __iter__(self):
        return self

    def __next__(self):
        """Gets next molecule
        """

        if self.iteration < self.max_iteration:

            # generate random index list
            if self.iteration % len(self.size) == 0:
                self.index_list = torch.randperm(len(self.size))

            # get molecule from index list
            index = self.index_list[self.iteration % len(self.size)]
            size = self.size[index]
            data = self.data[index][0:size*self.n_per_conn]
            data = torch.transpose(data.view(1, size, -1), 1, 2)
            scc = self.scc[index][0:size]
            type_data = self.type[index][0:size]

            # calculate distances
            single = self.single_features(data)
            pairs = self.pair_features(data, 1, size)

            # standardize distances in connections
            single[:, 8, :] -= self.dist[0, 0]
            single[:, 8, :] /= self.dist[1, 0]
            pairs[:, 16:18, :, :] -= self.dist[0, 0]
            pairs[:, 16:18, :, :] /= self.dist[1, 0]
            # standardize distances in interconnections
            pairs[:, 18:22, :, :] -= self.dist[0, 1]
            pairs[:, 18:22, :, :] /= self.dist[1, 1]

            # move to device
            scc = scc.to(self.device)
            type_data = type_data.to(self.device)
            single = single.to(self.device)
            pairs = pairs.to(self.device)

            # increment iteration
            self.iteration += 1

            return (self.iteration, single, pairs, scc.view(1, -1),
                    type_data.view(1, -1), size)
        else:
            raise StopIteration

    def preprocess_data(self):
        """Preprocesses data from csv files and saves to files
        """

        print("Preprocessing train data from csv files ...", end=' ')

        data = pd.read_csv(os.path.join(self.path, 'train.csv'))
        structures = pd.read_csv(os.path.join(self.path, 'structures.csv'))

        data = data_prep.merge_data_with_structures(data, structures)
        del structures

        # drop unnecessary information
        data.drop(columns=['atom_index_0', 'atom_index_1'], inplace=True)
        data.drop(columns=['id'], inplace=True)

        # extract ssc and drop from data
        scc = data_prep.extract_scc(data, self.N_max)
        data.drop(columns=['scalar_coupling_constant'], inplace=True)

        # one hot encodes type
        data, type_data = data_prep.encode_and_extract_type(data, self.N_max)

        # write conns in one line and convert to Tensor
        data = data_prep.write_conns_in_one_line(data,
                                                 self.N_max,
                                                 self.n_per_conn)

        # sort and save to files
        data_prep.standardize_and_save_train(data, scc, type_data, self.N_max,
                                                  self.n_per_conn, self.path)

        self._calculate_dist_mean_std()

        print("Done!")

    def _calculate_dist_mean_std(self):
        """Calculates mean and standard deviation of distances
        """

        self.data = torch.load(os.path.join(self.path, 'train.pt'))
        self.size = torch.load(os.path.join(self.path, 'size.pt'))
        num_molecules = len(self.data)

        # calculate mean
        single_mean = 0
        single_num = 0
        pairs_mean = 0
        pairs_num = 0

        for index in range(num_molecules):

            # get data
            size = self.size[index]
            data = self.data[index][0:size*self.n_per_conn]
            data = torch.transpose(data.view(1, size, -1), 1, 2)

            # calculate distances
            single = self.single_features(data)
            pairs = self.pair_features(data, 1, size)

            # count
            single_mean += single[:, 8, :].sum().item()
            single_num += size.item()
            pairs_mean += pairs[:, 18:22, :, :].sum().item()
            pairs_num += 4*size.item()*(size.item()-1)

        # divide by number to get mean
        single_mean /= single_num
        pairs_mean /= pairs_num

        # calculate std
        single_var = 0
        pairs_var = 0

        for index in range(num_molecules):

            # get data
            size = self.size[index]
            data = self.data[index][0:size*self.n_per_conn]
            data = torch.transpose(data.view(1, size, -1), 1, 2)

            # calculate distances
            single = self.single_features(data)
            pairs = self.pair_features(data, 1, size)

            # count
            single_var += ((single[:, 8, :]-single_mean)**2).sum().item()
            pairs_var += ((pairs[:, 18:22, :, :]-pairs_mean)**2).sum().item()

        # divide by number to get variance
        single_var /= single_num
        pairs_var /= pairs_num

        # take sqrt to get std
        single_std = math.sqrt(single_var)
        pairs_std = math.sqrt(pairs_var)

        # save mean and std of scc
        np.savetxt(os.path.join(self.path, 'dist_mean_std.csv'),
                np.array([[single_mean, pairs_mean], [single_std, pairs_std]]),
                delimiter=',')

    def load_data(self):
        """Loads data from files
        """

        self.data = torch.load(os.path.join(self.path, 'train.pt'))
        self.scc = torch.load(os.path.join(self.path, 'scc.pt'))
        self.type = torch.load(os.path.join(self.path, 'train_type.pt'))
        self.size = torch.load(os.path.join(self.path, 'size.pt'))
        self.dist = np.loadtxt(os.path.join(self.path, 'dist_mean_std.csv'),
                               delimiter=',')


class Test_loader(Loader):
    """Loads molecules in batches from test set
    """

    def __init__(self, *args):
        super().__init__(*args)

    def __iter__(self):

        # set counters to 0
        self.size = 0
        self.index = 0

        return self

    def __next__(self):
        """Gets next batch
        """

        # find size and batch size
        while True:
            if self.size < len(self.data):
                batch_size = min(self.batch_size,
                                 len(self.id[self.size])-self.index)
                if batch_size > 0:
                    break
                else:
                    self.index = 0
                    self.size += 1
                    print(self.size+1, end=' ')
            else:
                raise StopIteration

        # get size and index slice
        size = len(self.id[self.size][0])
        index_slice = slice(self.index, self.index+batch_size, 1)

        # get data
        data = self.data[self.size][index_slice]
        data = torch.transpose(data.view(batch_size, size, -1), 1, 2)
        type_data = self.type[self.size][index_slice].to(self.device)
        id_data = self.id[self.size][index_slice]

        # calculate distances
        single = self.single_features(data, device=self.device)
        pairs = self.pair_features(data, batch_size, size, device=self.device)

        # standardize distances in connections
        single[:, 8, :] -= self.dist[0, 0]
        single[:, 8, :] /= self.dist[1, 0]
        pairs[:, 16:18, :, :] -= self.dist[0, 0]
        pairs[:, 16:18, :, :] /= self.dist[1, 0]
        # standardize distances in interconnections
        pairs[:, 18:22, :, :] -= self.dist[0, 1]
        pairs[:, 18:22, :, :] /= self.dist[1, 1]

        # increment by batch size
        self.index += batch_size

        return single, pairs, type_data, id_data, batch_size, size

    def preprocess_data(self):
        """Preprocesses data from csv files and saves to files
        """

        print("Preprocessing test data from csv files ...", end=' ')

        data = pd.read_csv(os.path.join(self.path, 'test.csv'))
        structures = pd.read_csv(os.path.join(self.path, 'structures.csv'))

        data = data_prep.merge_data_with_structures(data, structures)
        del structures

        # drop unnecessary information
        data.drop(columns=['atom_index_0', 'atom_index_1'], inplace=True)

        # extract ids and drop them
        id_data = data_prep.get_ids(data, self.N_max)
        data.drop(columns=['id'], inplace=True)

        # one hot encodes type
        data, type_data = data_prep.encode_and_extract_type(data, self.N_max)

        # write conns in one line and convert to Tensor
        data = data_prep.write_conns_in_one_line(data,
                                                 self.N_max,
                                                 self.n_per_conn)

        # sort and save to files
        data_prep.sort_and_save_test(data, type_data, id_data, self.N_max,
                                            self.n_per_conn, self.path)

        print("Done!")

    def load_data(self):
        """Loads data from files
        """

        self.data = torch.load(os.path.join(self.path, 'test.pt'))
        self.type = torch.load(os.path.join(self.path, 'test_type.pt'))
        self.id = torch.load(os.path.join(self.path, 'id.pt'))
        self.dist = np.loadtxt(os.path.join(self.path, 'dist_mean_std.csv'),
                               delimiter=',')
