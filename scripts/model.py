import time
import datetime
import math
import os
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.modules as modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from network import Network
import data
import utils


class Model:

    def __init__(self, data_root='..', networks_root='..', device='cuda'):
        """Initializes the model

        Args:
            data_root (str, optional): data root. Defaults to '..'.
            networks_root (str, optional): neworks root. Defaults to '..'.
            device (str, optional): device to work with. Defaults to 'cuda'.
        """

        # Set parameters
        self.N_max = 135 # maximum number of connections
        self.n_per_conn = 14 # 6 positions and 8 types
        self.data_path = os.path.join(data_root, 'data')
        self.networks_path = os.path.join(networks_root, 'networks')

        # Set device
        self.device = device
        if not torch.cuda.is_available() and device == 'cuda':
            print("Cuda not available")

    def _preprocess_train_data(self):

        self.train_loader = data.Train_loader(self.N_max,
                                              self.n_per_conn,
                                              self.data_path,
                                              self.device)
        self.train_loader.preprocess_data()

    def _preprocess_test_data(self):

        self.test_loader = data.Test_loader(self.N_max,
                                            self.n_per_conn,
                                            self.data_path,
                                            self.device)

        self.test_loader.preprocess_data()

    def preprocess_data(self):
        """Preprocesses both train and test data from csv files
        """

        self._preprocess_train_data()
        self._preprocess_test_data()

    def _load_train_data(self):
        """Set ups train loader and loads data
        """

        self.train_loader = data.Train_loader(self.N_max, self.n_per_conn,
                                              self.data_path, self.device)
        self.train_loader.load_data()

        # load mean and std
        scc_mean_std = np.loadtxt(
            os.path.join(self.data_path, 'scc_mean_std.csv'), delimiter=',')
        self.mean = torch.Tensor(scc_mean_std[0])
        self.std = torch.Tensor(scc_mean_std[1])

    def _load_test_data(self):
        """Set ups test loader and loads data
        """

        self.test_loader = data.Test_loader(self.N_max, self.n_per_conn,
                                            self.data_path, self.device)

        self.test_loader.load_data()

        # load mean and std from train
        scc_mean_std = np.loadtxt(
            os.path.join(self.data_path, 'scc_mean_std.csv'), delimiter=',')
        self.mean = torch.Tensor(scc_mean_std[0])
        self.std = torch.Tensor(scc_mean_std[1])

    def load_data(self):
        """Loads both test and train data
        """

        self._load_train_data()
        self._load_test_data()

    def set_network(self, pair_blocks=1, base_channels=512, layers=5):
        """Sets and randomly initializes the network
        """

        # store architecture
        self.pair_blocks = pair_blocks
        self.base_channels = base_channels
        self.layers = layers

        self.net = Network(pair_blocks, base_channels, layers, self.device)
        self.train_loader.index = 0

        self._loaded = False
        self.time_stamp_path = None

    def _set_optimizer(self):
        """Sets optimizer according to name of optimizer
        """

        if self.optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(),
                                        lr=self.learning_rate,
                                        betas=self.betas,
                                        eps=1e-8,
                                        weight_decay=self.weight_decay)
        elif self.optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(),
                                       lr=self.learning_rate,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay)
        elif self.optimizer_name == 'SGD_Nesterov':
            self.optimizer = optim.SGD(self.net.parameters(),
                                       lr=self.learning_rate,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay,
                                       nesterov=True)
        elif self.optimizer_name == 'RMSprop':
            self.optimizer = optim.Adagrad(self.net.parameters(),
                                           lr=self.learning_rate,
                                           momentum=self.momentum,
                                           weight_decay=self.weight_decay)
        elif self.optimizer_name == 'Adagrad':
            self.optimizer = optim.Adagrad(self.net.parameters(),
                                           lr=self.learning_rate,
                                           weight_decay=self.weight_decay)
        else:
            print("Optimizer '" + self.optimizer_name + "' not implemented.")

    def load(self, time_stamp):
        """Loads parameters from file with fitting time_stamp
        """

        self.time_stamp_path = os.path.join(self.networks_path,
                                            time_stamp)
        # load parameters
        self.net.load_state_dict(torch.load(os.path.join(self.time_stamp_path,
                                                         'params.pt')))

        # load index and index list
        index_ = torch.load(os.path.join(self.time_stamp_path, 'index.pt'))
        self.train_loader.iteration = index_[0]
        self.train_loader.index_list = index_[1]

        # load loss list
        loss_list = np.loadtxt(os.path.join(self.time_stamp_path, 'loss.csv'),
                               delimiter=', ')
        self._loss_list = list(loss_list[:, 1])

        # load best loss
        self.loss_best = np.load(os.path.join(self.time_stamp_path,
                                              'loss_best.npy'))

        self._loaded = True

    def _save(self, iteration):
        """Saves to file
        """

        # save parameters
        torch.save(self.net.state_dict(), os.path.join(self.time_stamp_path,
                                                       'params.pt'))

        # save index and index_list
        index_ = [self.train_loader.iteration, self.train_loader.index_list]
        torch.save(index_, os.path.join(self.time_stamp_path, 'index.pt'))

        # save history
        dt = time.time() - self._time_start_save
        mean, std = self._loss_std_mean(self.save_every)
        history_string = "{:d}, {:.2e}, {:.2e}, {:.2f}\n".format(
            iteration, mean, std, dt)
        with open(os.path.join(self.time_stamp_path,
                               'save_history.csv'), "a") as f:
            f.write(history_string)
        self._time_start_save = time.time()

        # save loss
        with open(os.path.join(self.time_stamp_path,
                               'loss.csv'), "a") as f:

            for itr in range(iteration-self.save_every, iteration):
                f.write("{:d}, {:.2e}\n".format(itr+1, self._loss_list[itr]))

        # if best, save
        if mean < self.loss_best:
            shutil.copyfile(os.path.join(self.time_stamp_path, 'params.pt'),
                            os.path.join(self.time_stamp_path,
                                         'params_best.pt'))
            self.loss_best = mean
            np.save(os.path.join(self.time_stamp_path, 'loss_best.npy'),
                                 np.array(self.loss_best))

    def _save_hyper(self):
        """Saves hyper parameters
        """

        file_name = os.path.join(self.time_stamp_path, 'hyper.txt')

        # architecture
        if os.path.isfile(file_name) is False:
            with open(file_name, 'w') as f:
                f.write('architecture:\n')
                f.write('    pair blocks: {:d}\n'.format(self.pair_blocks))
                f.write('    base channels: {:d}\n'.format(self.base_channels))
                f.write('    layers: {:d}\n'.format(self.layers))

        # training
        with open(file_name, 'a') as f:
            f.write('training starting at iteration {:d}:\n'.format(
                                                self.train_loader.iteration+1))
            f.write('    optimizer: ' + self.optimizer_name + '\n')
            f.write('    learning_rate: {:.2e}\n'.format(self.learning_rate))
            f.write('    weight_decay: {:.2e}\n'.format(self.weight_decay))
            if self.optimizer_name in ['SGD', 'SGD_Nesterov', 'RMSprop']:
                f.write('    momentum: {:.5e}\n'.format(self.momentum))
            if self.optimizer_name == 'Adam':
                f.write('    betas: {:.5e}, {:.5e}\n'.format(*self.betas))

    def train(self, iterations=100, optimizer='Adam', learning_rate=1e-4,
              weight_decay=0, momentum=0, betas=(0.9, 0.999), save_name=None,
              save_every=None, print_every=10):
        """Trains the network

        Args:
            iterations (int, optional): Number of iterations. Defaults to 10.
            optimizer (str, optional): 'Adam', 'SGD', 'SGD_Nesterov', 'RMSprop'
                                        or 'Adagrad'. Defaults to 'Adam'.
            learning_rate (float, optional): Learning rate. Defaults to 1e-4.
            weight_decay (float, optional): Regularization parameter.
                                            Defaults to 0.
            momentum (float, optional): Momentum of 'SGD', 'SGD_Nesterov'
                                        or 'RMSprop'. Defaults to 0.
            betas (tuple of floats, optional): Betas for Adam.
                                               Defaults to (0.9, 0.999).
            save_every (int, optional): Saves every specified iteration.
                                        Defaults to None.
            save_name (str, optional): String added to time_stamp.
                                       Defaults to None.
            print_every (int, optional): Prints every specified iteration.
                                         Defaults to None.
        """

        # Store hyper parameters
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.betas = betas
        self.print_every = print_every
        self.save_every = save_every

        # reset if not loaded
        if self._loaded is False:
            self.train_loader.iteration = 0
            self.loss_best = float('inf')
            self._loss_list = []
            self._loaded = True

        # create new time stamp and folder if necessary
        if save_every is not None and self.time_stamp_path is None:
            time_stamp = utils.get_time_stamp()
            if save_name is not None:
                time_stamp = time_stamp + '_' + save_name
            print("timestamp: " + time_stamp + '\n')
            self.time_stamp_path = os.path.join(self.networks_path,
                                                time_stamp)
            os.mkdir(self.time_stamp_path)

        # save hyper parameters
        if save_every is not None:
            self._save_hyper()

        # pass number of iterations to train loader
        self.train_loader.set_iterations(iterations)

        # set optimizer and loss
        self._set_optimizer()
        self.loss = modules.MSELoss()

        # set timers
        time_start_total = time.time()
        self._time_start_print = time.time()
        self._time_start_save = time.time()

        self.net.train(mode=True)
        # training loop
        for _, (iteration, single, pairs,
                scc, type_data, size) in enumerate(self.train_loader):

            self.optimizer.zero_grad()
            pred = self.net(single, pairs, 1, size)
            loss = self.loss(pred, scc)
            loss.backward()
            self.optimizer.step()

            self._loss_list.append(loss.item())

            # print message
            if print_every is not None and (iteration % print_every == 0):
                self._print_loss(iteration)

            # save to file
            if save_every is not None and (iteration % save_every == 0):
                self._save(iteration)

        # print total time
        dt = time.time()-time_start_total
        print("\ntotal time: " + utils.get_formatted_time(dt))

    def test(self, max_batch_size=16):
        """Creates predictions for submission by processing batches of
           molecules of the same size

        Args:
            max_batch_size (int): maximum batch size
        """

        print("Creating predictions for submission ...")
        time_start_total = time.time()

        # pass batch size to test loader
        self.test_loader.batch_size = max_batch_size

        pred_list = []
        id_list = []

        self.net.train(mode=False)
        # test loop
        for _, (single, pairs, type_data, id_data,
                batch_size, size) in enumerate(self.test_loader):

            # get prediction
            with torch.no_grad():
                pred = self.net(single, pairs, batch_size, size)

            # multiply with std and add mean from train set
            for i in range(8):
                pred[type_data==i] *= self.std[i]
                pred[type_data==i] += self.mean[i]

            pred_list.append(pred.cpu().flatten())
            id_list.append(id_data.flatten())

        self._create_submission_file(pred_list, id_list)

        # print total time
        dt = time.time()-time_start_total
        print("\ntotal time: "+utils.get_formatted_time(dt))

    def _loss_std_mean(self, iterations):
        """Returns mean and standard deviation of the last specified number
           of iterations
        """

        loss_array = np.array(self._loss_list[-iterations:])
        return loss_array.mean(), loss_array.std()

    def _print_loss(self, iteration):
        """Prints iteration, loss mean, loss std, elapsed time since last print
        """

        dt = time.time() - self._time_start_print
        loss_string = "iteration {:d}".format(iteration)
        loss_string += ", loss: {:.2e} \u00b1 {:.2e}".format(
            *self._loss_std_mean(self.print_every))
        loss_string += ", time: " + utils.get_formatted_time(dt)
        self._time_start_print = time.time()

        print(loss_string)

    def plot(self, time_stamp=None, scale='log', xlim=None, ylim=None,
             every=1, figsize=(10,5), save_name=None):
        """Plots loss from current time_stamp. Optionaly a time_stamp can be
           provided by argument.
        Args:
            time_stamp (str): time stamp to use instead of internal time stamp
                              Defaults to None.
            scale (str): scale of y axis
            x_lim (list): list with lower and upper limit of x axis
                          Defaults to None.
            y_lim (list): list with lower and upper limit of y axis
                          Defaults to None.
            every (int): only plots every every'th point. Defaults to 1.
            save_name (str): save name. Defaults to None.
        """

        sns.set()

        if time_stamp is not None:
            time_stamp_path = os.path.join(self.networks_path, time_stamp)
        else:
            time_stamp_path = self.time_stamp_path

        data = np.loadtxt(os.path.join(time_stamp_path, 'loss.csv'),
                          delimiter=',')
        iteration = data[::every, 0]
        loss = data[::every, 1]

        plt.figure(figsize=figsize)
        plt.plot(iteration, loss, 'b-')

        plt.xlabel('iterations')
        if xlim is None:
            plt.xlim([iteration[0], iteration[-1]])
        else:
            plt.xlim(xlim)

        plt.ylabel('L2 loss')
        if ylim is None:
            plt.ylim([min(loss), max(loss)])
        else:
            plt.ylim(ylim)
        plt.yscale(scale)

        plt.tight_layout()

        if save_name is not None:
            plt.savefig(os.path.join(time_stamp_path, save_name))

        return None

    def _create_submission_file(self, pred_list, id_list):

        # convert to numpy
        pred = torch.cat(pred_list).numpy()
        ids = torch.cat(id_list).numpy()

        # sort ids
        sorted_indices = np.argsort(ids)
        pred = pred[sorted_indices]
        ids = ids[sorted_indices]

        # write to file
        with open(os.path.join(self.time_stamp_path, 'submission.csv'),
                  "w") as f:

            f.write("id,scalar_coupling_constant\n")
            for i in range(len(pred)):
                f.write("%i,%f\n" % (ids[i], pred[i]))
