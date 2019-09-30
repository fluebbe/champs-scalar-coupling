import os
import numpy as np
import pandas as pd
import torch


def merge_data_with_structures(data, structures):
    """Merges data and structure files
    """

    # conversion to right dataframe
    d = np.column_stack([structures['molecule_name'].values,
                         structures['atom_index'].values,
                         structures['x'].values,
                         structures['y'].values,
                         structures['z'].values])
    c = ['molecule_name', 'atom_index_0',
         'x0', 'y0', 'z0']

    structures = pd.DataFrame(data=d, columns=c)
    del d, c

    # merging of structures and data for atom 0
    data = pd.merge(data, structures, on=['molecule_name', 'atom_index_0'])

    # merging of structures and data for atom 1
    columns = {'atom_index_0': 'atom_index_1',
               'x0': 'x1', 'y0': 'y1', 'z0': 'z1'}
    structures.rename(columns=columns, inplace=True)

    return pd.merge(data, structures, on=['molecule_name', 'atom_index_1'])

def extract_scc(data, N_max):
    """Extracts ssc from data and converts it to np array
    """

    # copy scalar coupling constants
    scc = data[['molecule_name', 'scalar_coupling_constant']].copy()

    # write all scc of a molecule into one line
    scc = scc.groupby('molecule_name').apply(lambda x: x.values.reshape(-1))

    # pad with zeros
    scc_list = []
    for i in range(len(scc)):
        n = N_max*2 - len(scc[i])
        scc_list.append(np.pad(scc[i], (0, n), 'constant',
                        constant_values=(0, np.nan)))

    # build np array and remove molecule names
    scc = np.vstack(scc_list)
    del scc_list
    scc = np.delete(scc, [2*i for i in range(N_max)], 1)

    # convert to array and return
    return scc.astype(float)

def encode_and_extract_type(input_data, N_max):
    """One hot encodes type, adds it to data and returns data and types
    """

    # copy types
    conn_type = input_data[['type']].copy()

    # encode types
    type_dict = {'1JHC': [1, 0, 0, 0, 0, 0, 0, 0],
                 '1JHN': [0, 1, 0, 0, 0, 0, 0, 0],
                 '2JHC': [0, 0, 1, 0, 0, 0, 0, 0],
                 '2JHH': [0, 0, 0, 1, 0, 0, 0, 0],
                 '2JHN': [0, 0, 0, 0, 1, 0, 0, 0],
                 '3JHC': [0, 0, 0, 0, 0, 1, 0, 0],
                 '3JHH': [0, 0, 0, 0, 0, 0, 1, 0],
                 '3JHN': [0, 0, 0, 0, 0, 0, 0, 1]}

    conn_type['type'] = conn_type['type'].map(type_dict)
    conn_type = np.column_stack([conn_type.values.tolist()]).reshape(-1, 8)
    conn_type = pd.DataFrame(data=conn_type,
                             columns=['1', '2', '3', '4', '5', '6', '7', '8'])

    data = input_data.merge(conn_type, left_index=True, right_index=True)

    del conn_type

    # drop type
    data.drop(columns=['type'], inplace=True)

    # reorder columns
    columns = ['molecule_name',
               '1', '2', '3', '4', '5', '6', '7', '8',
               'x0', 'y0', 'z0',
               'x1', 'y1', 'z1']
    data = data.reindex(columns=columns)
    del columns

    # get type matrix
    # copy types
    type_data = input_data[['molecule_name', 'type']].copy()

    # encode types
    type_dict = {'1JHC': 0, '1JHN': 1, '2JHC': 2, '2JHH': 3,
                 '2JHN': 4, '3JHC': 5, '3JHH': 6, '3JHN': 7}
    type_data['type'] = type_data['type'].map(type_dict)

    # write all types of a molecule into one line
    type_data = type_data.groupby('molecule_name').apply(lambda x:
                                                         x.values.reshape(-1))

    # pad with nans
    type_list = []
    for i in range(len(type_data)):
        n = N_max*2 - len(type_data[i])
        type_list.append(np.pad(type_data[i], (0, n), 'constant',
                                constant_values=(0, -1)))

    # build np array and remove molecule names
    type_data = np.vstack(type_list)
    del type_list
    type_data = np.delete(type_data, [2*i for i in range(N_max)], 1)

    return data, type_data

def write_conns_in_one_line(data, N_max, n_per_conn):
    """Writes all connections of a molecule into one line
       and returns as np array
    """

    # write all connections of a molecule into one line
    data = data.groupby('molecule_name').apply(lambda x: x.values.reshape(-1))

    # pad with nans
    mol_list = []
    for i in range(len(data)):
        n = N_max * (n_per_conn+1) - len(data[i])
        mol_list.append(np.pad(data[i], (0, n), 'constant',
                               constant_values=(0, np.nan)))

    # build np array
    data = np.vstack(mol_list)
    del mol_list

    # remove molecule names
    data = np.delete(data, [(n_per_conn+1)*i for i in range(N_max)], 1)

    # convert to np array
    return data.astype(float)

def standardize_and_save_train(data, scc, type_data, N_max, n_per_conn, path):
    """Standardizes scc by type and saves everything
    """

    size = (N_max*n_per_conn - np.isnan(data).sum(axis=1)) // n_per_conn

    # remove size 1 molecules
    data = data[size!=1]
    scc = scc[size!=1]
    type_data = type_data[size!=1]
    size = size[size!=1]

    # standardize
    type_mean = []
    type_std = []

    for i in range(8):
        type_mean.append(scc[type_data == i].mean())
        type_std.append(scc[type_data == i].std())
        scc[type_data == i] -= type_mean[i]
        scc[type_data == i] /= type_std[i]

    # convert to Tensor
    data = torch.Tensor(data)
    scc = torch.Tensor(scc)
    type_data = torch.from_numpy(type_data.astype(np.int32))
    size = torch.from_numpy(size.astype(np.int32))

    # save
    torch.save(data, os.path.join(path, 'train.pt'))
    torch.save(scc, os.path.join(path, 'scc.pt'))
    torch.save(type_data, os.path.join(path, 'train_type.pt'))
    torch.save(size, os.path.join(path, 'size.pt'))

    # save mean and std of scc
    np.savetxt(os.path.join(path, 'scc_mean_std.csv'),
               np.array([type_mean, type_std]),
               delimiter=',')

def get_ids(data, N_max):
    """Extracts and returns id data
    """

    id_data = data[['molecule_name', 'id']].copy()

    # write molecules into one line
    id_data = id_data.groupby('molecule_name').apply(lambda x:
                                                     x.values.reshape(-1))

    # pad with nans
    id_list = []
    for i in range(len(id_data)):
        n = N_max*2 - len(id_data[i])
        id_list.append(np.pad(id_data[i], (0, n), 'constant',
                                constant_values=(0, np.nan)))

    # build np array and remove molecule names
    id_data = np.vstack(id_list)
    del id_list
    id_data = np.delete(id_data, [2*i for i in range(N_max)], 1)

    return id_data

def sort_and_save_test(data, type_data, id_data, N_max, n_per_conn, path):
    """Sorts molecules by size and saves everything
    """

    sizes = (N_max*n_per_conn - np.isnan(data).sum(axis=1)) // n_per_conn

    # get sorted indices
    sorted_indices = np.argsort(sizes)

    # sort
    data = data[sorted_indices]
    sizes = sizes[sorted_indices]
    type_data = type_data[sorted_indices]
    id_data = id_data[sorted_indices]

    # counter variable
    i = 0

    data_ = []
    type_ = []
    id_ = []

    # loop over sizes
    for size in range(1, N_max+1):
        data_list = []
        type_list = []
        id_list = []

        while i < len(data) and size == sizes[i]:
            data_list.append(data[i, 0:(size*n_per_conn)])
            type_list.append(type_data[i, 0:size].astype(int))
            id_list.append(id_data[i, 0:size].astype(int))
            i += 1

        if len(data_list) > 1:
            data_.append(torch.tensor(np.vstack(data_list),
                                      dtype=torch.float32))
            type_.append(torch.from_numpy(np.vstack(type_list)))
            id_.append(torch.from_numpy(np.vstack(id_list)))
        elif len(data_list) == 1:
            data_.append(torch.from_numpy(data_list[0], dtype=torch.float32))
            type_.append(torch.from_numpy(type_list[0]))
            id_.append(torch.from_numpy(id_list[0]))
        else:
            continue

    torch.save(data_, os.path.join(path, 'test.pt'))
    torch.save(type_, os.path.join(path, 'test_type.pt'))
    torch.save(id_, os.path.join(path, 'id.pt'))
