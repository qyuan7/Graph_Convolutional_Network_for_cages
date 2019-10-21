#!/usr/bin/env python

import logging
import sqlite3
import logging
import argparse
import torch
from gcn_utils import load_data_cross, data_process
import torch.utils.data as utils
from gcn_collapse import train_gcn

logger = logging.getLogger(__name__)


def dataset_cross(db, reaction, train, cage_property, table, split=False):
    if table:
        logger.setLevel(logging.ERROR)
    fps, tops, labels = load_data_cross(db, reaction, train, cage_property)
    fps, labels = data_process(fps, labels)
    dataset = utils.TensorDataset(fps, labels)
    data = utils.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
    if split:
        torch.manual_seed(4)
        train_len = int(len(dataset)*0.8)
        val_len = len(dataset) - train_len
        train_dataset, val_dataset = utils.random_split(dataset, (train_len, val_len))
        train_data = utils.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
        val_data = utils.DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=False)
        torch.manual_seed(torch.initial_seed())
        return train_data, val_data
    else:
        return data


def main(device):
    reacts = {
        1: 'amine2aldehyde3',
        2: 'aldehyde2amine3',
        3: 'alkene2alkene3',
        4: 'alkyne2alkyne3',
        5: 'amine2carboxylic_acid3',
        6: 'carboxylic_acid2amine3',
        7: 'thiol2thiol3'
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('database_path')
    parser.add_argument('type', choices=['train', 'test'])
    parser.add_argument(
        'reactions', type=int, nargs='+', metavar='REACTION',
        help=f'Reaction to train on. Given by {reacts}.')
    parser.add_argument(
        '--table', action='store_true',
        help='Print out results in tex table format.')
    args = parser.parse_args()

    db = sqlite3.connect(args.database_path)
    for reaction in args.reactions:
        reverse = (args.type == 'test')
        logger.debug(f'{reacts[reaction]} - reverse {reverse}')
        train_data, val_data = dataset_cross(db=db,
                                             reaction=reacts[reaction],
                                             train=True if not reverse else False,
                                             cage_property=None,
                                             table=args.table,
                                             split=True)
        test_data = dataset_cross(db=db,
                                  reaction=reacts[reaction],
                                  train=False if not reverse else True,
                                  cage_property=None,
                                  table=args.table,
                                  split=False)
        train_gcn(device, train_data, val_data, test_data, table=args.table, save=None)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device)
