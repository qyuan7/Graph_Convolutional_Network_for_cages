#!/usr/bin/env python

import logging
import sqlite3
import logging
import argparse
import torch
from gcn_utils import load_data_cross, data_process
import torch.utils.data as utils
from gcn_regression import train_gcn

logger = logging.getLogger(__name__)


def dataset_cross(db, reaction, train, cage_property, table=None, split=False):
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

def main():
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
    parser.add_argument(
        'cage_property',
        help='The cage property you want to do regression on.')
    parser.add_argument(
        'reactions', type=int, nargs='+', metavar='REACTION',
        help=f'Reaction to train on. Given by {reacts}.')
    args = parser.parse_args()

    db = sqlite3.connect(args.database_path)
    for reaction in args.reactions:
        mse, mae, r2 = train(
              db=db,
              cage_property=args.cage_property,
              reaction=reacts[reaction],
              reverse=False)
        print(f'mse - {mse:.2f}\nmae - {mae:.2f}\nr2 - {r2:.2f}')
        mse, mae, r2 = train(
              db=db,
              cage_property=args.cage_property,
              reaction=reacts[reaction],
              reverse=True)
        print(f'mse - {mse:.2f}\nmae - {mae:.2f}\nr2 - {r2:.2f}')
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
    parser.add_argument(
        'cage_property',
        help='The cage property you want to do regression on.')
    parser.add_argument(
        'reactions', type=int, nargs='+', metavar='REACTION',
        help=f'Reaction to train on. Given by {reacts}.')
    args = parser.parse_args()

    db = sqlite3.connect(args.database_path)
    for reaction in args.reactions:
        print(f'Using reaction {reacts[reaction]} as training set')
        train_data, val_data = dataset_cross(db=db,
                                             reaction=reacts[reaction],
                                             train=True,
                                             cage_property=args.cage_property,
                                             split=True)
        test_data = dataset_cross(db=db,
                                  reaction=reacts[reaction],
                                  train=False,
                                  cage_property=args.cage_property,
                                  split=False)
        train_gcn(device, train_data, val_data, test_data, save=None)
        print(f"Using reaction {reacts[reaction]} as test set")
        train_data2, val_data2 = dataset_cross(db=db,
                                             reaction=reacts[reaction],
                                             train=False,
                                             cage_property=args.cage_property,
                                             split=True)
        test_data2 = dataset_cross(db=db,
                                  reaction=reacts[reaction],
                                  train=True,
                                  cage_property=args.cage_property,
                                  split=False)
        train_gcn(device, train_data2, val_data2, test_data2,save=None)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device)
