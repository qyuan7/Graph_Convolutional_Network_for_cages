#!/usr/bin/env python


from sklearn.model_selection import (cross_validate,
                                     cross_val_predict)
import numpy as np
import logging
import argparse
import sqlite3
import torch
from torch import optim
import torch.nn as nn
from models.gcn_models import gcn
import torch.utils.data as utils
import pandas as pd
from utils.gcn_utils import load_data, data_process
from utils.early_stopping import EarlyStopping


logger = logging.getLogger(__name__)


def save_r2_data(
                 cage_property,
                 reactions,
                 topologies,
                 reg,
                 fingerprints,
                 targets,
                 cv):
    np.random.seed(4)
    y_predict = cross_val_predict(reg,
                                  fingerprints,
                                  targets,
                                  cv=cv,
                                  n_jobs=-1)
    r = '_'.join(reactions)
    t = '_'.join(topologies)
    targets.dump(f'r{cage_property}_{r}_t_{t}_y_true.np')
    y_predict.dump(f'r{cage_property}_{r}_t_{t}_y_pred.np')


def dataset_prep(db, reactions, topologies, cage_property):
    logger.debug(f'Reactions: {reactions}.')
    logger.debug(f'Topologies: {topologies}.')
    fps, tops, labels = load_data(db, reactions, topologies=topologies, cage_property=cage_property)
    fps, labels = data_process(fps, labels)
    dataset = utils.TensorDataset(fps, labels)
    torch.manual_seed(66)
    big_len = int(len(dataset) * 0.8)
    test_len = len(dataset) - big_len
    val_len = int(big_len * 0.25)
    train_val_len = big_len - val_len
    big_train_dataset, test_dataset = utils.random_split(dataset, (big_len, test_len))
    train_dataset, val_dataset = utils.random_split(big_train_dataset, (train_val_len, val_len))
    torch.manual_seed(torch.initial_seed())
    big_train_data = utils.DataLoader(big_train_dataset,batch_size=64,shuffle=True,drop_last=False)
    train_data = utils.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
    val_data = utils.DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=False)
    test_data = utils.DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=False)
    return big_train_data,train_data, val_data, test_data


def train_gcn(device, train_data, val_data, test_data, save):

    graph_gcn = gcn(512, 128, dropout=0.5,n_class=1)
    graph_gcn = graph_gcn.to(device)
    #optimizer = optim.SGD(graph_gcn.parameters(), lr=1*1e-3, momentum=0.9)
    optimizer = optim.Adam(graph_gcn.parameters(), lr=1*1e-3, weight_decay=0.1)
    criterion = nn.L1Loss()

    train_losses, val_losses, val_acc = [],[],0
    A_hat = np.matrix([[1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                      [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
                      [0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
                      [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                      [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                      [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                      [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                      [0, 0, 1, 1, 0, 0, 0, 0, 0, 1]])

    def train():
        graph_gcn.train()
        cnt = 0
        running_loss = 0
        step = 0
        y_trues = []
        y_preds = []
        for step, batch in enumerate(train_data):
            X_inpt, y_true = batch[0], batch[1]
            X_inpt = X_inpt.to(device)
            y_true = y_true.to(device)
            cnt += len(y_true)
            step += 1
            A = A_hat
            optimizer.zero_grad()
            res = graph_gcn(X_inpt, A)
            res = res.squeeze()
            y_true = y_true.clone().detach().float()
            y_true = y_true.squeeze()
            loss = criterion(res, y_true)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            y_trues.extend(y_true)
            y_preds.extend(res)
            train_losses.append(loss.item())
        return running_loss, cnt, step,y_trues,y_preds

    def test(val=False):
        graph_gcn.eval()
        A = A_hat
        val_loss = 0
        cnt = 0
        step = 0
        y_trues = []
        y_preds = []
        if val:
            data = val_data
        else:
            data = test_data
        with torch.no_grad():
            for x, y in data:
                x = x.to(device)
                y = y.to(device)
                cnt += len(y)
                res = graph_gcn(x, A)
                res = res.squeeze()
                y = y.clone().detach().float()
                y = y.squeeze()
                val_loss += criterion(res, y)
                val_losses.append(val_loss)
                step += 1
                y_trues.extend(y)
                y_preds.extend(res)
        return val_loss, cnt, step, y_trues,y_preds
    early_stopping = EarlyStopping(patience=10, verbose=False)
    for epoch in range(50):
        train_loss, train_cnt, train_step,train_trues,train_preds =train()
        val_loss, val_cnt, val_step,val_trues,val_preds = test(val=True)
        test_loss, test_cnt, test_step, *args = test(val=False)
        #early_stopping(val_loss, graph_gcn)
        #if early_stopping.trained:
        print('{:.2f}, {:.2f}, {:.2f}'.
                  format(train_loss / train_step, val_loss / val_step, test_loss / test_step))

        #if early_stopping.early_stop:
        #    print('Early stopping')
        train_trues = [train_true.data.numpy() for train_true in train_trues]
        train_preds = [train_pred.data.numpy() for train_pred in train_preds]
        train_trues = np.array(train_trues)
        train_preds = np.array(train_preds)
        df = pd.DataFrame()
        df['train_true'] = train_trues
        df['train_pred'] = train_preds
        df.to_csv('train_single.csv', index=False)
        #break

def main(device):
    reacts = {
        1: 'amine2aldehyde3',
        2: 'aldehyde2amine3',
        3: 'alkene2alkene3',
        4: 'alkyne2alkyne3',
        5: 'amine2carboxylic_acid3',
        6: 'carboxylic_acid2amine3',
        7: 'thiol2thiol3',
        8: 'amine4aldehyde3',
        9: 'amine4aldehyde2',
        10: 'amine3aldehyde3',
        11: 'aldehyde4amine3',
        12: 'aldehyde4amine2'
    }

    tops = {
        1: 'FourPlusSix',
        2: 'EightPlusTwelve',
        3: 'SixPlusTwelve',
        4: 'SixPlusEight',
        5: 'FourPlusFour'
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('database_path')
    parser.add_argument(
        'cage_property',
        help='The cage property you want to do regression on.')
    parser.add_argument(
        '-r', '--reactions', type=int, nargs='+', metavar='REACTION',
        help=f'Reactions to train on. Given by {reacts}.')
    parser.add_argument(
        '-t', '--topologies',
        type=int, nargs='+', metavar='TOPOLOGY',
        help=f'Topologies to train on. Given by {tops}.')
    parser.add_argument(
        '--join', action='store_true',
        help=('Toggles if all reactions should be used to train '
              'one model or many.'))
    parser.add_argument(
        '-s', '--save', action='store_true',
        help='Toggles to save each models r2 results.')

    args = parser.parse_args()

    db = sqlite3.connect(args.database_path)

    if args.join:
        big_train_data, train_data, val_data, test_data =\
            dataset_prep(db=db,
                         reactions=[reacts[i] for i in args.reactions],
                         topologies=[tops[i] for i in args.topologies],
                         cage_property=args.cage_property)
        train_gcn(device=device,
                  train_data=train_data,
                  val_data=val_data,
                  test_data=test_data,
                  save=args.save)
    else:
        for react in args.reactions:
            big_train_data, train_data, val_data, test_data = \
                dataset_prep(db=db,
                             reactions=[reacts[react]],
                             topologies=[tops[i] for i in args.topologies],
                             cage_property=args.cage_property)
            train_gcn(device=device,
                      train_data=train_data,
                      val_data=val_data,
                      test_data=test_data,
                      save=args.save)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device)
