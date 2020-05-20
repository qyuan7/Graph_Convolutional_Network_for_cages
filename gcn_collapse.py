#!/usr/bin/env python

import numpy as np
import sqlite3
import logging
import argparse
import torch
from torch import optim
import torch.nn as nn
from models.gcn_models import gcn
from utils.gcn_utils import load_data, data_process
import torch.utils.data as utils
from utils.early_stopping import EarlyStopping


logger = logging.getLogger(__name__)


def dataset_prep(db, reactions, topologies,batch_size=128):
    logger.debug(f'Reactions: {reactions}.')
    logger.debug(f'Topologies: {topologies}.')
    fps, tops, labels = load_data(db, reactions, topologies=topologies, cage_property=False)
    fps, labels = data_process(fps, labels)
    dataset = utils.TensorDataset(fps, labels)
    #torch.manual_seed(43) acc 089
    #torch.manual_seed(43)  
    big_len = int(len(dataset) * 0.85)
    test_len = len(dataset) - big_len
    val_len = int(big_len * 0.176)
    train_val_len = big_len - val_len
    big_train_dataset, test_dataset = utils.random_split(dataset, (big_len, test_len))
    train_dataset, val_dataset = utils.random_split(big_train_dataset, (train_val_len, val_len))
    big_train_data = utils.DataLoader(big_train_dataset,batch_size=batch_size,shuffle=True,drop_last=False)
    train_data = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_data = utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_data = utils.DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=False)
    #torch.manual_seed(torch.initial_seed())
    #print(len(big_train_data)*64, len(train_data)*64, len(val_data)*64,len(test_data)*64)
    return big_train_data, train_data, val_data, test_data


def train_data_process(data):
    cnt, y1, y0 = 0, 0, 0
    for step, batch in enumerate(data):
        _, y_true = batch[0], batch[1]
        cnt += len(y_true)
        y = y_true.clone().detach().long()

        y1 += (y == 1).sum()
        y0 += (y == 0).sum()
    return cnt, y1, y0
        

def train_gcn(device, train_data, val_data, test_data, table, save):
    if table:
        logger.setLevel(logging.ERROR)

    graph_gcn = gcn(512,64,dropout=0.6, n_class=2)
    graph_gcn = graph_gcn.to(device)
    optimizer = optim.Adam(graph_gcn.parameters(), lr=4*1e-4)
    #optimizer = optim.SGD(graph_gcn.parameters(),lr=2*1e-3,momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=0, last_epoch=-1)
    train_size, train_true, train_false = train_data_process(train_data)
    print(train_size, train_true, train_false)
    eta = 0.1
    weights = torch.tensor((train_false.item()/train_size+eta,train_true.item()/train_size-eta))
    criterion = nn.CrossEntropyLoss(weight=weights)
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
    #np.clip(A_hat, 0.9, 1, out=A_hat)
    def train():
        graph_gcn.train()
        cnt = 0
        running_loss = 0
        corr = 0
        
        for step, batch in enumerate(train_data):
            X_inpt, y_true = batch[0], batch[1]
            cnt += len(y_true)
            optimizer.zero_grad()
            res = graph_gcn(X_inpt, A_hat)
            y_true = y_true.clone().detach().long()
            pred = torch.argmax(res.data,1)
            corr += (pred == y_true).sum().item()
            loss = criterion(res, y_true)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        return corr, running_loss, cnt

    def test(val=False):
        graph_gcn.eval()
        test_loss, correct = 0, 0
        cnt = 0
        correct1, correct0 = 0, 0
        y1, y0 = 0, 0
        p1, p0 = 0,0
        if val:
            data = val_data
        else:
            data = test_data
        with torch.no_grad():
            for x, y in data:
                cnt += len(y)
                res = graph_gcn(x, A_hat)
                y = y.clone().detach().long()
                test_loss += criterion(res, y)*cnt
                pred = torch.argmax(res.data, 1)
                correct += (pred == y).sum().item()
                correct1 += ((pred ==1)& (y == 1)).sum().item()
                correct0 += ((pred ==0)&(y == 0)).sum().item()
                y1 += (y == 1).sum().item()
                y0 += (y == 0).sum().item()
                p1 += (pred == 1).sum().item()
                p0 += (pred == 0).sum().item()
        return correct, test_loss, cnt, correct1, correct0, y1, y0, p1, p0

    early_stopping = EarlyStopping(patience=20, verbose=False)
    print('train_acc, train_loss, test_acc_val, test_loss_val, test_acc_noval,p1,r1,p0,r0')
    #print('train_acc, train_loss, test_loss, test_acc')
    for epoch in range(100):
        train_corr, train_loss, train_cnt =train()
        #acc, loss, cnt, *args = test(val=True)
        test_corr, test_loss, test_cnt, c1, c0, y1, y0, p1, p0 = test()
        scheduler.step() 
        #if epoch == 0:
        #    print(train_cnt, cnt, test_cnt)
        #early_stopping(loss, graph_gcn)
        #if early_stopping.trained:
        print('train_acc, train_loss, test_acc_val, test_loss_val, test_acc_noval')
        #print('{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f},{:.2f}, {:.2f},{:.2f},{:.2f}'.
        #       format(train_corr / train_cnt, train_loss, acc / cnt, loss.item()/cnt,
        #       test_corr / test_cnt, c1 / p1, c1 / y1, c0 / p0, c0 / y0))
        #print('{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(train_corr/train_cnt, train_loss, acc/cnt,loss.item()/cnt,test_corr/test_cnt))
        print('{:.2f},  {:.2f}, {:.2f}, {:.3f}'.format(train_corr/train_cnt, train_loss, test_loss/test_cnt,test_corr/test_cnt))
        if early_stopping.early_stop:
            torch.save(graph_gcn.state_dict(), 'collapse_gcn.ckpt')
            print('Early stopping')
            break


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
        '--table', action='store_true',
        help='Print out results in tex table format.')
    parser.add_argument(
        '-s', '--save', action='store_true',
        help='Toggles to save each trained model.')

    args = parser.parse_args()

    db = sqlite3.connect(args.database_path)

    if args.join:
        big_train_data, train_data, val_data, test_data = \
            dataset_prep(db=db,
                         reactions=[reacts[i] for i in args.reactions],
                         topologies=[tops[i] for i in args.topologies])
        train_gcn(device=device,
                  train_data=big_train_data,
                  val_data=val_data,
                  test_data=test_data,
                  table=args.table,
                  save=args.save)
    else:
        for react in args.reactions:
            big_train_data, train_data, val_data, test_data = \
                dataset_prep(db=db,
                             reactions=[reacts[react]],
                             topologies=[tops[i] for i in args.topologies])
            train_gcn(device=device,
                      train_data=train_data,
                      val_data=val_data,
                      test_data=test_data,
                      table=args.table,
                      save=args.save)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(device)
