import sqlite3
import numpy as np
import torch
from sklearn.preprocessing import LabelBinarizer


def load_data(db, reactions,topologies=["FourPlusSix"], cage_property=None):

    if cage_property:
        query = '''
                SELECT fingerprint, topology, {}
                FROM cages
                WHERE
                reaction IN ({}) AND
                collapsed = 0 AND
                topology IN ({})
                '''.format(cage_property,
                           ', '.join('?'*len(reactions)),
                           ', '.join('?'*len(topologies)))

    else:
        query = '''
                SELECT fingerprint, topology, collapsed
                FROM cages
                WHERE
                     reaction IN ({}) AND
                     collapsed IS NOT NULL AND
                     topology IN ({})
                '''.format(', '.join('?' * len(reactions)),
                           ', '.join('?' * len(topologies)))
    results = ((eval(fp), top, label) for fp, top, label in
               db.execute(query, reactions + topologies))
    fps, tops, labels = zip(*results)
    tops = LabelBinarizer().fit_transform(tops)
    fps = np.array(fps)
    labels = np.array(labels)
    return fps, tops, labels


def load_data_cross(db, reaction, train, cage_property=None):
    sign = '=' if train else '!='
    if cage_property:
        query = f'''
                 SELECT fingerprint, topology, {cage_property}
                 FROM cages
                 WHERE
                 reaction {sign} "{reaction}" AND
                 collapsed = 0 AND
                 topology="FourPlusSix"
                 '''
    else:
        query = f'''
                 SELECT fingerprint, topology, collapsed
                 FROM cages
                 WHERE
                 reaction {sign} "{reaction}" AND
                 collapsed IS NOT NULL AND
                 topology = "FourPlusSix"
                 '''
    results = ((eval(fp), top, label) for fp, top, label in db.execute(query))
    fps, tops, labels = zip(*results)
    tops = LabelBinarizer().fit_transform(tops)
    fps = np.array(fps)
    labels = np.array(labels)
    return fps, tops, labels


def data_process(fps, labels):
    fps = fps.reshape(len(fps), 2, -1)
    bbs = fps[:, 0, :]
    lks = fps[:, 1, :]
    bbs = np.expand_dims(bbs, axis=1)
    lks = np.expand_dims(lks, axis=1)
    fps = np.concatenate((bbs, bbs, bbs, bbs, lks, lks, lks, lks, lks, lks), axis=1)
    fps = torch.from_numpy(fps).float()
    labels = torch.from_numpy(labels).float()
    return fps, labels

def gen_dataset(fps, labels):
    pass

