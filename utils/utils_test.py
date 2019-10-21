#!/usr/bin/env python

import sqlite3
import unittest
from gcn_utils import *

db = sqlite3.connect('../../cage_prediction.db')

class Test(unittest.TestCase):

    def test_load(self):
        fps, tops, labels = load_data(db, ['amine2aldehyde3'], topologies=["FourPlusSix"], cage_property=False)
        self.assertEqual(len(fps), 4583)
        self.assertEqual(len(fps[0]), 1024)

    def test_process(self):
        fps, tops, labels = load_data(db, ['amine2aldehyde3'], topologies=["FourPlusSix"], cage_property=False)
        fps, labels = data_process(fps, labels)
        self.assertEqual(len(fps), 4583)
        self.assertEqual(len(labels), 4583)
        self.assertEqual((fps[0].shape), (10,512))

    def test_load_cross(self):
        fps_train, tops_train, labels_train = load_data_cross(db, 'amine2aldehyde3', train=False, cage_property=None)
        fps_test, tops_test, labels_test = load_data_cross(db, 'amine2aldehyde3',train=True, cage_property=None)
        self.assertEqual(len(fps_test), 4583)
        self.assertEqual(len(fps_train), 30304)



if __name__ == '__main__':
    unittest.main()

