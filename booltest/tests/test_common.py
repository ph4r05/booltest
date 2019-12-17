#!/usr/bin/env python
# -*- coding: utf-8 -*-

from booltest.booltest_main import Booltest
import random
import base64
import unittest
import pkg_resources
from booltest import common


__author__ = 'dusanklinec'


class CommonTest(unittest.TestCase):
    """Simple BoolTest tests"""

    def __init__(self, *args, **kwargs):
        super(CommonTest, self).__init__(*args, **kwargs)

    def test_seed(self):
        self.assertEqual(b'1fe40505e131963c', common.generate_seed(0))
        self.assertEqual(b'f39e9205e31b36fa', common.generate_seed(1))
        self.assertEqual(b'c2bf37890011dfed', common.generate_seed(2))
        self.assertEqual(b'8b66adc2d468e8ef', common.generate_seed(3))
        self.assertEqual(b'e14e803b4335a358', common.generate_seed(10))
        self.assertEqual(b'18acf33bd271456c', common.generate_seed(65539))

    def test_rank(self):
        self.assertEqual(common.rank([0, 1], 10), 0)
        self.assertEqual(common.rank([0, 2], 10), 1)
        self.assertEqual(common.rank([1, 0], 10), 9)
        self.assertEqual(common.rank([2, 0], 10), 17)
        self.assertEqual(common.rank([0, 1, 2, 3], 10), 0)
        self.assertEqual(common.rank([0, 1, 8, 9], 10), 27)
        self.assertEqual(common.rank([1, 2, 3, 4], 10), 84)

    def test_unrank(self):
        self.assertEqual(common.unrank(0, 10, 4), [0, 1, 2, 3])
        self.assertEqual(common.unrank(1, 10, 4), [0, 1, 2, 4])
        self.assertEqual(common.unrank(27, 10, 4), [0, 1, 8, 9])
        self.assertRaises(ValueError, lambda: common.unrank(240, 10, 4))

    def test_rank_unrank(self):
        for idx in range(0, 200, 7):
            s = common.unrank(idx, 10, 4)
            v = common.rank(s, 10)
            self.assertEqual(v, idx)

    def test_to_bitarray(self):
        self.assertEqual(len(common.to_bitarray(b'\x00')), 8)
        self.assertEqual(len(common.to_bitarray(b'\x00\x00\x00')), 3*8)
        self.assertEqual(len(common.to_bitarray(b'test_test')), 9*8)
        self.assertEqual(common.hw(common.to_bitarray(b'\x00\x00')), 0)
        self.assertEqual(common.hw(common.to_bitarray(b'\x01\x01')), 2)
        self.assertEqual(common.hw(common.to_bitarray(b'\xff\xff')), 16)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover


