#!/usr/bin/env python
# -*- coding: utf-8 -*-

from booltest.booltest_main import Booltest
import random
import base64
import unittest
import pkg_resources
from booltest import common


__author__ = 'dusanklinec'


class TermEvalTest(unittest.TestCase):
    """Simple Booltest tests"""

    def __init__(self, *args, **kwargs):
        super(TermEvalTest, self).__init__(*args, **kwargs)

    def test_eval1(self):
        data = b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0'  # 12 x \xf0
        data_bin = common.to_bitarray(data)

        te = common.TermEval(16, 2)
        te.load(data_bin)
        r1 = te.eval_terms(1)
        self.assertEqual(r1[0], 6)
        self.assertEqual(r1[1], 6)
        self.assertEqual(r1[5], 0)

        r2 = te.eval_terms(2)
        self.assertEqual(r2[0], 6)  # x0x1
        self.assertEqual(r2[4], 0)  # x0x5

        r3 = te.eval_terms(3)
        self.assertEqual(r3[0], 6)
        self.assertEqual(r3[2], 0)  # x0x1x4

        r2x = te.eval_all_terms(3)
        self.assertEqual(r1, r2x[1])
        self.assertEqual(r2, r2x[2])
        self.assertEqual(r3, r2x[3])


if __name__ == "__main__":
    unittest.main()  # pragma: no cover


