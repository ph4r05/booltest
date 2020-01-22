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
    """Simple BoolTest tests"""

    def __init__(self, *args, **kwargs):
        super(TermEvalTest, self).__init__(*args, **kwargs)

    def get_test_poly(self):
        return [
            [[0]],
            [[0], [0]],
            [[0], [1]],
            [[0, 1]],
            [[0, 1], [0]],
            [[0, 1], [2]],
            [[0, 1, 2]],
            [[0, 1, 2], [0]],
            [[0, 1, 2], [0, 1]],
            [[0, 1, 2], [3]],
            [[0, 1, 2], [0, 1, 3]],
            [[0, 1, 2], [2, 3, 4]],
            [[0, 1, 2], [1, 2, 3]],
            [[0, 1, 2], [3, 4, 5]],
            [[5, 6, 7], [8, 9, 10]],
            [[5, 6, 7], [7, 8, 9]],
            [[1, 2], [2, 3], [1, 3]],
            [[1, 2], [2, 3], [1, 3], [1, 5], [2, 9]],
            [[0, 1, 2], [2, 3, 4], [5, 6, 7]],
            [[0, 1, 2], [2, 3, 4], [1, 2, 3]],
            [[0, 1, 2], [2, 3, 5], [0, 3, 7, 9]],
        ]

    def test_exp(self):
        te = common.TermEval(16, 6)
        self.assertEqual(te.expp_term_deg(1), 0.5)
        self.assertEqual(te.expp_term_deg(2), 0.25)
        self.assertEqual(te.expp_term_deg(4), 0.0625)

        self.assertEqual(te.expp_term([0]), 0.5)
        self.assertEqual(te.expp_term([10]), 0.5)
        self.assertEqual(te.expp_term([1, 1]), 0.5)
        self.assertEqual(te.expp_term([1, 2]), 0.25)

        self.assertEqual(te.expp_poly([[1]]), 0.5)
        self.assertEqual(te.expp_poly([[1], [1]]), 0)  # always zero
        self.assertEqual(te.expp_poly([[1], [2]]), 0.5)
        self.assertEqual(te.expp_poly([[1, 2], [2]]), 0.25)
        self.assertEqual(te.expp_poly([[1, 2], [3]]), 0.5)
        self.assertEqual(te.expp_poly([[1, 2, 3, 4, 5], [6]]), 0.5)  # xor randomizes
        self.assertEqual(te.expp_poly([[1, 2, 3], [2]]), 0.375)
        self.assertEqual(te.expp_poly([[1, 2, 3], [4], [5]]), 0.5)
        self.assertEqual(te.expp_poly([[1, 2, 3], [4], [5], [6], [7]]), 0.5)
        self.assertEqual(te.expp_poly([[1, 2, 3], [4], [5], [6], [7, 8]]), 0.5)
        self.assertEqual(te.expp_poly([[1, 2, 3], [4], [5], [6], [6, 7]]), 0.5)
        self.assertEqual(te.expp_poly([[1, 2, 3], [1], [2], [3]]), 0.375)  # deps

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

    def test_exp_poly(self):
        """
        Verify expected number eval optimized heuristic with brute-force approach
        """
        polys = self.get_test_poly()
        term_eval = common.TermEval(blocklen=12, deg=3)

        for idx, poly in enumerate(polys):
            expp = term_eval.expp_poly(poly)
            expp2 = term_eval.expp_poly_sim(poly)
            self.assertAlmostEqual(expp, expp2)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover


