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
    """Simple Booltest tests"""

    def __init__(self, *args, **kwargs):
        super(CommonTest, self).__init__(*args, **kwargs)

    def test_seed(self):
        """
        Simple test placeholder
        :return:
        """
        self.assertEqual(b'1fe40505e131963c', common.generate_seed(0))
        self.assertEqual(b'f39e9205e31b36fa', common.generate_seed(1))
        self.assertEqual(b'c2bf37890011dfed', common.generate_seed(2))
        self.assertEqual(b'8b66adc2d468e8ef', common.generate_seed(3))
        self.assertEqual(b'e14e803b4335a358', common.generate_seed(10))
        self.assertEqual(b'18acf33bd271456c', common.generate_seed(65539))


if __name__ == "__main__":
    unittest.main()  # pragma: no cover


