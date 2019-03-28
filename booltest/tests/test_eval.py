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

    def test_nothing(self):
        self.assertTrue(1)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover


