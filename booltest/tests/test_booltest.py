#!/usr/bin/env python
# -*- coding: utf-8 -*-

from booltest.booltest_main import Booltest
import random
import base64
import unittest
import pkg_resources


__author__ = 'dusanklinec'


class BooltestTest(unittest.TestCase):
    """Simple Booltest tests"""

    def __init__(self, *args, **kwargs):
        super(BooltestTest, self).__init__(*args, **kwargs)

    def test_nothing(self):
        """
        Simple test placeholder
        :return:
        """
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover


