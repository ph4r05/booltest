#!/usr/bin/env python
# -*- coding: utf-8 -*-

from booltest.booltest_main import Booltest
import random
import base64
import unittest
import pkg_resources
from booltest import misc


__author__ = 'dusanklinec'


class MiscTest(unittest.TestCase):
    """Simple BoolTest tests"""

    def __init__(self, *args, **kwargs):
        super(MiscTest, self).__init__(*args, **kwargs)

    def test_cpuinfo(self):
        """
        Get CPU
        :return:
        """
        info = misc.try_get_cpu_info()
        self.assertIsNotNone(info)

    def test_hostname(self):
        """
        Get hostname
        :return:
        """
        hst = misc.try_get_hostname()
        self.assertIsNotNone(hst)

    def test_cpu_pcnt(self):
        """
        Get cpu percent
        :return:
        """
        res = misc.try_get_cpu_percent()
        self.assertIsNotNone(res)

    def test_cpu_load(self):
        """
        Get cpu load
        :return:
        """
        res = misc.try_get_cpu_load()
        self.assertIsNotNone(res)


if __name__ == "__main__":
    unittest.main()  # pragma: no cover


