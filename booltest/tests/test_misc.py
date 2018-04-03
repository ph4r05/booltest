#!/usr/bin/env python
# -*- coding: utf-8 -*-

from booltest.booltest_main import Booltest
import random
import base64
import unittest
import pkg_resources
from booltest import misc


__author__ = 'dusanklinec'


class CommonTest(unittest.TestCase):
    """Simple Booltest tests"""

    def __init__(self, *args, **kwargs):
        super(CommonTest, self).__init__(*args, **kwargs)

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


if __name__ == "__main__":
    unittest.main()  # pragma: no cover


