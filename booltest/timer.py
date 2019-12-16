#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time


class Timer(object):
    """
    Simple stopwatch timer
    """
    def __init__(self, start=False):
        self.time_start = time.time() if start else None
        self.time_acc = 0

    def stop(self):
        if self.time_start is None:
            raise ValueError('Invalid timing state')
        self.time_acc += time.time() - self.time_start
        self.time_start = None

    def start(self):
        if self.time_start is not None:
            raise ValueError('Invalid timing state')
        self.time_start = time.time()

    def total(self):
        if self.time_start is not None:
            self.stop()
        return self.time_acc

    def cur(self):
        res = self.time_acc
        if self.time_start is not None:
            res += time.time() - self.time_start
        return res

    def reset(self):
        self.time_start = None
        self.time_acc = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


