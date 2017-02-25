from past.builtins import basestring
from functools import reduce
import argparse
import logging
import coloredlogs
import common
import os
import re
import six
import sys
import math
import random
import json
import types
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.misc
import scipy.stats
import subprocess
from main import *

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


# Main - argument parsing + processing
class RandVerif(App):
    def __init__(self, *args, **kwargs):
        super(RandVerif, self).__init__(*args, **kwargs)
        self.args = None
        self.tester = None
        self.blocklen = None
        self.term_map = []
        self.input_poly = []

    def work(self):
        """
        Main entry point - data processing
        :return:
        """
        self.blocklen = int(self.defset(self.args.blocklen, 128))
        deg = int(self.defset(self.args.degree, 3))
        tvsize_orig = int(self.defset(self.process_size(self.args.tvsize), 1024*256))
        zscore_thresh = float(self.args.conf)
        rounds = int(self.args.rounds) if self.args.rounds is not None else None
        top_k = int(self.args.topk) if self.args.topk is not None else None
        top_comb = int(self.defset(self.args.combdeg, 2))
        reffile = self.defset(self.args.reffile)
        all_deg = self.args.alldeg
        tvsize = tvsize_orig

        # Load input polynomials
        self.load_input_poly()
        script_path = common.get_script_path()

        logger.info('Basic settings, deg: %s, blocklen: %s, TV size: %s, rounds: %s'
                    % (deg, self.blocklen, tvsize_orig, rounds))

        total_terms = int(scipy.misc.comb(self.blocklen, deg, True))
        hwanalysis = HWAnalysis()
        hwanalysis.deg = deg
        hwanalysis.blocklen = self.blocklen
        hwanalysis.top_comb = top_comb
        hwanalysis.comb_random = self.args.comb_random
        hwanalysis.top_k = top_k
        hwanalysis.combine_all_deg = all_deg
        hwanalysis.zscore_thresh = zscore_thresh
        hwanalysis.do_ref = reffile is not None
        hwanalysis.input_poly = self.input_poly
        hwanalysis.no_comb_and = self.args.no_comb_and
        hwanalysis.no_comb_xor = self.args.no_comb_xor
        hwanalysis.prob_comb = self.args.prob_comb
        hwanalysis.all_deg_compute = len(self.input_poly) == 0
        logger.info('Initializing test')
        hwanalysis.init()

        for test_idx in range(self.args.tests):
            seed = random.randint(0, 2**32-1)
            cmd = ''
            if self.args.test_randc:
                path = os.path.realpath(os.path.join(script_path, '../assets/rndgen-c/rand'))
                cmd = '%s %s' % (path, seed)
            elif self.args.test_java:
                path = os.path.realpath(os.path.join(script_path, '../assets/rndgen-java/'))
                cmd = 'java -cp %s Main %s' % (path, seed)
            else:
                raise ValueError('No generator to test')

            # Subprocess to redirect generator to a pipe we can read from
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1024, close_fds=True, shell=True)
            iobj = common.FileLikeInputObject(fh=proc.stdout, desc=cmd)

            size = iobj.size()
            logger.info('Testing input object: %s, size: %d kB, iteration: %d' % (iobj, size/1024.0, test_idx))

            # size smaller than TV? Adapt tv then
            if size >= 0 and size < tvsize:
                logger.info('File size is smaller than TV, updating TV to %d' % size)
                tvsize = size

            hwanalysis.reset()
            logger.info('BlockLength: %d, deg: %d, terms: %d' % (self.blocklen, deg, total_terms))
            with iobj:
                data_read = 0
                cur_round = 0

                while size < 0 or data_read < size:
                    if rounds is not None and cur_round > rounds:
                        break

                    data = iobj.read(tvsize)
                    bits = common.to_bitarray(data)
                    if len(bits) == 0:
                        logger.info('File read completely')
                        break

                    logger.info('Pre-computing with TV, deg: %d, blocklen: %04d, tvsize: %08d = %8.2f kB = %8.2f MB, '
                                'round: %d, avail: %d' %
                                (deg, self.blocklen, tvsize, tvsize/1024.0, tvsize/1024.0/1024.0, cur_round, len(bits)))

                    hwanalysis.proces_chunk(bits, None)
                    cur_round += 1
                pass

            proc.kill()
            logger.info('Finished processing %s ' % iobj)
            logger.info('Data read %s ' % iobj.data_read)
            logger.info('Read data hash %s ' % iobj.sha1.hexdigest())
        logger.info('Processing finished')

    def main(self):
        logger.debug('App started')

        parser = argparse.ArgumentParser(description='PolyDist')
        parser.add_argument('-t', '--threads', dest='threads', type=int, default=None,
                            help='Number of threads to use')

        parser.add_argument('--debug', dest='debug', action='store_const', const=True,
                            help='enables debug mode')

        parser.add_argument('--verbose', dest='verbose', action='store_const', const=True,
                            help='enables verbose mode')

        parser.add_argument('--ref', dest='reffile',
                            help='reference file with random data')

        parser.add_argument('--block', dest='blocklen',
                            help='block size in bits')

        parser.add_argument('--degree', dest='degree',
                            help='maximum degree of computation')

        parser.add_argument('--tv', dest='tvsize',
                            help='Size of one test vector, in this interpretation = number of bytes to read from file. '
                                 'Has to be aligned on block size')

        parser.add_argument('-r', '--rounds', dest='rounds',
                            help='Maximal number of rounds')

        parser.add_argument('--top', dest='topk', default=30, type=int,
                            help='top K number of best distinguishers to combine together')

        parser.add_argument('--comb-rand', dest='comb_random', default=0, type=int,
                            help='number of terms to add randomly to the combination set')

        parser.add_argument('--combine-deg', dest='combdeg', default=2, type=int,
                            help='Degree of combination')

        parser.add_argument('--conf', dest='conf', type=float, default=1.96,
                            help='Zscore failing threshold')

        parser.add_argument('--alldeg', dest='alldeg', action='store_const', const=True, default=False,
                            help='Add top K best terms to the combination group also for lower degree, not just top one')

        parser.add_argument('--poly', dest='polynomials', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='input polynomial to evaluate on the input data instead of generated one')

        parser.add_argument('--poly-file', dest='poly_file', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='input file with polynomials to test, one polynomial per line, in json array notation')

        parser.add_argument('--poly-ignore', dest='poly_ignore', action='store_const', const=True, default=False,
                            help='Ignore input polynomial variables out of range')

        parser.add_argument('--poly-mod', dest='poly_mod', action='store_const', const=True, default=False,
                            help='Mod input polynomial variables out of range')

        parser.add_argument('--no-comb-xor', dest='no_comb_xor', action='store_const', const=True, default=False,
                            help='Disables XOR combinations')

        parser.add_argument('--no-comb-and', dest='no_comb_and', action='store_const', const=True, default=False,
                            help='Disables AND combinations')

        parser.add_argument('--prob-comb', dest='prob_comb', type=float, default=1.0,
                            help='Probability the given combination is going to be chosen.')

        parser.add_argument('--test-randc', dest='test_randc', action='store_const', const=True, default=False,
                            help='Test randc generator')

        parser.add_argument('--test-java', dest='test_java', action='store_const', const=True, default=False,
                            help='Test java generator')

        parser.add_argument('--tests', dest='tests', type=int, default=100,
                            help='Number of tests to do')

        self.args = parser.parse_args()
        self.work()


# Launcher
app = None
if __name__ == "__main__":
    app = RandVerif()
    app.main()

