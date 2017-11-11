#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import scipy.misc
import scipy.stats

from . import common
from .booltest_main import *

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


# Main - argument parsing + processing
class RandVerif(Booltest):
    def __init__(self, *args, **kwargs):
        super(RandVerif, self).__init__(*args, **kwargs)
        self.args = None
        self.tester = None
        self.blocklen = None
        self.input_poly = []

    # noinspection PyBroadException
    def work(self):
        """
        Main entry point - data processing.

        RandVerif is used to benchmark particular data sources, with different seeds.
        It takes e.g., AES-CTR(SHA256(random)), runs it 1000 times and stores the results. This particular
        computation was used to determine reference z-scores of the test.

        Another usual scenario is to take Java util.Random, seed it 1000 times with a different random seed
        and analyze the results.

        RandVerif supports supplying custom distinguishers so the whole space is not searched. This
        setup is used to assess found distinguishers on more data / independent data streams with different seeds.

        RandVerif produces results on STDOUT, it contains multiple sections, separated by -----BEGIN SECTION-----
        Sections contain stats (avg. z-score), all z-scores, the best distinguishers for further analysis...
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

        top_distinguishers = []

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
        hwanalysis.skip_print_res = True
        hwanalysis.input_poly = self.input_poly
        hwanalysis.no_comb_and = self.args.no_comb_and
        hwanalysis.no_comb_xor = self.args.no_comb_xor
        hwanalysis.prob_comb = self.args.prob_comb
        hwanalysis.all_deg_compute = len(self.input_poly) == 0
        hwanalysis.do_only_top_comb = self.args.only_top_comb
        hwanalysis.do_only_top_deg = self.args.only_top_deg
        hwanalysis.no_term_map = self.args.no_term_map
        hwanalysis.use_zscore_heap = self.args.topterm_heap
        hwanalysis.sort_best_zscores = max(self.args.topterm_heap_k, top_k, 100)
        hwanalysis.best_x_combinations = self.args.best_x_combinations
        logger.info('Initializing test')
        hwanalysis.init()

        dist_result_map = {}
        for idx, poly in enumerate(self.input_poly):
            dist_result_map[idx] = []

        for test_idx in range(self.args.tests):
            seed = random.randint(0, 2**32-1)
            iobj = None
            if self.args.test_randc:
                path = os.path.realpath(os.path.join(script_path, '../assets/rndgen-c/rand'))
                cmd = '%s %s' % (path, seed)
                iobj = common.CommandStdoutInputObject(cmd=cmd, seed=seed, desc='randc-%s' % seed)

            elif self.args.test_randc_small:
                path = os.path.realpath(os.path.join(script_path, '../assets/rndgen-c-small/rand'))
                cmd = '%s %s' % (path, seed)
                iobj = common.CommandStdoutInputObject(cmd=cmd, seed=seed, desc='randc-small-%s' % seed)

            elif self.args.test_java:
                path = os.path.realpath(os.path.join(script_path, '../assets/rndgen-java/'))
                cmd = 'java -cp %s Main %s' % (path, seed)
                iobj = common.CommandStdoutInputObject(cmd=cmd, seed=seed, desc='randjava-%s' % seed)

            elif self.args.test_aes:
                iobj = common.AESInputObject(seed=seed)

            else:
                raise ValueError('No generator to test')

            size = iobj.size()
            logger.info('Testing input object: %s, size: %d kB, iteration: %d' % (iobj, size/1024.0, test_idx))

            # size smaller than TV? Adapt tv then
            if size >= 0 and size < tvsize:
                logger.info('File size is smaller than TV, updating TV to %d' % size)
                tvsize = size

            if tvsize*8 % self.blocklen != 0:
                rem = tvsize*8 % self.blocklen
                logger.warning('Input data size not aligned to the block size. '
                               'Input bytes: %d, block bits: %d, rem: %d' % (tvsize, self.blocklen, rem))
                tvsize -= rem//8
                logger.info('Updating TV to %d' % tvsize)

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

            res = hwanalysis.input_poly_last_res
            if res is not None and len(res) > 0:
                res_top = res[0]
                top_distinguishers.append((res_top, seed))

                for cur in res:
                    dist_result_map[cur.idx].append(cur.zscore)

            elif hwanalysis.last_res is not None and len(hwanalysis.last_res) > 0:
                res_top = hwanalysis.last_res[0]
                top_distinguishers.append((res_top, seed))

            else:
                raise ValueError('No data from the analysis')

            logger.info('Finished processing %s ' % iobj)
            logger.info('Data read %s ' % iobj.data_read)
            logger.info('Read data hash %s ' % iobj.sha1.hexdigest())

        all_zscores = []
        print('-----BEGIN JSON-----')
        js = []
        for dist in top_distinguishers:
            cr = collections.OrderedDict()
            cr['z'] = dist[0].zscore
            try:
                cr['d'] = dist[0].idx
            except:
                pass

            cr['seed'] = dist[1]
            js.append(cr)
            all_zscores.append(dist[0].zscore)
        print(json.dumps(js, indent=2))

        print('-----BEGIN JSON-STATS-----')
        js = []
        for idx in dist_result_map:
            cur = dist_result_map[idx]
            cr = collections.OrderedDict()
            cr['idx'] = idx
            cr['poly'] = common.poly2str(self.input_poly[idx])
            cr['avg'] = sum([abs(x) for x in cur])/float(len(cur))
            cr['cnt'] = len(cur)
            cr['zscores'] = cur
            js.append(cr)
        print(json.dumps(js, indent=2))

        print('-----BEGIN RUN-CONFIG-----')
        js = collections.OrderedDict()
        js['block'] = self.blocklen
        js['deg'] = deg
        js['top_comb'] = top_comb
        js['top_k'] = top_k
        js['tvsize'] = tvsize
        js['tests'] = self.args.tests
        js['prob_comb'] = self.args.prob_comb
        js['all_deg'] = all_deg
        print(json.dumps(js))

        print('-----BEGIN Z-SCORES-NORM-----')
        print(all_zscores)
        print('-----BEGIN Z-SCORES-ABS-----')
        print([abs(x) for x in all_zscores])
        print('-----BEGIN Z-SCORES-AVG-----')
        print(sum([abs(x) for x in all_zscores])/float(len(all_zscores)))
        print('-----BEGIN Z-SCORES-NAVG-----')
        print(sum([x for x in all_zscores])/float(len(all_zscores)))

        if self.args.csv_zscore:
            print('-----BEGIN Z-SCORES-CSV-----')
            print('zscore')
            for x in [abs(x) for x in all_zscores]:
                print(x)

        logger.info('Processing finished')

    def main(self):
        logger.debug('App started')

        parser = argparse.ArgumentParser(description='RandVerif benchmarks particular data sources, with different seeds.')
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

        parser.add_argument('--only-top-comb', dest='only_top_comb', action='store_const', const=True, default=False,
                            help='If set only the top combination is performed, otherwise all up to given combination degree')

        parser.add_argument('--only-top-deg', dest='only_top_deg', action='store_const', const=True, default=False,
                            help='If set only the top degree if base polynomials combinations are considered, otherwise '
                                 'also lower degrees are input to the topk for next state - combinations')

        parser.add_argument('--no-term-map', dest='no_term_map', action='store_const', const=True, default=False,
                            help='Disables term map precomputation, uses unranking algorithm instead')

        parser.add_argument('--prob-comb', dest='prob_comb', type=float, default=1.0,
                            help='Probability the given combination is going to be chosen.')

        parser.add_argument('--topterm-heap', dest='topterm_heap', action='store_const', const=True, default=False,
                            help='Use heap to compute best X terms for stats & input to the combinations')

        parser.add_argument('--topterm-heap-k', dest='topterm_heap_k', default=None, type=int,
                            help='Number of terms to keep in the heap')

        parser.add_argument('--best-x-combs', dest='best_x_combinations', default=None, type=int,
                            help='Number of best combinations to return. If defined, heap is used')

        parser.add_argument('--csv-zscore', dest='csv_zscore', action='store_const', const=True, default=False,
                            help='CSV output with zscores')

        parser.add_argument('--test-randc', dest='test_randc', action='store_const', const=True, default=False,
                            help='Test randc generator')

        parser.add_argument('--test-randc-small', dest='test_randc_small', action='store_const', const=True, default=False,
                            help='Test randc_small generator')

        parser.add_argument('--test-java', dest='test_java', action='store_const', const=True, default=False,
                            help='Test java generator')

        parser.add_argument('--test-aes', dest='test_aes', action='store_const', const=True, default=False,
                            help='AES test')

        parser.add_argument('--tests', dest='tests', type=int, default=100,
                            help='Number of tests to do')

        self.args = parser.parse_args()
        self.work()


# Launcher
app = None
if __name__ == "__main__":
    app = RandVerif()
    app.main()

