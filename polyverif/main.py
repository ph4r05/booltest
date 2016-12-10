import argparse
import logging, coloredlogs
import common
import os

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.DEBUG)


# Main - argument parsing + processing
class App(object):
    def __init__(self, *args, **kwargs):
        self.args = None
        self.tester = None

    def defset(self, val, default=None):
        return val if val is not None else default

    def work(self):
        blocklen = int(self.defset(self.args.blocklen, 128))
        deg = int(self.defset(self.args.degre, 3))
        tvsize_orig = long(self.defset(self.args.tvsize, 1024*256))
        zscore_thresh = float(self.args.conf)
        reffile = self.defset(self.args.reffile)
        #self.tester = common.Tester(reffile=reffile)

        for file in self.args.files:
            tvsize = tvsize_orig

            if not os.path.exists(file):
                logger.error('File does not exist: %s' % file)

            size = os.path.getsize(file)
            logger.info('Testing file: %s, size: %d kB' % (file, size/1024.0))

            # size smaller than TV? Adapt tv then
            if size < tvsize:
                logger.warning('File size is smaller than TV, updating TV to %d' % size)
                tvsize = size

            term_eval = common.TermEval(blocklen=blocklen, deg=deg)

            # read the file until there is no data.
            with open(file, 'r') as fh:
                data_read = 0
                round = 0
                while data_read < size:
                    data = fh.read(tvsize)
                    bits = common.to_bitarray(data)

                    # pre-compute
                    logger.info('Pre-computing with TV, deg: %d, blocklen: %04d, tvsize: %08d, round: %d' %
                                (deg, blocklen, tvsize, round))

                    term_eval.load(bits)
                    round += 1

                    # evaluate all terms of the given degree
                    logger.info('Evaluating all terms')
                    probab = term_eval.expp_term_deg(deg)
                    exp_count = term_eval.cur_evals * probab
                    hws = term_eval.eval_terms(deg)

                    difs = [(abs(x-exp_count), idx) for idx,x in enumerate(hws)]
                    difs.sort(key=lambda x: x[0], reverse=True)
                    zscores = [common.zscore(x, exp_count, term_eval.cur_evals) for x in hws]

                    # top x diffs
                    for x in difs[0:5]:
                        observed = hws[x[1]]
                        zscore = common.zscore(observed, exp_count, term_eval.cur_evals)
                        fail = 'x' if abs(zscore) > zscore_thresh else ''
                        print(' - zscore: %+05.5f, observed: %08d, expected: %08d %s'
                              % (zscore, observed, exp_count, fail))

                    for x in difs[-5:]:
                        observed = hws[x[1]]
                        zscore = common.zscore(observed, exp_count, term_eval.cur_evals)
                        fail = 'x' if abs(zscore) > zscore_thresh else ''
                        print(' - zscore: %+05.5f, observed: %08d, expected: %08d %s'
                              % (zscore, observed, exp_count, fail))

                    mean = sum(hws)/float(len(hws))
                    mean_zscore = sum(zscores)/float(len(zscores))

                    fails = sum([1 for x in zscores if abs(x) > zscore_thresh])
                    print('Mean value: %s' % mean)
                    print('Mean zscore: %s' % mean_zscore)
                    print('Num of fails: %s = %02f.5%%' % (fails, 100.0*float(fails)/len(zscores)))


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
        parser.add_argument('--degre', dest='degre',
                            help='maximum degre of computation')
        parser.add_argument('--tv', dest='tvsize',
                            help='Size of one test vector')
        parser.add_argument('--conf', dest='conf', type=float, default=1.96,
                            help='Zscore failing threshold')

        parser.add_argument('--stdin', dest='verbose', action='store_const', const=True,
                            help='read data from STDIN')

        parser.add_argument('files', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='files to process')

        self.args = parser.parse_args()
        self.work()


# Launcher
app = None
if __name__ == "__main__":
    app = App()
    app.main()

