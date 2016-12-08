import argparse
import logging, coloredlogs
import common

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
        blocklen = self.defset(self.args.blocklen, 128)
        deg = self.defset(self.args.degre, 3)
        reffile = self.defset(self.args.reffile)

        self.tester = common.Tester(reffile=reffile)



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

        parser.add_argument('files', nargs=argparse.ZERO_OR_MORE, default=[],
                            help='files to process')

        self.args = parser.parse_args()
        self.work()


# Launcher
app = None
if __name__ == "__main__":
    app = App()
    app.main()

