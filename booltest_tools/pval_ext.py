import argparse
import json
import os
import glob
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO)

parser = argparse.ArgumentParser(description='BoolTest pval extractor')
parser.add_argument('--pval', dest='pval', type=float, default=1/40000.,
                    help='Threshold pvalue')
parser.add_argument('--full', dest='full', action='store_const', const=True,
                    help='full output')
parser.add_argument('--pattern', dest='pattern', default=None,
                    help='Search pattern instead of files')
parser.add_argument('files', nargs=argparse.ZERO_OR_MORE, default=[],
                    help='files to process')
args = parser.parse_args()


def getcfg(js):
    try:
        return js['config']['config']['spec']['gen_cfg']['file_name']
    except:
        return None


files = args.files
if args.pattern:
    files += sorted(glob.glob(args.pattern))

for file in files:
    try:
        js = json.load(open(file))
        if 'booltest_res' not in js:
            continue
        if js['data_read'] <= 10:
            continue
        bres = js['booltest_res']
        if len(bres) <= 1 or 'halvings' not in bres[1]:
            continue
        if bres[1]['halvings'][0]['pval'] >= args.pval:
            continue

        cfg = getcfg(js)
        bid = '-'.join((file.split('.')[0]).split('-')[-4:]) if not args.full else file
        dist = bres[1]['halvings'][0]
        cres = ((cfg if cfg else file, bid, dist['pval'], (dist['poly'])))
        print(json.dumps(cres))

    except Exception as e:
        logger.error('Could not process %s: %s' % (file, e), exc_info=e)
