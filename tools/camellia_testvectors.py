import requests
import re


def striphex(line):
    line = re.sub(r'^.+?:\s+', '', line)
    return line.replace(' ', '')


r = requests.get('https://info.isl.ntt.co.jp/crypt/eng/camellia/dl/cryptrec/t_camellia.txt')
tv_text = r.text

tv_lines = tv_text.split('\n')[2:]
tests = [[], []]
cur_key = None
cur_pl = None

for line in tv_lines:
    line = line.strip()
    if line == '':
        continue
    if line.startswith('Camellia'):
        continue

    fch = line[0].lower()
    if fch == 'k':
        cur_key = striphex(line)
    elif fch == 'p':
        cur_pl = striphex(line)
    elif fch == 'c':
        cip = striphex(line)
        test_idx = 0 if len(cur_key) == 2*16 else 1
        tests[test_idx].append((cur_key, cur_pl, cip))
    else:
        raise ValueError('wtf prefix')


for tidx in range(2):
    tname = 'test-vectors-%d.txt' % (18 if tidx == 0 else 24)
    with open(tname, 'w+') as fh:
        for test in tests[tidx]:
            fh.write('%s\n%s\n%s\n\n' % (test[0],test[1], test[2]))

