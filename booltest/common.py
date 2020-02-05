#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from past.builtins import basestring
from past.builtins import xrange

import hashlib
import logging
import math
import os
import json
import random
import signal
import subprocess
import sys
import binascii
import collections
from functools import reduce
from booltest import jsonenc

import scipy.misc
import ufx.uf_hash as ufh

if sys.version_info >= (3, 2):
    from functools import lru_cache
else:
    from repoze.lru import lru_cache

from repoze.lru import LRUCache

from booltest.crypto_util import aes_ecb, dump_uint
from booltest import input_obj
from booltest.jsonenc import NoIndent
from booltest.jsonenc import unwrap_rec


if hasattr(scipy.misc, 'comb'):
    scipy_comb = scipy.misc.comb
else:
    import scipy.special
    scipy_comb = scipy.special.comb


# Enables bitarray - with native C extension
FAST_IMPL = True

# Enables bitarray - with native C extension with eval_monic()
FAST_IMPL_PH4 = True


if FAST_IMPL:
    import bitarray
else:
    from bitstring import Bits, BitArray, BitStream, ConstBitStream


logger = logging.getLogger(__name__)


Combined = collections.namedtuple('Combined', ['poly', 'expp', 'exp_cnt', 'obs_cnt', 'zscore'])
CombinedIdx = collections.namedtuple('CombinedIdx', ['poly', 'expp', 'exp_cnt', 'obs_cnt', 'zscore', 'idx'])
ValueIdx = collections.namedtuple('ValueIdx', ['value', 'idx'])


def comb2dict(comb, indent_fix=False):
    poly = immutable_poly(comb.poly)
    return collections.OrderedDict([
        ('expp', comb.expp),
        ('exp_cnt', comb.exp_cnt),
        ('obs_cnt', comb.obs_cnt),
        ('diff', (100.0 * (comb.exp_cnt - comb.obs_cnt) / comb.exp_cnt) if comb.exp_cnt else None),
        ('zscore', comb.zscore),
        ('poly', NoIndent(poly) if indent_fix else poly)
    ])


def immutable_poly(poly):
    if not poly:
        return poly
    return tuple([tuple(x) for x in poly])


def mutable_poly(poly):
    if not poly:
        return poly
    return [list(x) for x in poly]


def jsunwrap(val):
    if isinstance(val, Combined):
        return Combined(unwrap_rec(val.poly, jsunwrap), *val[1:])
    elif isinstance(val, CombinedIdx):
        return CombinedIdx(unwrap_rec(val.poly, jsunwrap), *val[1:])
    else:
        return unwrap_rec(val, jsunwrap)


def jswrap(val):
    if isinstance(val, Combined):
        return Combined(NoIndent(val.poly), *val[1:])
    elif isinstance(val, CombinedIdx):
        return CombinedIdx(NoIndent(val.poly), *val[1:])
    elif isinstance(val, list):
        return [jswrap(x) for x in val]
    elif isinstance(val, tuple):
        return tuple([jswrap(x) for x in val])
    elif isinstance(val, collections.OrderedDict):
        return collections.OrderedDict([(x, jswrap(val[x])) for x in val])
    elif isinstance(val, dict):
        return dict([(x, jswrap(val[x])) for x in val])
    else:
        return val


def noindent_poly(val, key=None):
    if key and key == 'poly':
        return NoIndent(val)
    elif isinstance(val, list):
        return [noindent_poly(x) for x in val]
    elif isinstance(val, tuple):
        return tuple([noindent_poly(x) for x in val])
    elif isinstance(val, collections.OrderedDict):
        return collections.OrderedDict([(x, noindent_poly(val[x], x)) for x in val])
    elif isinstance(val, dict):
        return dict([(x, noindent_poly(val[x], x)) for x in val])
    else:
        return val


def pos_generator(spec=None, dim=None, maxelem=None):
    """
    Creates a generator that iterates over the specified range of positions.
    Combinatorial generator.

    e.g., if spec = [255,255,255] then generator iterates over 3 groups of 255.
    spec is the maximal element generated.
    :param spec:
    :param dim:
    :param maxelem:
    :return:
    """
    if spec is None:
        if dim is not None and maxelem is not None:
            spec = [maxelem] * dim
        else:
            raise ValueError('Parameter specification invalid')

    ln = len(spec)
    idx = [0] * ln
    while True:
        yield idx

        # increment with overflow
        c = ln-1
        while c >= 0:
            idx[c] += 1
            if idx[c] <= spec[c]:
                break

            # Overflow, reset the current digit, increment the next one
            idx[c] = 0
            c -= 1
        if c < 0:
            return


def term_generator(deg, maxelem, prob_choose=1.0, idx=None, clone=False):
    """
    Generates all terms of the given degree with given max len.

    e.g. for deg = 3, ln = 9:
    [0,1,2] .. [7,8,9]
    :param deg:
    :param maxelem:
    :param prob_choose: probability the given element will be chosen
    :return:
    """
    idx = idx if idx else [0] * deg
    for i in range(deg):
        idx[i] = i
        if i > maxelem:
            raise ValueError('deg too big for the maxelem')

    while True:
        if prob_choose >= 1.0 or random.random() < prob_choose:
            yield idx if not clone else list(idx)

        # increment with overflow
        c = deg - 1
        while c >= 0:
            idx[c] += 1
            if idx[c] <= maxelem-deg+c+1:
                break

            # Overflow, reset the current digit, increment the next one
            idx[c] = idx[c - 1] + 2 if c > 0 else maxelem-deg
            for cc in range(max(1, c+1), deg):
                idx[cc] = idx[cc - 1] + 1
            c -= 1
        if c < 0:
            return


@lru_cache(maxsize=1024)
def comb(n, k, exact=False):
    return scipy_comb(n, k, exact=exact)


def zscore(observed, expected, N):
    """
    Computes z-score for the normal distribution
    :param observed: number of occurrences
    :param expected: number of occurrences
    :param N: sample size
    :return:
    """
    observed = float(observed) / float(N)
    expected = float(expected) / float(N)
    return (observed-expected) / math.sqrt((expected*(1.0-expected))/float(N))


@lru_cache(maxsize=32)
def zscore_denominator(expected, N):
    """
    Computes denominator for the zscore computation, denominator is fixed for expected, N
    :param expected:
    :param N:
    :return:
    """
    expected = float(expected) / float(N)
    return math.sqrt((expected*(1.0-expected))/float(N))


def zscore_den(observed, expected, N, denom):
    """
    Computes zscore with precomputed denominator.
    :param observed:
    :param expected:
    :param N:
    :param denom:
    :return:
    """
    x = float(observed - expected) / float(N)
    return x / denom


def zscore_p(observed, expected, N):
    """
    Computes z-score for the normal distribution
    :param observed: prob.
    :param expected: prob.
    :param N:
    :return:
    """
    return (observed-expected) / math.sqrt((expected*(1.0-expected))/float(N))

def expp_xor_indep(p1, p2):
    """
    Probability of term t1 XOR t2 being 1 if t1 is 1 with p1 and t2 is 1 with p2.
    t1 and t2 has to be independent (no common sub-term).
    Due to associativity can be computed on multiple terms: t1 ^ t2 ^ t3 ^ t4 = (((t1 ^ t2) ^ t3) ^ t4) - zipping.

    XOR:
      a b | r
      ----+---
      1 1 | 0
      1 0 | 1  = p1    * (1-p2)
      0 1 | 1  = (1-p1)* p2
      0 0 | 0

    :param p1:
    :param p2:
    :return:
    """
    return p1*(1-p2)+(1-p1)*p2


def empty_bitarray(size=None):
    """
    Initializes empty bit array (of given size)
    :param size:
    :return:
    """
    if FAST_IMPL:
        if size is None:
            return bitarray.bitarray()
        else:
            a = bitarray.bitarray(size)
            a.setall(False)
            return a
    else:
        if size is None:
            return BitArray()
        else:
            return BitArray(uint=0, length=size)


def to_bitarray(inp=None, const=True):
    """
    Converts input to bitarray for computation with TermEval
    :return:
    """
    if FAST_IMPL:
        if isinstance(inp, basestring):
            a = bitarray.bitarray()
            a.frombytes(inp)
            return a
        elif isinstance(inp, bitarray.bitarray):
            a = bitarray.bitarray(inp)
            return a
        else:
            raise ValueError('Unknown input')
    else:
        constructor = Bits if const else BitArray
        if isinstance(inp, basestring):
            return constructor(bytes=inp)
        elif isinstance(inp, (Bits, BitStream, BitArray, ConstBitStream)):
            return constructor(inp)
        else:
            raise ValueError('Unknown input')


def clone_bitarray(other, src=None):
    """
    Fast clone of the bit array. The actual function used depends on the implementation
    :param other:
    :param src:
    :return:
    """
    if FAST_IMPL_PH4 and src is not None:
        src.fast_copy(other)
        return src

    return to_bitarray(other)


def hw(bits):
    """
    Hamming weight of the bitarray
    :param bits:
    :return:
    """
    if FAST_IMPL:
        return bits.count()
    else:
        return bits.count(True)


def build_term_map(deg, blocklen):
    """
    Builds term map (degree, index) -> term

    :param deg:
    :param blocklen:
    :return:
    """
    term_map = [[0] * comb(blocklen, x, True) for x in range(deg + 1)]
    for dg in range(1, deg + 1):
        for idx, x in enumerate(term_generator(dg, blocklen - 1)):
            term_map[dg][idx] = x
    return term_map


@lru_cache(maxsize=8192)
def comb_cached(n, k):
    """
    Computes C(n,k) - combinatorial number of N elements, k choices
    :param n:
    :param k:
    :return:
    """
    if (k > n) or (n < 0) or (k < 0):
        return 0
    val = 1
    for j in range(min(k, n - k)):
        val = (val * (n - j)) // (j + 1)
    return val


def rank(s, n):
    """
    Returns index of the combination s in (N,K)
    https://computationalcombinatorics.wordpress.com/2012/09/10/ranking-and-unranking-of-combinations-and-permutations/
    :param s:
    :return:
    """
    k = len(s)
    r = 0
    for i in range(0, k):
        for v in range(s[i-1]+1 if i > 0 else 0, s[i]):
            r += comb_cached(n - v - 1, k - i - 1)
    return r


@lru_cache(maxsize=8192)
def unrank(i, n, k):
    """
    returns the i-th combination of k numbers chosen from 0,2,...,n-1, indexing from 0
    """
    c = []
    r = i+0
    j = 0
    for s in range(1, k+1):
        cs = j+1

        while True:
            if n - cs < 0:
                raise ValueError('Invalid index')
            decr = comb(n - cs, k - s)
            if r > 0 and decr == 0:
                raise ValueError('Invalid index')
            if r - decr >= 0:
                r -= decr
                cs += 1
            else:
                break

        c.append(cs-1)
        j = cs
    return c


def are_disjoint(sets):
    full_count = sum([len(x) for x in sets])
    res = set()
    for x in sets:
        res |= x
    return len(res) == full_count


def union(sets):
    res = set()
    for x in sets:
        res |= x
    return res


def get_poly_cache_key(poly):
    return tuple(tuple(term) for term in poly)


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def poly2str(poly):
    """
    Converts polynomial to a string representation.
    :param poly:
    :return:
    """
    terms = []
    for term in poly:
        vars = ''.join(['x_{%d}' % x for x in term])
        terms.append(vars)
    return ' + '.join(terms)


def range2(*args):
    """
    xrange optimization, py2, py3 compatible, takes xrange limits into account
    :param args:
    :return:
    """
    largs = len(args)
    idx_from = 0
    idx_to = 0
    if largs > 1:
        idx_from = args[0]
        idx_to = args[1]
    elif largs == 1:
        idx_to = args[0]
    else:
        raise ValueError('No args in range()')

    if idx_from > 2147483647 or idx_to > 2147483647:
        return range(idx_from, idx_to)
    else:
        return xrange(idx_from, idx_to)


def replace_none(iterable, replacement=0):
    """
    replaces None in the iterable with replacement 0
    :param iterable:
    :param replacement:
    :return:
    """
    return [x if x is not None else replacement for x in iterable]


def merge_dicts(dicts):
    """
    Merges dictionaries
    :param dicts:
    :return:
    """
    dres = {}
    for dc in dicts:
        dres.update(dc)
    return dres


def copy_poly(dst, src):
    dst = dst if dst else [None]*len(src)
    dst[:] = src[:]
    return dst


class AutoJSONEncoder(jsonenc.IndentingJSONEncoder):
    """
    JSON encoder trying to_json() first
    """
    def unwrap(self, obj):
        return jsunwrap(obj)

    def default(self, obj):
        try:
            return obj.to_json()
        except AttributeError:
            return self.default_classic(obj)

    def default_classic(self, o):
        if isinstance(o, set):
            return list(o)
        elif isinstance(o, bytes):
            return o.decode('utf8')
        else:
            try:
                return super(AutoJSONEncoder, self).default(o)
            except:
                return str(o)


def json_dumps(obj, **kwargs):
    """
    Json dump with the encoder
    :param obj:
    :param kwargs:
    :return:
    """
    return json.dumps(obj, cls=AutoJSONEncoder, **kwargs)


def json_dump(obj, fp, **kwargs):
    return json.dump(obj, fp, cls=AutoJSONEncoder, **kwargs)


def try_json_dumps(obj, **kwargs):
    """
    Json dump with the encoder
    :param obj:
    :param kwargs:
    :return:
    """
    try:
        return json_dumps(obj, **kwargs)
    except:
        pass


def try_json_load(data):
    try:
        return json.loads(data)
    except:
        return None


def defval(val, default=None):
    """
    Returns val if is not None, default instead
    :param val:
    :param default:
    :return:
    """
    return val if val is not None else default


def defvalkey(js, key, default=None, take_none=True):
    """
    Returns js[key] if set, otherwise default. Note js[key] can be None.
    :param js:
    :param key:
    :param default:
    :param take_none:
    :return:
    """
    if js is None:
        return default
    if key not in js:
        return default
    if js[key] is None and not take_none:
        return default
    return js[key]


def defvalkeys(js, key, default=None):
    """
    Returns js[key] if set, otherwise default. Note js[key] can be None.
    Key is array of keys. js[k1][k2][k3]...

    :param js:
    :param key:
    :param default:
    :param take_none:
    :return:
    """
    if js is None:
        return default
    if not isinstance(key, (tuple, list)):
        key = key.split('.')
    try:
        cur = js
        for ckey in key:
            cur = cur[ckey]
        return cur
    except:
        pass
    return default


def generate_seed(iteration=0):
    """
    Deterministic seed generation for experiment comparison.
    :param iteration:
    :return:
    """
    seed0 = b'1fe40505e131963c'
    if iteration == 0:
        return seed0

    in_block = bytearray(dump_uint(iteration))
    in_block = in_block + bytearray([0] * (16 - len(in_block)))

    aes = aes_ecb(hashlib.sha256(seed0).digest())
    seed = aes.encrypt(bytes(in_block))[:8]
    return binascii.hexlify(seed)


# Re-exports, compatibility


InputObject = input_obj.InputObject
FileInputObject = input_obj.FileInputObject
StdinInputObject = input_obj.StdinInputObject
FileLikeInputObject = input_obj.FileLikeInputObject
CommandStdoutInputObject = input_obj.CommandStdoutInputObject
AESInputObject = input_obj.AESInputObject
BinaryInputObject = input_obj.BinaryInputObject
LinkInputObject = input_obj.LinkInputObject


class TermEval(object):
    def __init__(self, blocklen=128, deg=1, *args, **kwargs):
        # block length in bits, term size.
        self.blocklen = blocklen

        # term degree
        self.deg = deg

        # evaluated base terms of deg=1
        self.base = []
        self.cur_tv_size = None
        self.cur_evals = None
        self.last_base_size = None

        # caches
        self.sim_norm_cache = LRUCache(8192)
        self.pows = [math.pow(2, -1*x) for x in range(deg + 1)]
        self.bbase = None
        self.base_buff_size = 4096

    def base_size(self):
        """
        Returns base size of the vector - same as size of the base
        :return:
        """
        return len(self.base[0])

    def new_buffer(self):
        """
        Returns the allocated bitarray - non-intialized of the size of the base.
        :return:
        """
        return empty_bitarray(len(self.base[0]))

    def gen_term(self, indices, blocklen=None):
        """
        Generates a bit term mask from the indices created by term_generator().
        blocklen wide.
        :param indices: array of bit indices, e.g., [0,8] -> x0x8
        :param blocklen: length of the term bit representation. If none, default block len is used
        :return: bit representation of the term
        """
        if blocklen is None:
            blocklen = self.blocklen

        term = empty_bitarray(blocklen)
        for bitpos in indices:
            term[bitpos] = 1
        return term

    def mask_with_term(self, term, block):
        """
        Masks input with the term.
        block has to be a multiple of the term size. term is evaluated by sliding window of size=term.
        :param term: bit representation of the term
        :param block: bit representation of the input
        :return: bit representation of the result, size = size of block.
        """
        ln = len(block)
        lnt = len(term)
        res = empty_bitarray()
        for idx in range(0, ln, lnt):
            res.append(block[idx:idx + self.blocklen] & term)
        return res

    def term_bit_to_idx(self, term):
        return [idx for idx, x in enumerate(term) if x]

    def term_idx_to_bit(self, term):
        res = empty_bitarray(self.blocklen)
        for x in term:
            res[x] = True
        return res

    def poly_bit_to_idx(self, poly):
        return [self.term_bit_to_idx(term) for term in poly]

    def poly_idx_to_bit(self, poly):
        return [self.term_idx_to_bit(term) for term in poly]

    def eval_term_idx_single(self, term, block):
        """
        Evaluates term on the raw input - bit array. Uses [] operator to access bits in block.
        Returns a single number, evaluates polynomial on single block
        :param term: term in the index notation
        :param block: bitarray indexable by []
        :return:
        """
        cval = 1
        for idx in term:
            if block[idx] == 0:
                cval = 0
                break
        return cval

    def eval_term_bit_data(self, term, data, res=None):
        """
        Evaluates term on the bitarray input. Uses & on the whole term and the block slices.
        Block has to be a multiple of the term size. term is evaluated by sliding window of size=term.
        In result each bit represents a single term evaluation on the given sub-block
        :param term: bit representation of the term
        :param data: bit representation of the input, bit array of term evaluations. size = size of block / size of term.
        :param res: res bitarray buffer
        :return: bitarray with results
        """
        ln = len(data)
        lnt = len(term)
        res = empty_bitarray(ln // lnt) if res is None else res

        ctr = 0
        for idx in range(0, ln, lnt):
            res[ctr] = ((data[idx:idx + self.blocklen] & term) == term)
            ctr += 1
        return res

    def eval_poly_idx_single(self, poly, block):
        """
        Evaluates polynomial on the raw input - bit array. Uses [] operator to access bits in block.
        Returns a single number, evaluates polynomial on single block
        :param poly: polynomial in the index notation
        :param block: bitarray indexable by []
        :return: hamming weight of the result.
        """
        res = 0

        # for each term &&-operation
        for term in poly:
            cval = 1
            for idx in term:
                if block[idx] == 0:
                    cval = 0
                    break
            res ^= cval
        return res

    def eval_poly_bit_data(self, poly, data, res=None, sres=None):
        """
        Evaluates polynomial stored as [bitarray0, bitarray1] on the input block
        :param poly:
        :param data:
        :param res: bitarray buffer for the result (accumulator)
        :param sres: bitarray buffer for term eval
        :return: bitarray of results
        """
        res = empty_bitarray(len(data) // self.blocklen) if res is None else res
        sres = empty_bitarray(len(data) // self.blocklen) if sres is None else sres
        for tidx in range(0, len(poly)):
            res ^= self.eval_term_bit_data(poly[tidx], data, sres)
        return res

    def eval_poly_idx_data(self, poly, data):
        """
        Evaluates polynomial on the input data, without precomputed base
        :param poly:
        :param data: bitarray
        :return: hamming weight of the result.
        """
        res = 0
        ln = len(data)
        for idx in range(0, (ln // self.blocklen) * self.blocklen, self.blocklen):
            res += self.eval_poly_idx_single(poly, data[idx:idx + self.blocklen])
        return res

    def eval_polys_bit_data(self, polys, data):
        """
        Evaluates multiple bit polynomials over data with single pass over the data.
        Single pass supports generated data.
        :param polys: array of terms, bitarray notation
        :param data:
        :return:
        """
        ln = len(data)
        res_size = ln // self.blocklen
        ctr = -1

        res = [empty_bitarray(res_size) for _ in range(len(polys))]
        for idx in range(0, (ln // self.blocklen) * self.blocklen, self.blocklen):
            cdata = data[idx:idx + self.blocklen]
            ctr += 1

            for pidx, poly in enumerate(polys):
                vi = 0
                for term in poly:
                    vi ^= ((cdata & term) == term)
                res[pidx][ctr] = vi
        return res

    def eval_polys_bit_data_basis(self, polys, data):
        """
        Similar to eval_polys_bit_data but builds monomial basis first, then evaluates the polynomial
        over the basis. Is faster if there are large amount of polynomials to evaluate.

        :param polys:
        :param data:
        :return:
        """
        res = [None] * len(polys)
        sub_eval = self.load_base_new(data)

        buff_subres = sub_eval.new_buffer()
        for pidx, poly in enumerate(polys):
            cres = sub_eval.eval_poly(poly, None, buff_subres)
            res[pidx] = cres
        return res

    def eval_polys_idx_data(self, polys, data):
        """
        Evaluates multiple bit polynomials over data with single pass over the data.
        Single pass supports generated data.
        :param polys: array of terms, index notation
        :param data:
        :return: [pidx => hamming weight]
        """
        ln = len(data)
        ctr = -1

        res = [0] * len(polys)
        for idx in range(0, (ln // self.blocklen) * self.blocklen, self.blocklen):
            cdata = data[idx:idx + self.blocklen]
            ctr += 1

            for pidx, poly in enumerate(polys):
                vi = 0
                for term in poly:
                    cval = 1
                    for tidx in term:
                        if cdata[tidx] == 0:
                            cval = 0
                            break
                    vi ^= cval
                res[pidx] += vi

        return res

    def eval_polys_idx_data_strategy(self, polys, data, to_bits=None):
        """
        Evaluates polynomial collection on the input data, producing hamming weight list, value per polynomial.
        Picks strategy depending on the number of polynomials and.
        :param polys:
        :param data:
        :param to_bits: manual strategy selection. If true, data is loaded as basis, poly evaluated on basis
        :return:
        """
        if to_bits is None:
            to_bits = len(polys) * sum(len(x) for x in polys[0]) >= self.blocklen

        polys = [self.poly_idx_to_bit(poly) for poly in polys]
        if to_bits:  # convert to terms, build base, eval on base
            return self.eval_polys_bit_data_basis(polys, data)

        else:  # eval individually, with one pass over data - faster, support generators
            return self.eval_polys_bit_data(polys, data)

    def hw(self, block):
        """
        Computes hamming weight of the block
        :param block: bit representation of the input
        :return:
        """
        return hw(block)

    def term_generator(self, deg=None, idx=None):
        """
        Returns term generator for given deg (internal if none is given) and blocklen
        :return:
        """
        if deg is None:
            deg = self.deg

        return term_generator(deg, self.blocklen-1, 1, idx)

    def load(self, block, **kwargs):
        """
        Precomputes data
        :param block:
        :return:
        """
        self.gen_base(block, **kwargs)

    def gen_base(self, block, eval_only_vars=None, **kwargs):
        """
        Generate base for term evaluation from the block.
        Evaluates each base term (deg=1) on the input, creates a base for further evaluation of high order terms.
        :param block: bit representation of the input
        :param eval_only_vars: if not None, evals only those variables mentioned
        :return:
        """
        if (len(block) % self.blocklen) != 0:
            raise ValueError('Input data not multiple of block length, %s vs. %s' % (len(block), self.blocklen))

        self.cur_tv_size = len(block)/8
        self.cur_evals = len(block) / self.blocklen

        ln = len(block)
        res_size = ln // self.blocklen

        if self.base is None or self.last_base_size != (self.blocklen, res_size):
            self.base = [None] * self.blocklen

        for bitpos in range(0, self.blocklen):
            ctr = 0
            if bitpos != 0 and eval_only_vars is not None and bitpos not in eval_only_vars:
                continue

            if self.last_base_size != (self.blocklen, res_size):
                self.base[bitpos] = empty_bitarray(res_size)

            if FAST_IMPL_PH4:
                self.base[bitpos].eval_monic(block, bitpos, self.blocklen)

            else:
                # For verification purposes we have also another independent evaluation here.
                for idx in range(0, ln, self.blocklen):
                    self.base[bitpos][ctr] = block[idx+bitpos] == 1
                    ctr += 1
                assert ctr == res_size

            if not FAST_IMPL:
                self.base[bitpos] = Bits(self.base[bitpos])

        self.last_base_size = (self.blocklen, res_size)
        if FAST_IMPL_PH4:
            from bitarray import tbase
            self.bbase = tbase(self.base, self.base_buff_size)

    def load_base_new(self, data):
        sub_eval = TermEval(self.blocklen, self.deg)
        sub_eval.load(data)
        return sub_eval

    def num_terms(self, deg, include_all_below=False, exact=False):
        """
        Computes number of terms of given degree.
        :param deg:
        :param include_all_below: if true, all lower degree counts are summed
        :param exact: if true exact value is computed. Otherwise just approximation is given (larger).
        :return:
        """
        if deg == 1:
            return self.blocklen

        rng = range(1 if include_all_below else deg, deg+1)
        if exact:
            return sum([(comb(self.blocklen, x, True)) for x in rng])
        else:
            return sum([(comb(self.blocklen, x, False) + 2) for x in rng])

    def eval_term(self, term, res=None):
        """
        Evaluates term on the block using the precomputed base.
        :param term: term represented as an array of bit positions
        :param res: bitarray buffer to put result to
        :return: bit representation of the result, each bit represents single term evaluation on the given sub-block
        """
        ln = len(term)
        idx_start = 1
        if res is None:
            res = to_bitarray(self.base[term[0]], const=False)
        else:
            if FAST_IMPL_PH4:
                res.fast_copy(self.base[term[0]])
            else:
                idx_start = 0
                res.setall(True)

        for i in range(idx_start, ln):
            res &= self.base[term[i]]
        return res

    def eval_terms(self, deg=None):
        """
        Evaluates all terms on the input data precomputed in the base.
        Returns array of hamming weights.
        :param deg: degree of the terms to generate. If none, default degree is taken.
        :return: array of hamming weights. idx = 0 -> HW for term with index 0 evaluated on input data.
        """
        if deg is None:
            deg = self.deg

        hws = [0] * self.num_terms(deg, False, exact=True)
        res = empty_bitarray(len(self.base[0]))

        if not FAST_IMPL_PH4:
            return self.eval_terms_raw_slow(deg, False, hws, res=res)

        ctr = 0
        for term in self.term_generator(deg):
            res.fast_copy(self.base[term[0]])      # Copy the first term to the result
            for i in range(1, deg-1):              # And the remaining terms
                res &= self.base[term[i]]
            hws[ctr] = res.fast_hw_and(self.base[term[deg-1]])
            ctr += 1
        assert ctr == len(hws)
        return hws

    def eval_all_terms(self, deg=None):
        """
        Evaluates all terms of deg [1, deg].

        Evaluation is done with a caching of the results from higher orders. e.g., when evaluating a+b+c,
        the c goes over all possible options, a+b result is cached until b is changed.
        The last order is evaluated in memory without actually storing an AND result anywhere.

        The lower orders are evaluated as a side product of caching - each new cache entry means a new combination
        of the lower order.

        Some lower orders evaluations are not included in the caching, e.g., for 128, 3 combination, the highest
        term in the ordering is [125, 126, 127] so with caching you cannot get [126, 127] evaluation.
        To fill missing gaps we have a term generator for each lower degree, it runs in parallel with the caching
        and if there are some missing terms we compute it manualy - raw AND, without cache.

        :warning: Works only with fast ph4r05 implementation.
        :param deg:
        :return: hw, hamming weight of the terms [deg=>[tidx=>hw]]
        """
        if deg is None:
            deg = self.deg

        hw = [0] * (deg+1)
        hw[0] = []
        for idx in range(1, deg+1):
            hw[idx] = [0] * self.num_terms(idx, False, exact=True)

        # center_hw = len(self.base[0]) * 2 ** (-1 * deg)
        # logger.info('Now the funny part! %s' % center_hw)
        # arr = bitarray.eval_all_terms(self.base, deg=deg, topk=128, hw_center=center_hw)
        # logger.info('Done')
        #logger.info('heap:   %s' % arr)
        #logger.info('sorted: %s' % sorted(arr, reverse=True))

        # deg1 is simple - just use HW on the basis
        hw[1] = [x.count() for x in self.base]
        if deg <= 1:
            return hw

        # deg2 is simple to compute without optimisations, if it is the top order we are interested in.
        if deg == 2:
            hw[2] = [0] * self.num_terms(2, False, exact=True)
            for idx, term in enumerate(self.term_generator(2)):
                hw[2][idx] = self.base[term[0]].fast_hw_and(self.base[term[1]])
            return hw

        # deg3 and more - optimisations in place.
        # temp buffer for computing missing evaluations
        res = self.new_buffer()

        # Sub evaluations of high orders.
        # Has deg-1 as it makes no sense to cache the last term - it is the result directly.
        # sub[0] = a      - deg1 result - basis reference
        # sub[1] = a+b    - deg2 result
        # sub[2] = a+b+c  - deg3 result
        sub = [self.new_buffer() for _ in range(0, deg-1)]

        # Lower degree indices update here.
        subdg = [0] * deg

        # Lower degree generators for filling up missing pieces
        subgen = [self.term_generator(x) for x in range(1, deg+1)]

        # Last term indices to detect changes in high orders
        # lst = [0,1,2,3,4,5] - for deg 6, triggers new caching if top 5 indices changes: [0,1,2,3,5,6]
        lst = [-1] * deg

        # Do the combination of the highest degree, cache when lower degree sub-combination changes.
        for idx, term in enumerate(self.term_generator(deg)):
            # Has high order cached element changed?
            # Make a diff term vs lst. If there is a diff, recompute cached sub-results.
            if term[:-1] != lst[:-1]:
                # Get the leftmost index in the term list where the change happened.
                # The 0 index is not considered as this change is not interesting - it is base[] anyway.
                # Thus domain of changed_from is 1 .. deg-2
                changed_from = deg-2
                for chidx in range(0, deg-1):
                    changed_from = chidx
                    if term[chidx] != lst[chidx]:
                        break

                # Recompute changed, from the more general to less. e.g., from a+b to a+b+c+d+e+f....
                for chidx in range(changed_from, deg-1):
                    if chidx == 0:
                        sub[chidx] = self.base[term[0]]
                    else:             # recursive definition - use the previous result.
                        sub[chidx].fast_copy(sub[chidx-1])
                        sub[chidx] &= self.base[term[chidx]]

                        # Run update generator up to this position to fill missing pieces
                        # Missing piece = [126,127] for deg = 3 cos max elem is [125,126,127]
                        for missing_piece in subgen[chidx]:
                            if missing_piece == term[0: 1 + chidx]:
                                break
                            res.fast_copy(self.base[missing_piece[0]])
                            for subi in range(1, 1+chidx):
                                res &= self.base[missing_piece[subi]]

                            hw[1 + chidx][subdg[chidx]] = res.count()
                            subdg[chidx] += 1
                            # logger.info('Fill in missing: %s, cur: %s' % (missing_piece, term[0: 1 + chidx]))

                        # Update lower degree HW
                        hw[1 + chidx][subdg[chidx]] = sub[chidx].count()
                        subdg[chidx] += 1

            # Evaluate the current expression using the cached results + fast hw.
            hw[deg][idx] = sub[deg-2].fast_hw_and(self.base[term[deg-1]])

            # copy the last term
            copy_poly(lst, term)

        # Finish generators - add missing combinations not reached by the caching from the higher ones.
        # E.g. for (128, 3) the higher combination is [125, 126, 127] so the maximal cached deg 2
        # is [125, 126]. This finishes the sequence for [126, 127].
        for finish_deg in range(2, deg):
            for missing_piece in subgen[finish_deg-1]:
                res.fast_copy(self.base[missing_piece[0]])
                for subi in range(1, finish_deg):
                    res &= self.base[missing_piece[subi]]

                hw[finish_deg][subdg[finish_deg-1]] = res.count()
                subdg[finish_deg-1] += 1
                # logger.info('Missing piece: %s' % missing_piece)

        return hw

    def eval_terms_raw_slow(self, deg, include_all_below, hws, res=None):
        """
        Subroutine for evaluating all terms, the slower one without our bitarray optimisations.
        Should be used only for testing.
        :param deg:
        :param include_all_below:
        :param hws: hamming weight accumulator
        :param res: working bitarray buffer
        :return: hws, hamming weight accumulator
        """
        if res is None:
            res = self.new_buffer()

        ctr = 0
        for cur_deg in range(1 if include_all_below else deg, deg + 1):
            for term in self.term_generator(deg):
                res.setall(True)
                for i in range(0, deg):
                    res &= self.base[term[i]]

                hw = self.hw(res)
                hws[ctr] = hw
                ctr += 1

        return hws

    def eval_poly_hw(self, poly, res=None, subres=None):
        """
        Evaluates polynomial Hamming weight on the input-precomputed base
        :param poly:
        :param res:
        :param subres:
        :return:
        """
        if FAST_IMPL_PH4:
            # return self.hw(self.eval_poly(poly, res=res, subres=subres))
            # return bitarray.eval_polynomial_hw(self.base, poly)
            return self.bbase.eval_poly_hw(poly)
        else:
            return self.hw(self.eval_poly(poly, res=res, subres=subres))

    def eval_polys_hw(self, polys, res=None, subres=None, hws=None):
        """
        Evaluates polynomial Hamming weight on the input-precomputed base
        :param poly:
        :param res:
        :param subres:
        :return:
        """
        if FAST_IMPL_PH4:
            return self.bbase.eval_poly_hw(None, polys, hws)
        else:
            res = [0] * len(polys) if not hws else hws
            for ix, poly in enumerate(polys):
                res[ix] = self.hw(self.eval_poly(poly, res=res, subres=subres))
            return res

    def eval_poly(self, poly, res=None, subres=None):
        """
        Evaluates a polynomial on the input precomputed data
        :param poly: polynomial specified as [term, term, term], e.g. [[1,2], [3,4], [5,6]] == x1x2 + x3x4 + x5x6
        :param res: buffer to use to store the result (optimization purposes)
        :param subres: working buffer for temporary computations (optimization purposes)
        :return: bitarray of results
        """
        ln = len(poly)
        if res is None:
            res = self.new_buffer()
        if subres is None:
            subres = self.new_buffer()

        self.eval_term(poly[0], res=res)
        for i in range(1, ln):
            res ^= self.eval_term(poly[i], res=subres)
        return res

    def expp_term_deg(self, deg):
        """
        Returns expected probability of result=1 of a term with given degree under null hypothesis of uniformity.
        :param deg:
        :return:
        """
        # return math.pow(2, -1 * deg)
        return self.pows[deg]  # if deg < self.blocklen else math.pow(2, -1 * deg)

    def expp_term(self, term):
        """
        Computes expected probability of result=1 of the given term under null hypothesis of uniformity.
        O(1) time, O(n lg n) w.r.t. term length (distinct bit positions).
        :param term: index list of the variables, e.g. [0,10] ~ x0x10
        :return:
        """
        dislen = len(set(term))
        # return math.pow(2, -1 * dislen)
        return self.pows[dislen]  # if dislen < self.blocklen else math.pow(2, -1 * dislen)

    def expp_xor_indep(self, p1, p2):
        """
        Probability of term t1 XOR t2 being 1 if t1 is 1 with p1 and t2 is 1 with p2.
        t1 and t2 has to be independent (no common sub-term).
        Due to associativity can be computed on multiple terms: t1 ^ t2 ^ t3 ^ t4 = (((t1 ^ t2) ^ t3) ^ t4) - zipping.

        XOR:
          a b | r
          ----+---
          1 1 | 0
          1 0 | 1  = p1    * (1-p2)
          0 1 | 1  = (1-p1)* p2
          0 0 | 0

        :param p1:
        :param p2:
        :return:
        """
        return p1*(1-p2)+(1-p1)*p2

    def term_remap(self, term):
        """
        Remaps the term to the lower indices - for simulation.
        remapping may lead to degree reduction, e.g., x1x2x3x3 -> x0x1x2
        :param term:
        :return:
        """
        return range(0, len(set(term)))

    def poly_remap(self, poly):
        """
        Remaps the polynomial to lower indices
        e.g., x7x8x9 + x110x112 -> x0x1x2 + x3x4
        :param poly:
        :return: new normalized polynomial, var idx -> new var idx map
        """
        # mapping phase
        idx = 0
        idx_map_rev = {}  # orig idx -> new idx

        res_poly = []
        for term in poly:
            res_term = []
            for bitpos in term:
                if bitpos not in idx_map_rev:
                    idx_map_rev[bitpos] = idx
                    res_term.append(idx)
                    idx += 1
                else:
                    res_term.append(idx_map_rev[bitpos])
            if len(res_term) > 0:
                res_poly.append(sorted(res_term))

        return res_poly, idx_map_rev

    def poly_fix_var(self, poly, neg, idx, val):
        """
        Reduces normed polynomial = fixes variable with bitpos=idx to val.

        Neg represents constant 1 as it has no representation in this polynomial form.
        Empty terms [] evaluates to 0 by definition. To be able to express term after reduction
        like p = 1 + 1 + 1 + x1x2 the neg is a constant part XORed to the result of the polynomial evaluation.
        In this case neg = 1 + 1 + 1 = 1. for x1=1 and x2=1 the p evaluates to neg + 1 = 1 + 1 = 0.
        For x1=1 and x2=0 the p evaluates to neg + 0 = 1 + 0 = 1.
        Neg is some kind of carry value.

        Method is used in the recursive polynomial evaluation with branch pruning.

        :param poly: polynomial representation
        :param neg: carry value for the XOR constant term
        :param idx: variable idx to fix
        :param val: value to fix variable to
        :return: (poly, neg)
        """
        res_poly = []
        for term in poly:
            # variable not in term - add to the polynomial unmodified
            if idx not in term:
                res_poly.append(term)
                continue

            # idx is in the term. remove
            if val == 0:
                # val is 0 -> whole term is zero, do not add.
                continue

            # val is 1 -> remove from term as it is constant now.
            n_term = [x for x in term if x != idx]

            if len(n_term) == 0:
                # term is empty -> is 1, xor with neg
                neg ^= 1

            else:
                # term is non-empty (at least one variable), add to poly
                res_poly.append(n_term)

        return res_poly, neg

    def expnum_poly_sim(self, poly):
        """
        Simulates the given polynomial w.r.t. null hypothesis, for all variable values combinations.
        O(2^n)
        :param poly:
        :return: number of polynomial evaluations to 1
        """
        npoly, idx_map_rev = self.poly_remap(poly)
        return self.expnum_poly_sim_norm_cached(npoly, len(idx_map_rev))

    def expp_poly_sim(self, poly):
        """
        Simulates the given polynomial w.r.t. null hypothesis, for all variable values combinations.
        O(2^n)
        :param poly:
        :return: probability the polynomial evals to 1 on random independent data
        """
        npoly, idx_map_rev = self.poly_remap(poly)
        return self.expnum_poly_sim_norm_cached(npoly, len(idx_map_rev)) / float(2**len(idx_map_rev))

    def expnum_poly_sim_norm_cached(self, poly, deg):
        """
        Computes how many times the given polynomial evaluates to 1 for all variable combinations.
        :param poly:
        :param deg:
        :return: number of polynomial evaluations to 1
        """
        if self.sim_norm_cache is None:
            return self.expnum_poly_sim_norm(poly, deg)

        # LRU cached sim variant
        key = get_poly_cache_key(poly)
        val = self.sim_norm_cache.get(key)
        if val is not None:
            return val

        val = self.expnum_poly_sim_norm(poly, deg)
        self.sim_norm_cache.put(key, val)
        return val

    def expnum_poly_sim_norm(self, poly, deg):
        """
        Computes how many times the given polynomial evaluates to 1 for all variable combinations.
        :param poly:
        :param deg:
        :return: number of polynomial evaluations to 1
        """
        # current evaluation is simple - iterate over all possible values of variables.
        num_one = 0

        gen = pos_generator(dim=deg, maxelem=1)
        for val in gen:
            # val is [x0,x1,x2,x3,x4,...] current variable value vector.
            # res = current polynomial evaluation
            res = 0

            # for each term &&-operation
            for term in poly:
                cval = 1
                for idx in term:
                    if val[idx] == 0:
                        cval = 0
                        break
                res ^= cval

            if res > 0:
                num_one += 1

        return num_one

    def expp_poly_dep(self, poly, neg=0):
        """
        Computes expected probability of result=1 of the given polynomial under null hypothesis of uniformity.
        It is assumed each term in the polynomial shares at least one variable with a different term
        in the polynomial so it cannot be easily optimised.
        :param poly:
        :param neg: internal, for recursion
        :return: probability of polynomial evaluating to 1 over all possibilities of variables
        """

        # at first - degenerate cases
        ln = len(poly)
        if ln == 1:
            # only one term - evaluate independently
            return self.expp_term(poly[0])

        # More than 1 term. Remap term for evaluation & degree detection.
        npoly, idx_map_rev = self.poly_remap(poly)
        deg = len(idx_map_rev)

        # if degree is small do the all combinations algorithm
        ones = self.expnum_poly_sim_norm_cached(npoly, deg)
        return float(ones) / float(2**deg)

        # for long degree or long polynomial we can do this:
        # a) Isolate independent variables, substitute them with a single one:
        #  x1x2x3x7x8 + x2x3x4x9x10x11 + x1x4x12x23 can be simplified to
        #  x1x2x3A    + x2x3x4B        + x1x4C - we don't need to iterate over independent variables
        #  here (e.g., x7x8x9x10,...). New variables A, B, C aggregate independent variables in the
        #  original equation. We evaluate A,B,C only one, A is 1 with prob 1/4, B 1/8, C 1/4. This is considered
        #  during the evaluation.
        #
        # b) recursive evaluation with branch pruning.
        # recursively do:
        #  - Is polynomial a const? Return.
        #  - Is polynomial of 1 term only? Return.
        #  - 1. fix the fist variable x1=0, use poly_fix_var, (some terms drop out), evaluate recursively.
        #  - 2. fix the fist variable x1=1, use poly_fix_var, evaluate recursively.
        #  - result = 0.5 * fix0 + 0.5 * fix1
        #  The pruning on the 0 branches can potentially save a lot of evaluations.

    def expp_poly(self, poly):
        """
        Computes expected probability of result=1 of the given polynomial under null hypothesis of uniformity.
        Due to non-independence between terms this evaluation can take some time - simulating.
        Independent terms are simulated, i.e. all variable combinations are computed.
        :param poly: list of terms in index notation
        :return: probability of polynomial evaluating to 1 over all possibilities of variables
        """
        # Optimization: find independent terms, move aside, can be evaluated without simulation.
        # independent terms are XORed with expp_xor_indep()
        #
        # Non-independent terms need to be evaluated together. Naive method is trying all combinations of involved
        # variables and compute number of 1s.
        #
        # Optimization can be done if this is performed recursively: let x1=0, (and x1=1 (two rec. branches))
        # then reduce polynomial and apply recursively. Often a lot of branches can be pruned from the computation as
        # it leads to 0 due to ANDs. For non-zero branches, number of 1 evaluations need to be computed.
        # same principle can be used for this.
        #
        # Find independent terms.
        ln = len(poly)

        # degenerate case = 1 term
        if ln == 1:
            return self.expp_term(poly[0])
        terms = [set(x) for x in poly]

        # Degenerate - disjoint sets
        term_lens = [len(x) for x in terms]
        all_terms_disjoin = len(union(terms)) == sum(term_lens)
        if all_terms_disjoin:
            res = self.expp_term_deg(term_lens[0])
            for term_len in term_lens[1:]:
                res = expp_xor_indep(res, self.expp_term_deg(term_len))
            return res
            # probs = [self.expp_term_deg(term_len) for term_len in term_lens]
            # return reduce(lambda x, y: self.expp_xor_indep(x, y), probs)

        if ln == 2:
            return self.expp_poly_dep(poly)

        # General case:
        #   Finding independent terms = create a graph from the terms in the polynomial.
        #   There is a connection if t1 and t2 share at least one element.
        #   Find connected components of the graph. Union-find (disjoint sets) data structure is helping with it.
        uf = ufh.UnionFind()
        for idx, term in enumerate(terms):
            uf.make_set(idx)
        for idx, term in enumerate(terms):
            for idx2 in range(idx+1, ln):
                if len(term & terms[idx2]) > 0:
                    uf.union(idx, idx2)
                pass
            pass
        pass

        # clusterize terms related to each other.
        # [[t1,t2,t3], [t4], [t5]]
        clusters = [[poly[y] for y in x] for x in uf.get_set_map().values()]

        # Each cluster can be evaluated independently and XORed with the rest.
        res = self.expp_poly_dep(clusters[0])
        for x in clusters[1:]:
            res = expp_xor_indep(res, self.expp_poly_dep(x))
        return res

        # probs = [self.expp_poly_dep(x) for x in clusters]
        #
        # # reduce the prob list with independent term-xor formula..
        # res = reduce(lambda x, y: self.expp_xor_indep(x, y), probs)
        # return res


class Tester(object):
    """
    Polynomial tester
    """

    def __init__(self, reffile=None, *args, **kwargs):
        self.reffile = reffile

    def load_file(self, file):
        """
        Loads file to the
        :param file:
        :return:
        """
        pass

    def work(self):
        pass


def chunks(iterable, size):
    from itertools import chain, islice
    iterator = iter(iterable)
    for first in iterator:
        yield list(chain([first], islice(iterator, size - 1)))


def sidak_alpha(alpha, m):
    """
    Compute new significance level alpha for M independent tests.
    More tests -> unexpected events occur more often thus alpha has to be adjusted.
    Overall test battery fails if min(pvals) < new_alpha.
    """
    return 1 - (1 - alpha)**(1./m)


def sidak_inv(alpha, m):
    """
    Inverse transformation of sidak_alpha function.
    Used to compute final p-value of M independent tests if while preserving the
    same significance level for the resulting p-value.
    """
    return 1 - (1 - alpha)**m


def merge_pvals(pvals, batch=2):
    """
    Merging pvals with Sidak.

    Note that the merging tree has to be symmetric, otherwise the computation on pvalues is not correct.
    Note: 1-(1-(1-(1-x)^3))^2 == 1-((1-x)^3)^2 == 1-(1-x)^6.
    Example: 12 nodes, binary tree: [12] -> [2,2,2,2,2,2] -> [2,2,2]. So far it is symmetric.
    The next layer of merge is problematic as we merge [2,2] and [2] to two p-values.
    If a minimum is from [2,2] (L) it is a different expression as from [2] R as the lists
    have different lengths. P-value from [2] would increase in significance level compared to Ls on this new layer
    and this it has to be corrected.
    On the other hand, the L minimum has to be corrected as well as it came from
    list of the length 3. We want to obtain 2 p-values which can be merged as if they were equal (exponent 2).
    Thus the exponent on the [2,2] and [2] layer will be 3/2 as we had 3 p-values in total and we are producing 2.
    """
    if len(pvals) <= 1:
        return pvals

    batch = min(max(2, batch), len(pvals))  # norm batch size
    parts = list(chunks(pvals, batch))
    exponent = len(pvals) / len(parts)
    npvals = []
    for p in parts:
        pi = sidak_inv(min(p), exponent)
        npvals.append(pi)
    return merge_pvals(npvals, batch)


def booltest_pval(nfails=1, ntests=36, alpha=1/20000):
    acc = [scipy_comb(ntests, k) * (1 - alpha) ** (ntests - k) * alpha ** k for k in range(nfails)]
    return max(0, 1 - sum(acc))


def booltest_pval_log(nfails=1, ntests=36, alpha=1/20000):
    log = math.log2
    acc = [2 ** (log(scipy_comb(ntests, k)) + (ntests - k) * log((1 - alpha)) + k * log(alpha)) for k in range(nfails)]
    return max(0, 1 - sum(acc))


def bar_chart(sources=None, values=None, res=None, error=None, xlabel=None, title=None):
    import numpy as np
    import matplotlib.pyplot as plt
    if res is not None:
        sources = [x[0] for x in res]
        values = [x[1] for x in res]

    plt.rcdefaults()
    y_pos = np.arange(len(sources))
    plt.barh(y_pos, values, align='center', xerr=error, alpha=0.4)
    plt.yticks(y_pos, sources)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()
