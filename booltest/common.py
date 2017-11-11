#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from past.builtins import basestring
from past.builtins import xrange

import hashlib
import logging
import math
import os
import random
import signal
import subprocess
import sys
from functools import reduce

import scipy.misc
import ufx.uf_hash as ufh

import bitarray
from bitstring import Bits, BitArray, BitStream, ConstBitStream
from repoze.lru import lru_cache, LRUCache

from .crypto_util import aes_ctr, get_zero_vector

# Enables bitarray - with native C extension
FAST_IMPL = True

# Enables bitarray - with native C extension with eval_monic()
FAST_IMPL_PH4 = True


logger = logging.getLogger(__name__)


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
        yield list(idx)

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


def term_generator(deg, maxelem, prob_choose=1.0):
    """
    Generates all terms of the given degree with given max len.

    e.g. for deg = 3, ln = 9:
    [0,1,2] .. [7,8,9]
    :param deg:
    :param maxelem:
    :param prob_choose: probability the given element will be chosen
    :return:
    """
    idx = [0] * deg
    for i in range(deg):
        idx[i] = i
        if i > maxelem:
            raise ValueError('deg too big for the maxelem')

    while True:
        if prob_choose >= 1.0:
            yield list(idx)
        elif random.random() < prob_choose:
            yield list(idx)

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
    return scipy.misc.comb(n, k, exact=exact)


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


@lru_cache(maxsize=1024)
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


@lru_cache(maxsize=1024)
def unrank(i, n, k):
    """
    returns the i-th combination of k numbers chosen from 0,2,...,n-1, indexing from 0
    """
    c = []
    r = i+0
    j = 0
    for s in range(1, k+1):
        cs = j+1
        while r-comb_cached(n-cs, k-s) >= 0:
            r -= comb_cached(n-cs, k-s)
            cs += 1
        c.append(cs-1)
        j = cs
    return c


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


class InputObject(object):
    """
    Input stream object.
    Can be a file, stream, or something else
    """
    def __init__(self, *args, **kwargs):
        self.sha1 = hashlib.sha1()
        self.data_read = 0

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return 'InputObject()'

    def check(self):
        """
        Checks if stream is readable
        :return:
        """

    def size(self):
        """
        Returns the size of the data
        :return:
        """
        return -1

    def read(self, size):
        """
        Reads size of data
        :param size:
        :return:
        """
        raise NotImplementedError('Not implemented - base class')


class FileInputObject(InputObject):
    """
    File input object - reading from the file
    """
    def __init__(self, fname, *args, **kwargs):
        super(FileInputObject, self).__init__(*args, **kwargs)
        self.fname = fname
        self.fh = None

    def __enter__(self):
        super(FileInputObject, self).__enter__()
        self.fh = open(self.fname, 'r')

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(FileInputObject, self).__exit__(exc_type, exc_val, exc_tb)
        try:
            self.fh.close()
        except:
            logger.error('Error when closing file %s descriptor' % self.fname)

    def __repr__(self):
        return 'FileInputObject(file=%r)' % self.fname

    def __str__(self):
        return self.fname

    def check(self):
        if not os.path.exists(self.fname):
            raise ValueError('File %s was not found' % self.fname)

    def size(self):
        return os.path.getsize(self.fname)

    def read(self, size):
        data = self.fh.read(size)
        self.sha1.update(data)
        self.data_read += len(data)
        return data


class StdinInputObject(InputObject):
    """
    Reads data from the stdin
    """
    def __init__(self, desc=None, *args, **kwargs):
        super(StdinInputObject, self).__init__(*args, **kwargs)
        self.desc = desc

    def __enter__(self):
        super(StdinInputObject, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(StdinInputObject, self).__exit__(exc_type, exc_val, exc_tb)
        sys.stdin.close()

    def __repr__(self):
        return 'StdinInputObject()'

    def __str__(self):
        if self.desc is not None:
            return 'stdin-%s' % self.desc
        return 'stdin'

    def size(self):
        return -1

    def read(self, size):
        data = sys.stdin.read(size)
        self.sha1.update(data)
        self.data_read += len(data)
        return data


class FileLikeInputObject(InputObject):
    """
    Reads data from file like objects - e.g., stdout, sockets, ...
    """
    def __init__(self, fh=None, desc=None, *args, **kwargs):
        super(FileLikeInputObject, self).__init__(*args, **kwargs)
        self.fh = fh
        self.desc = desc

    def __repr__(self):
        return 'FileLikeInputObject()'

    def __str__(self):
        if self.desc is not None:
            return '%s' % self.desc
        return 'file-handle'

    def size(self):
        return -1

    def read(self, size):
        data = self.fh.read(size)
        self.sha1.update(data)
        self.data_read += len(data)
        return data


class CommandStdoutInputObject(InputObject):
    """
    Executes command, reads from its stdout - used with generators.
    """
    def __init__(self, cmd=None, seed=None, desc=None, *args, **kwargs):
        super(CommandStdoutInputObject, self).__init__(*args, **kwargs)
        self.cmd = cmd
        self.seed = seed
        self.desc = desc
        self.proc = None
        self.subio = None

    def __repr__(self):
        return 'CommandStdoutInputObject()'

    def __str__(self):
        if self.desc is not None:
            return '%s' % self.desc
        return 'cmd: %s' % self.cmd

    def __enter__(self):
        super(CommandStdoutInputObject, self).__enter__()

        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=1024, close_fds=True, shell=True, preexec_fn=os.setsid)
        self.subio = FileLikeInputObject(fh=self.proc.stdout, desc=self.cmd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.proc.stdout.close()
            self.proc.terminate()
            self.proc.kill()
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except Exception as e:
            logger.debug('Exception killing process: %s' % e)

        super(CommandStdoutInputObject, self).__exit__(exc_type, exc_val, exc_tb)

    def size(self):
        return -1

    def read(self, size):
        data = self.subio.read(size)
        self.sha1.update(data)
        self.data_read += len(data)
        return data


class AESInputObject(InputObject):
    """
    AES data input generation.
    """
    def __init__(self, seed=None, desc=None, *args, **kwargs):
        super(AESInputObject, self).__init__(*args, **kwargs)
        self.seed = seed
        self.desc = desc

    def __repr__(self):
        return 'AESInputObject(seed=%r)' % self.seed

    def __str__(self):
        if self.desc is not None:
            return '%s' % self.desc
        return 'aes-ctr(sha256(0x%x))' % self.seed

    def size(self):
        return -1

    def read(self, size):
        aes = aes_ctr(hashlib.sha256('%x' % self.seed).digest())
        data = aes.encrypt(get_zero_vector(size))
        self.sha1.update(data)
        self.data_read += len(data)
        return data


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
        self.sim_norm_cache = LRUCache(64)

    def base_size(self):
        """
        Returns base size of the vector - same as size of the base
        :return:
        """
        return len(self.base[0])

    def new_buffer(self):
        """
        Returns the allocated bitarray - non-intiialized of the size of the base.
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

    def eval_term_raw_single(self, term, block):
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

    def eval_term_raw(self, term, block):
        """
        Evaluates term on the bitarray input. Uses & on the whole term and the block slices.
        Block has to be a multiple of the term size. term is evaluated by sliding window of size=term.
        In result each bit represents a single term evaluation on the given sub-block
        :param term: bit representation of the term
        :param block: bit representation of the input, bit array of term evaluations. size = size of block / size of term.
        :return:
        """
        ln = len(block)
        lnt = len(term)
        ctr = 0
        res_size = ln // lnt
        res = empty_bitarray(res_size)
        for idx in range(0, ln, lnt):
            res[ctr] = ((block[idx:idx + self.blocklen] & term) == term)
            ctr += 1
        return res

    def eval_poly_raw_single(self, poly, block):
        """
        Evaluates polynomial on the raw input - bit array. Uses [] operator to access bits in block.
        Returns a single number, evaluates polynomial on single block
        :param poly: polynomial in the index notation
        :param block: bitarray indexable by []
        :return:
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

    def hw(self, block):
        """
        Computes hamming weight of the block
        :param block: bit representation of the input
        :return:
        """
        if FAST_IMPL:
            return block.count()
        else:
            return block.count(True)

    def term_generator(self, deg=None):
        """
        Returns term generator for given deg (internal if none is given) and blocklen
        :return:
        """
        if deg is None:
            deg = self.deg

        return term_generator(deg, self.blocklen-1)

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
            raise ValueError('Input data not multiple of block length')

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
        and if there are some missing terms we compute it mannualy - raw AND, without cache.

        :warning: Works only with fast ph4r05 implementation.
        :param deg:
        :return:
        """
        if deg is None:
            deg = self.deg

        hw = [None] * (deg+1)
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
            lst = term

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
        :return:
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

    def eval_poly(self, poly, res=None, subres=None):
        """
        Evaluates a polynomial on the input precomputed data
        :param poly: polynomial specified as [term, term, term], e.g. [[1,2], [3,4], [5,6]] == x1x2 + x3x4 + x5x6
        :param res: buffer to use to store the result (optimization purposes)
        :param subres: working buffer for temporary computations (optimization purposes)
        :return:
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
        return math.pow(2, -1 * deg)

    def expp_term(self, term):
        """
        Computes expected probability of result=1 of the given term under null hypothesis of uniformity.
        O(1) time, O(n lg n) w.r.t. term length (distinct bit positions).
        :param term:
        :return:
        """
        dislen = len(set(term))
        return math.pow(2, -1*dislen)

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
        return self.expnum_poly_sim_norm_cached(poly, len(idx_map_rev))

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
        key = ','.join(['-'.join([str(y) for y in x]) for x in poly])
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
        :param poly:
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

        # degenerate case = 2 terms
        if ln == 2:
            if len(terms[0] & terms[1]) == 0:
                return self.expp_xor_indep(self.expp_term(poly[0]), self.expp_term(poly[1]))
            else:
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
        probs = [self.expp_poly_dep(x) for x in clusters]

        # reduce the prob list with independent term-xor formula..
        res = reduce(lambda x, y: self.expp_xor_indep(x, y), probs)
        return res


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

