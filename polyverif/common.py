import numpy as np
from bitstring import Bits, BitArray, BitStream, ConstBitStream
import bitarray
import types
import math


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


def term_generator(deg, maxelem):
    """
    Generates all terms of the given degree with given max len.

    e.g. for deg = 3, ln = 9:
    [0,1,2] .. [7,8,9]
    :param deg:
    :param maxelem:
    :return:
    """
    idx = [0] * deg
    for i in range(deg):
        idx[i] = i
        if i > maxelem:
            raise ValueError('deg too big for the maxelem')

    while True:
        yield idx

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


def zscore_p(observed, expected, N):
    """
    Computes z-score for the normal distribution
    :param observed: prob.
    :param expected: prob.
    :param N:
    :return:
    """
    return (observed-expected) / math.sqrt((expected*(1.0-expected))/float(N))


class PolyCheck(object):
    def __init__(self, *args, **kwargs):
        # block length in bits, term size.
        self.blocklen = 128

        # term degree
        self.deg = 1

        # evaluated base terms of deg=1
        self.base = []

    def gen_term(self, indices):
        """
        Generates a bit term mask from the indices created by term_generator().
        blocklen wide.
        :param indices: array of bit indices, e.g., [0,8] -> x0x8
        :return: bit representation of the term
        """
        term = BitArray(uint=0, length=self.blocklen)
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
        res = BitArray()
        for idx in range(0, ln, lnt):
            res.append(block[idx:idx + self.blocklen] & term)
        return res

    def eval_term_raw(self, term, block):
        """
        Evaluates term on the input.
        Block has to be a multiple of the term size. term is evaluated by sliding window of size=term.
        In result each bit represents a single term evaluation on the given sub-block
        :param term: bit representation of the term
        :param block: bit representation of the input, bit array of term evaluations. size = size of block / size of term.
        :return:
        """
        ln = len(block)
        lnt = len(term)
        res = BitArray()
        for idx in range(0, ln, lnt):
            res.append((block[idx:idx + self.blocklen] & term) == term)
        return res

    def hw(self, block):
        """
        Computes hamming weight of the block
        :param block: bit representation of the input
        :return:
        """
        return block.count(True)

    def gen_base(self, block):
        """
        Generate base for polynomial evaluation from the block.
        Evaluates each base term (deg=1) on the input, creates a base for further evaluation of terms of high orders.
        :param block: bit representation of the input
        :return:
        """
        self.base = [None] * self.blocklen
        for bitpos in range(0, self.blocklen):
            term = self.gen_term([bitpos])
            self.base[bitpos] = Bits(self.eval_term_raw(term, block))

    def eval_term(self, term):
        """
        Evaluates term on the block using the precomputed base.
        :param term: term represented as an array of bit positions
        :return: bit representation of the result, each bit represents single term evaluation on the given sub-block
        """
        ln = len(term)
        res = BitArray(self.base[term[0]])

        for i in range(1, ln):
            res &= self.base[term[i]]
        return res

    def eval_terms(self):
        """
        Evaluates all terms on the input data precomputed in the base.
        Returns array of hamming weights.
        :return: array of hamming weights. idx = 0 -> HW for term with index 0 evaluated on input data.
        """
        hws = []
        for idx, term in enumerate(term_generator(self.deg, self.blocklen-1)):
            res = self.eval_term(term)
            hw = self.hw(res)
            hws.append(hw)

        return hws





class Const_bits(Bits):
    pass


class Bits(BitArray):
    pass


