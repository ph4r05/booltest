import numpy as np
from bitstring import Bits, BitArray, BitStream, ConstBitStream
import bitarray
import types


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

    def eval_term_raw(self, term, block):
        """
        Evaluates a term on the data block.
        block has to be a multiple of the term size. term is evaluated by sliding window of size=term.
        :param term: bit representation of the term
        :param block: bit representation of the input
        :return: bit representation of the result
        """
        ln = len(block)
        lnt = len(term)
        res = BitArray()
        for idx in range(0, ln, lnt):
            res.append(block[idx:idx + self.blocklen] & term)
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
        Evaluates each base term (deg=1) on the input, creates a base for futher evaluation of terms of high orders.
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
        :return: bit representation of the result
        """
        ln = len(term)
        res = BitArray(self.base[term[0]])

        for i in range(1, ln):
            res &= self.base[term[i]]
        return res





class Const_bits(Bits):
    pass


class Bits(BitArray):
    pass


