import numpy as np
from bitstring import Bits, BitArray, BitStream, ConstBitStream
import bitarray
import types
import math
import logging
import crypto_util
import ufx.uf_hash as ufh


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


def to_bitarray(inp=None, filename=None, const=True):
    """
    Converts input to bitarray for computation with TermEval
    :return:
    """
    constructor = Bits if const else BitArray
    if isinstance(inp, types.StringTypes):
        return constructor(bytes=inp)
    else:
        raise ValueError('Unknown input')



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

        term = BitArray(uint=0, length=blocklen)
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
        ctr = 0
        res_size = int(math.ceil(len(block)/float(len(term))))
        res = BitArray(uint=0, length=res_size)
        for idx in range(0, ln, lnt):
            res[ctr] = ((block[idx:idx + self.blocklen] & term) == term)
            ctr += 1
        return res

    def hw(self, block):
        """
        Computes hamming weight of the block
        :param block: bit representation of the input
        :return:
        """
        return block.count(True)

    def term_generator(self, deg=None):
        """
        Returns term generator for given deg (internal if none is given) and blocklen
        :return:
        """
        if deg is None:
            deg = self.deg

        return term_generator(deg, self.blocklen-1)

    def load(self, block):
        """
        Precomputes data
        :param block:
        :return:
        """
        self.gen_base(block)

    def gen_base(self, block):
        """
        Generate base for term evaluation from the block.
        Evaluates each base term (deg=1) on the input, creates a base for further evaluation of terms of high orders.
        :param block: bit representation of the input
        :return:
        """
        if (len(block)/8 % self.blocklen) != 0:
            raise ValueError('Input data not multiple of block length')

        self.cur_tv_size = len(block)/8
        self.cur_evals = self.cur_tv_size / self.blocklen

        ln = len(block)
        res_size = int(math.ceil(len(block) / float(self.blocklen)))

        self.base = [None] * self.blocklen
        for bitpos in range(0, self.blocklen):
            logger.info('bitpos %d' % bitpos)
            ctr = 0
            res = BitArray(uint=0, length=res_size)
            for idx in range(0, ln, self.blocklen):
                res[ctr] = block[idx+bitpos] == 1
                ctr += 1
            self.base[bitpos] = Bits(res)
            # old, slower method
            # term = self.gen_term([bitpos])
            # self.base[bitpos] = Bits(self.eval_term_raw(term, block))

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

    def eval_terms(self, deg=None):
        """
        Evaluates all terms on the input data precomputed in the base.
        Returns array of hamming weights.
        :param deg: degree of the terms to generate. If none, default degree is taken.
        :return: array of hamming weights. idx = 0 -> HW for term with index 0 evaluated on input data.
        """
        if deg is None:
            deg = self.deg

        hws = []
        for idx, term in enumerate(self.term_generator(deg)):
            res = self.eval_term(term)
            hw = self.hw(res)
            hws.append(hw)

        return hws

    def eval_poly(self, poly):
        """
        Evaluates a polynomial on the input precomputed data
        :param poly: polynomial specified as [term, term, term], e.g. [[1,2], [3,4], [5,6]] == x1x2 + x3x4 + x5x6
        :return:
        """
        ln = len(poly)
        res = self.eval_term(poly[0])
        for i in range(1, ln):
            res ^= self.eval_term(poly[i])
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
        :return:
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
        return self.expnum_poly_sim_norm(poly, idx_map_rev)

    def expnum_poly_sim_norm(self, poly, idx_map_rev):
        """
        Computes how many times the given polynomial evaluates to 1 for all variable combinations.
        :param poly:
        :param idx_set:
        :return: number of polynomial evaluations to 1
        """
        # current evaluation is simple - iterate over all possible values of variables.
        deg = len(idx_map_rev)
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
            # only one term - evluate independently
            return self.expp_term(poly[0])

        # More than 1 term. Remap term for evaluation & degree detection.
        npoly, idx_map_rev = self.poly_remap(poly)
        deg = len(idx_map_rev)

        # if degree is small do the all combinations algorithm
        ones = self.expnum_poly_sim_norm(npoly, idx_map_rev)
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

