#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generating configuration files for CryptoStreams
Not up to date with the newest versions, supports only simple generation strategies.
RPCS is not supported as it has been changed to tuple stream.

https://github.com/crocs-muni/CryptoStreams
"""

import collections
import math
import re
import random
import argparse

from . import common, misc

# Regex for parsing arg names from source codes
# if \(name == "(.+?)".+


FUNCTION_ESTREAM = 1
FUNCTION_SHA3 = 2
FUNCTION_BLOCK = 3
FUNCTION_HASH = 4


STREAM_TYPES = {
    FUNCTION_ESTREAM: 'estream',
    FUNCTION_SHA3: 'sha3',
    FUNCTION_BLOCK: 'block',
    FUNCTION_HASH: 'hash',
}


class StreamCodes:
    ZERO = 'false_stream'
    RANDOM = 'pcg32_stream'
    SAC = 'sac'
    SAC_STEP = 'sac_step'
    COUNTER = 'counter'
    XOR = 'xor_stream'
    RPCS = 'rnd_plt_ctx_stream'


class FunctionParams(object):
    __slots__ = ['block_size', 'key_size', 'iv_size', 'rounds', 'min_rounds', 'in_size', 'out_size', 'fname']

    def __init__(self, block_size=None, key_size=None, rounds=None, min_rounds=None, iv_size=None, in_size=None,
                 out_size=None, fname=None):
        self.block_size = block_size
        self.key_size = key_size
        self.iv_size = iv_size
        self.rounds = rounds
        self.min_rounds = min_rounds
        self.in_size = in_size
        self.out_size = out_size
        self.fname = fname

    def to_json(self):
        return dict(misc.slot_obj_dict(self))

    def __repr__(self):
        return 'FunctionParams(%r, block_size=%r, key_size=%r, iv_size=%r, rounds=%r, ' \
               'min_rounds=%r, in_size=%r, out_size=%r)' \
               % (self.fname, self.block_size, self.key_size, self.iv_size, self.rounds, self.min_rounds, self.in_size,
                  self.out_size)


class Stream(object):
    """
    Base stream object
    """
    def __init__(self):
        pass

    def is_randomized(self):
        return False

    def scode(self):
        return None


# eStream
ESTREAM = {
    'ABC': None,
    'Achterbahn': None,
    'Chacha': FunctionParams(rounds=20, block_size=32, iv_size=8, key_size=16, in_size=64),
    'Chacha_k32': FunctionParams(rounds=20, block_size=32, iv_size=8, key_size=32, in_size=64, fname='Chacha'),
    'CryptMT': None,
    'DECIM': FunctionParams(rounds=8, iv_size=4),
    'DICING': None,
    'Dragon': FunctionParams(rounds=16),
    'Edon80': None,
    'F-FCSR': FunctionParams(rounds=5),
    'Fubuki': FunctionParams(rounds=4),
    'Grain': FunctionParams(rounds=13),
    'HC-128': None,
    'Hermes': FunctionParams(rounds=10),
    'LEX': FunctionParams(rounds=10),
    'MAG': None,
    'MICKEY': None,
    'Mir-1': None,
    'Pomaranch': None,
    'Py': None,
    'Rabbit': FunctionParams(rounds=4),
    'Salsa20': FunctionParams(rounds=20),
    'SFINKS': None,
    'SOSEMANUK': FunctionParams(rounds=25),
    'Trivium': FunctionParams(rounds=9, iv_size=[4, 8, 10], key_size=10),
    'TSC-4': FunctionParams(rounds=32),
    'Yamb': None,
    'WG': None,
    'Zk-Crypt': None
}


# SHA3
SHA3 = {
    'Abacus': FunctionParams(rounds=135),
    'ARIRANG': FunctionParams(rounds=4),
    'AURORA': FunctionParams(rounds=17),
    'BLAKE': FunctionParams(rounds=16),
    'Blender': FunctionParams(rounds=32),
    'BMW': FunctionParams(rounds=16),
    'Boole': FunctionParams(rounds=16),
    'Cheetah': FunctionParams(rounds=16),
    'CHI': FunctionParams(rounds=20),
    'CRUNCH': FunctionParams(rounds=224),
    'CubeHash': FunctionParams(rounds=8),
    'DCH': FunctionParams(rounds=4),
    'DynamicSHA': FunctionParams(rounds=16),
    'DynamicSHA2': FunctionParams(rounds=17),
    'ECHO': FunctionParams(rounds=18),
    'EDON': None,
    'ESSENCE': FunctionParams(rounds=32, min_rounds=8),
    'Fugue': FunctionParams(rounds=8),
    'Grostl': FunctionParams(rounds=10),
    'Hamsi': FunctionParams(rounds=3),
    'JH': FunctionParams(rounds=42),
    'Keccak': FunctionParams(rounds=24),
    'Khichidi': None,
    'LANE': FunctionParams(rounds=12),
    'Lesamnta': FunctionParams(rounds=32),
    'Luffa': FunctionParams(rounds=8),
    'MCSSHA3': None,
    'MD6': FunctionParams(rounds=104),
    'MeshHash': FunctionParams(rounds=256),
    'NaSHA': None,
    'Sarmal': FunctionParams(rounds=16),
    'Shabal': None,
    'SHAMATA': None,
    'SHAvite3': FunctionParams(rounds=12),
    'SIMD': FunctionParams(rounds=4),
    'Skein': FunctionParams(rounds=72),
    'SpectralHash': None,
    'StreamHash': None,
    'Tangle': FunctionParams(rounds=80),
    'Twister': FunctionParams(rounds=9),
    'WaMM': FunctionParams(rounds=2),
    'Waterfall': FunctionParams(rounds=16),
    'Tangle2': FunctionParams(rounds=80),
}


# Hash
HASH = {
    'Gost': FunctionParams(rounds=32, block_size=32, out_size=32),
    'MD5': FunctionParams(rounds=64, block_size=16, out_size=16),
    'SHA1': FunctionParams(rounds=80, block_size=20, in_size=64, out_size=20),
    'SHA256': FunctionParams(rounds=64, block_size=32, in_size=64, out_size=32),
    'RIPEMD160': FunctionParams(rounds=80, block_size=20, in_size=64, out_size=20),
    'Tiger': FunctionParams(rounds=24, block_size=24, in_size=64, out_size=24),
    'Whirlpool': FunctionParams(rounds=10, block_size=64, in_size=64, out_size=64),
}


# Block ciphers
BLOCK = {
    'ARIA': FunctionParams(16, 16, rounds=12),
    'BLOWFISH': FunctionParams(8, 16, rounds=16),
    'TEA': FunctionParams(8, 16, rounds=64),
    'AES': FunctionParams(16, 16, rounds=10),
    'CAST': FunctionParams(8, 16, rounds=16),
    'CAMELLIA': FunctionParams(16, 16, rounds=18),
    'IDEA': FunctionParams(8, 16, rounds=8),
    'MARS': FunctionParams(16, 16, rounds=16),
    'RC4': FunctionParams(16, 16),
    'RC6': FunctionParams(16, 16, rounds=20),
    'SEED': FunctionParams(16, 16, rounds=16),
    'SERPENT': FunctionParams(16, 16, rounds=32),
    'SIMON': FunctionParams(16, 16, rounds=32),
    'SPECK': FunctionParams(16, 16, rounds=22),
    'SINGLE-DES': FunctionParams(8, 7, rounds=16),
    'TRIPLE-DES': FunctionParams(8, 24, rounds=16),
    'TWOFISH': FunctionParams(16, 16, rounds=16),
    'GOST_BLOCK': FunctionParams(rounds=32, block_size=8, key_size=32, fname='GOST'),
}


# Interesting rounds to test
ROUNDS = {
    'ARIA': [1, 2, 3, 4],
    'AES': [1, 2, 3, 4, 10],
    'ARIRANG': [3, 4],
    'AURORA': [2, 3],
    'BLAKE': [1, 2],
    'BLOWFISH': [1, 2, 3, 4, 5],
    'CAMELLIA': [1, 2, 3, 4, 5],
    'CAST': [1, 2, 3, 4, 5],
    'Chacha': [1, 2, 3, 4],
    'Cheetah': [4, 5],
    'CRUNCH': [7, 8],
    'CubeHash': [0, 1],
    'DCH': [1, 2],
    'DECIM': [5, 6],
    'DynamicSHA': [7, 8],
    'DynamicSHA2': [11, 12],
    'Dragon': [4, 5],
    'ECHO': [1, 2],
    'F-FCSR': [1, 2, 3],
    'Fubuki': [2, 3],
    'Grain': [2, 3],
    'Gost': [1, 2, 3, 4, 5],
    'GOST_BLOCK': [3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Grostl': [2, 3],
    'Hamsi': [0, 1],
    'Hermes': [1, 2],
    'IDEA': [1, 2, 3, 4, 8],
    'JH': [2, 3, 4, 5, 6, 7],
    'Keccak': [3, 4],
    'LEX': [3, 4],
    'Lesamnta': [3, 4],
    'Luffa': [7, 8],
    'MARS': [0, 1, 2, 3, 4],
    'MD5': [3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16],
    'MD6': [5, 6, 7, 8, 9, 10],
    'Rabbit': [1, 2, 3],
    'RIPEMD160': [6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 24, 32],
    'RC4': [1],
    'RC6': [1, 2, 3, 4, 5, 6],
    'SIMD': [0, 1],
    'Salsa20': [1, 2, 3, 4],
    'SEED': [1, 2, 3, 4],
    'SHA1': [4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18],
    'SHA256': [3, 4, 5, 6, 7, 11, 12, 13, 14],
    'SERPENT': [1, 2, 3, 4, 8],
    'SIMON': [16, 17, 18, 19, 20],
    'SINGLE-DES': [1, 2, 3, 4, 5, 6],
    'Skein': [4, 5, 6],
    'SOSEMANUK': [1, 2, 3, 4],
    'SPECK': [8, 9, 10, 11],
    'TEA': [1, 2, 3, 4, 5, 6],
    'TRIPLE-DES': [1, 2, 3, 4],
    'TSC-4': [12, 13, 14],
    'Tangle': [22, 23, 24],
    'Tangle2': [22, 23, 24],
    'Tiger': [1, 2, 3],
    'Twister': [6, 7],
    'Trivium': [1, 2, 3, 4],
    'TWOFISH': [1, 2, 3, 4, 5],
    'Whirlpool': [1, 2, 3],
}


ALL_FUNCTIONS = common.merge_dicts([SHA3, ESTREAM, HASH, BLOCK])


NARROW_SELECTION = {
    'AES', 'BLOWFISH', 'SINGLE-DES', 'TRIPLE-DES', 'SIMON', 'SPECK', 'TEA',
    'BLAKE', 'Grostl', 'Grain', 'JH', 'Keccak', 'MD6', 'Skein',
    'GOST_BLOCK', 'Salsa20', 'Chacha', 'MARS', 'RC6', 'SERPENT', 'TWOFISH',
    'Gost', 'MD5', 'SHA1', 'SHA256', 'RIPEMD160', 'Tiger', 'Whirlpool',
    'DECIM', 'SOSEMANUK', 'Trivium',
    'ARIA', 'CAST', 'CAMELLIA', 'IDEA', 'SEED',
}


NARROW_SELECTION_EXTPAPER = {
    'AES', 'BLOWFISH', 'SINGLE-DES', 'TRIPLE-DES', 'TEA',
    'Grostl', 'JH', 'Keccak', 'MD6', 'MD5', 'SHA1', 'SHA256',
}


BENCHMARK_SELECTION = {
    'AES', 'BLOWFISH', 'TWOFISH', 'SINGLE-DES', 'TRIPLE-DES'
}


NARROW_SELECTION_LOW = {x.lower() for x in NARROW_SELECTION}
NARROW_SELECTION_EXTPAPER_LOW = {x.lower() for x in NARROW_SELECTION_EXTPAPER}


# lower(function_name) -> function_name
FUNCTION_CASEMAP = {x.lower(): x for x in list(ALL_FUNCTIONS.keys())}


def all_functions():
    """
    Merges all functions together
    :return:
    """
    return ALL_FUNCTIONS


def filter_functions(input_set, filter_set):
    """
    Keeps only elements in the filter set
    :param input_set:
    :param filter_set:
    :return:
    """
    ns = {}
    filter_low = {x.lower() for x in filter_set}
    for x in input_set:
        xl = x.lower()
        if xl in filter_low:
            ns[x] = input_set[x]
    return ns


def normalize_function_name(function_name):
    """
    Fixes case of the function name
    :param function_name:
    :return:
    """
    return FUNCTION_CASEMAP[function_name.lower()]


def get_stream_type(stream_name):
    """
    Resolves stream type code to the stream name
    :param stream_name:
    :return:
    """
    stream_name = stream_name.lower()
    for x in STREAM_TYPES:
        if STREAM_TYPES[x] == stream_name:
            return x
    raise ValueError('Unknown stream name')


def function_to_stream_type(function_name):
    """
    Tries to determine function family
    :param function_name:
    :return:
    """
    function_name = normalize_function_name(function_name)
    if function_name in ESTREAM:
        return FUNCTION_ESTREAM
    elif function_name in SHA3:
        return FUNCTION_SHA3
    elif function_name in BLOCK:
        return FUNCTION_BLOCK
    elif function_name in HASH:
        return FUNCTION_HASH
    else:
        raise ValueError('Function family could not be determined for %s' % function_name)


def get_tv_size(stream_type, function_name=None):
    if stream_type == FUNCTION_ESTREAM:
        if function_name:
            fcfg = ESTREAM[normalize_function_name(function_name)]
            if fcfg and fcfg.block_size:
                return fcfg.block_size
        return 16

    if stream_type == FUNCTION_SHA3:
        if function_name:
            fcfg = SHA3[normalize_function_name(function_name)]
            if fcfg and fcfg.block_size:
                return fcfg.block_size
        return 32

    if stream_type == FUNCTION_HASH:
        if function_name:
            fcfg = HASH[normalize_function_name(function_name)]
            if fcfg and fcfg.block_size:
                return fcfg.block_size
        return 32

    if stream_type == FUNCTION_BLOCK:
        if function_name:
            fcfg = BLOCK[normalize_function_name(function_name)]
            if fcfg and fcfg.block_size:
                return fcfg.block_size
        return 16
    return 16


def is_function_egen(fnc):
    """
    Returns true if function is generated by EAcirc generator
    :param fnc:
    :return:
    """
    fl = fnc.lower()
    return fl in FUNCTION_CASEMAP


def is_3des(fnc):
    """
    Is function 3-des?
    Specific key constrains
    :param fnc:
    :return:
    """
    return fnc.lower() == 'triple-des'


def is_narrow(fname, narrow_type=0):
    """
    Returns true if function is in the narrow set
    :param fname:
    :param narrow_type:
    :return:
    """
    lw = fname.lower()
    if narrow_type == 0:
        narrow_set = NARROW_SELECTION_LOW
    elif narrow_type == 1:
        narrow_set = NARROW_SELECTION_EXTPAPER_LOW
    elif narrow_type == 2:
        narrow_set = BENCHMARK_SELECTION
    else:
        raise ValueError('Unknown narrow type')

    for fnc in narrow_set:
        flk = '-a%s-' % fnc
        if flk in lw:
            return True
    return False


class FunctionGenConfig(object):
    def __init__(self, function_name, stream_type=None, tvsize=None, rounds=None, tvcount=None, data=None, params=None, **kwargs):
        self.function_name = None
        self.stream_type = None
        self.tvsize = tvsize
        self.rounds = rounds
        self.params = params  # type: FunctionParams

        if function_name is not None:
            self.function_name = normalize_function_name(function_name)
            self.stream_type = function_to_stream_type(function_name=function_name) if stream_type is None else stream_type
            self.tvsize = get_tv_size(stream_type=self.stream_type, function_name=function_name) if tvsize is None else tvsize
            if self.rounds is None:
                raise ValueError('Rounds is not defined')

        self.num = tvcount
        if data is not None:
            self.num = int(math.ceil(data / float(self.tvsize)))

        if self.num is None:
            raise ValueError('Length of data not specified')

    def to_json(self):
        return dict(self.__dict__)
    
    def __repr__(self):
        return 'FunctionGenConfig(fnc=%r, ftype=%r, tvsize=%r, rounds=%r, num=%r)' \
               % (self.function_name, self.stream_type, self.tvsize, self.rounds, self.num)


def get_scode(stream):
    if 'scode' in stream:
        return stream['scode']
    return 'na'


def is_randomized(stream):
    """
    Returns true if the seed is randomized - affected by different seed values
    :param stream:
    :return:
    """
    seed_dep = common.defvalkey(stream, 'seed_dep', take_none=True)
    if seed_dep is not None:
        return seed_dep

    stype = common.defvalkey(stream, 'type')
    if stype in [StreamCodes.ZERO, StreamCodes.COUNTER]:
        return False

    elif stype in [StreamCodes.RANDOM, StreamCodes.SAC, StreamCodes.SAC_STEP, StreamCodes.RPCS]:
        return True

    elif stype in [StreamCodes.XOR]:
        return is_randomized(stream['source'])

    else:
        raise ValueError('Seed dependency is not known: %s' % stype)


def get_zero_stream(**kwargs):
    return {'type': StreamCodes.ZERO, 'scode': '0'}


def get_random_stream(di=None, **kwargs):
    return {'type': StreamCodes.RANDOM, 'scode': 'rnd' if di is None else 'rnd%s' % di}


def get_sac_stream(**kwargs):
    return {'type': StreamCodes.SAC, 'scode': 'sac'}


def get_sac_step_stream(**kwargs):
    return {'type': StreamCodes.SAC_STEP, 'scode': 'sacstep'}


def get_counter_stream(**kwargs):
    return {'type': StreamCodes.COUNTER, 'scode': 'ctr'}


def get_xor_stream(source=None, **kwargs):
    return {'type': StreamCodes.XOR, 'scode': 'xor', 'source': source}


def get_rpcs_stream(**kwargs):
    return {'type': StreamCodes.RPCS, 'scode': 'rpcs'}


def get_hw_stream(hw=3, increase_hw=False, randomize_start=False, randomize_overflow=False, **kwargs):
    ob = collections.OrderedDict()
    ob['type'] = 'hw_counter'
    ob['hw'] = hw
    ob['scode'] = 'hw%d%s%s%s' % (hw,
                                  '' if not randomize_overflow else 'r',
                                  '' if not randomize_start else 's',
                                  '' if not increase_hw else 'i')

    ob['increase_hw'] = bool(increase_hw) if increase_hw else False
    if randomize_start:
        ob['randomize_start'] = bool(randomize_start)
    if randomize_overflow:
        ob['randomize_overflow'] = bool(randomize_overflow)

    ob['seed_dep'] = bool(randomize_start) or bool(randomize_overflow)
    return ob


def column_stream(source, size=16):
    ob = collections.OrderedDict()
    ob['type'] = 'column_stream'
    ob['size'] = size
    ob['scode'] = 'col-' + get_scode(source)
    ob['source'] = source
    return ob


def pick_iv_size(params, default=None):
    """
    Picks IV size for the function
    :param params:
    :type params: FunctionParams
    :param default:
    :return:
    """
    if params is None or params.iv_size is None:
        return default
    if isinstance(params.iv_size, (list, tuple)):
        return params.iv_size[0]
    return params.iv_size


def pick_iv_size_fcfg(fun_cfg, default=None):
    """

    :param fun_cfg:
    :param default:
    :return:
    """
    if fun_cfg is None:
        return default
    return pick_iv_size(fun_cfg.params, default)


def get_function_config(func_cfg,
                        init_frequency='only_once', mode='ECB', generator='pcg32',
                        src_input=None, src_key=None, src_iv=None, **kwargs):
    """
    Function stream.

    :param func_cfg:
    :type func_cfg: FunctionGenConfig
    :param init_frequency:
    :param mode:
    :param generator:
    :param src_input:
    :param src_key: or random by default
    :param src_iv: or zero by defualt
    :return:
    """
    if init_frequency == 'o' or init_frequency is None:
        init_frequency = 'only_once'
    elif init_frequency == 'e':
        init_frequency = 'every_vector'

    fname = func_cfg.params.fname if func_cfg and func_cfg.params and func_cfg.params.fname else func_cfg.function_name
    stream_obj = collections.OrderedDict()
    stream_obj['type'] = STREAM_TYPES[func_cfg.stream_type]
    stream_obj['type_code'] = func_cfg.stream_type
    stream_obj['generator'] = generator
    stream_obj['algorithm'] = fname
    stream_obj['round'] = func_cfg.rounds
    stream_obj['block_size'] = func_cfg.tvsize

    def_input = get_random_stream() if func_cfg.stream_type in [FUNCTION_SHA3, FUNCTION_HASH] else get_zero_stream()

    src_iv = src_iv if src_iv else get_zero_stream()
    src_input = src_input if src_input else def_input
    src_key = src_key if src_key else get_random_stream()

    stream_obj['gen_inp'] = src_input
    stream_obj['gen_key'] = src_key
    stream_obj['scode_inp'] = get_scode(src_input)
    stream_obj['scode_key'] = get_scode(src_key)
    stream_obj['seed_dep'] = is_randomized(src_input) or is_randomized(src_iv) or is_randomized(src_key)

    if func_cfg.stream_type == FUNCTION_BLOCK and init_frequency == 'every_vector':
        init_frequency = 1

    if init_frequency == 'only_once':
        stream_obj['scode_init'] = '0'
    elif init_frequency == 'every_vector':
        stream_obj['scode_init'] = 'e'
    else:
        stream_obj['scode_init'] = init_frequency

    stream_obj['scode'] = 'tp%s-a%s-r%s-tv%s-in%s-k%s-ri%s' \
                          % (stream_obj['type'], fname, func_cfg.rounds, func_cfg.tvsize,
                             stream_obj['scode_inp'], stream_obj['scode_key'], stream_obj['scode_init'])

    if func_cfg.stream_type == FUNCTION_BLOCK:
        stream_obj['init_frequency'] = str(init_frequency)
        stream_obj['key_size'] = BLOCK[func_cfg.function_name].key_size
        stream_obj['plaintext'] = src_input
        stream_obj['key'] = src_key
        stream_obj['iv'] = src_iv
        stream_obj['mode'] = mode

    elif func_cfg.stream_type == FUNCTION_ESTREAM:
        stream_obj['init_frequency'] = str(init_frequency)
        stream_obj['key_size'] = ESTREAM[func_cfg.function_name].key_size if ESTREAM[func_cfg.function_name].key_size else 16
        stream_obj['plaintext'] = src_input
        stream_obj['key'] = src_key
        stream_obj['iv'] = src_iv

        iv_size = pick_iv_size_fcfg(func_cfg, None)
        if iv_size:
            stream_obj['iv_size'] = iv_size

    elif func_cfg.stream_type in [FUNCTION_SHA3, FUNCTION_HASH]:
        stream_obj['source'] = src_input
        stream_obj['hash_size'] = func_cfg.tvsize

    else:
        raise ValueError('Unknown stream type')

    return stream_obj


def zero_inp_reinit_key(func_cfg, key=None, **kwargs):
    """
    Init every-vector, zero input vector, random key by default
    :param func_cfg:
    :param kwargs:
    :return:
    """
    return get_function_config(func_cfg, init_frequency='every_vector',
                               src_input=get_zero_stream(), src_key=key, src_iv=get_zero_stream())


def rpcs_raw(source, **kwargs):
    ob = collections.OrderedDict()
    ob['type'] = StreamCodes.RPCS
    ob['scode'] = 'rpcs-' + get_scode(source)
    ob['source'] = source
    return ob


def xors_raw(source, **kwargs):
    ob = collections.OrderedDict()
    ob['type'] = StreamCodes.XOR
    ob['scode'] = 'xor-' + get_scode(source)
    ob['source'] = source
    return ob


def rpcs_inp(func_cfg, key=None, **kwargs):
    fcfg = get_function_config(func_cfg, src_input=get_zero_stream(), src_key=key)
    return rpcs_raw(fcfg)


def rpcs_inp_xor(func_cfg, key=None, **kwargs):
    return xors_raw(rpcs_inp(func_cfg, key=key))


def sac_inp(func_cfg, key=None, **kwargs):
    return get_function_config(func_cfg, src_input=get_sac_stream(), src_key=key)


def sac_xor_inp(func_cfg, key=None, **kwargs):
    return xors_raw(sac_inp(func_cfg, key=key, **kwargs))


def get_config_header(func_cfg, seed='1fe40505e131963c', stdout=None, filename=None, stream=None):
    """
    Generates function name.

    :param func_cfg:
    :type func_cfg: FunctionGenConfig

    :param seed:
    :param num:
    :param stdout:
    :param filename:
    :param stream:
    :return:
    """
    # TV-size hack
    tv_size = func_cfg.tvsize
    if stream and stream['type'] == StreamCodes.RPCS:
        # tv_size *= 2
        raise ValueError('RPCS not supported yet')

    js = collections.OrderedDict()
    js['notes'] = 'Configuration generated by poly-verif-egen'
    js['seed'] = seed
    js['function_name'] = func_cfg.function_name
    js['stream_type'] = func_cfg.stream_type
    js['tv_size'] = tv_size
    js['tv_count'] = func_cfg.num
    if stdout:
        js['stdout'] = True
    if filename:
        js['file_name'] = filename
    js['stream'] = stream
    return js


# noinspection PyUnusedLocal
def get_config(function_name, rounds=None, seed='1fe40505e131963c', stream_type=None,
               tvsize=None, tvcount=None, data=None,
               generator='pcg32', init_frequency='only_once', mode='ECB', plaintext_type='counter',
               stdout=False, filename=None,
               *args, **kwargs):
    """
    Generates generator configuration object.

    :param function_name: algorithm to use as a generator. e.g., AES
    :param rounds: number of rounds of function_name
    :param seed: seed of the PRNG used to generate input/keys/IVs
    :param stream_type: type of the algorithm user, e.g., estream, sha3,... Optional, determined automatically.
    :param tvsize: size of the test vector / block length of the primitive. Determined automatically.
    :param tvcount: number of testvectors to produce on the output
    :param data: number of bytes to produce on the output
    :param generator: PRNG used to generate inputs/keys/IVs to the generating algorithm
    :param init_frequency: initialization frequency for keys/IVs
    :param mode: encryption mode for block ciphers, ECB by default
    :param plaintext_type: algorithm for generating input of the algorithm to process, counter mode is the
            only one supported at the moment.
    :param args:
    :param kwargs:
    :return:
    """
    if function_name is None:
        raise ValueError('Function is null')
    function_name = normalize_function_name(function_name)

    if stream_type is None:
        stream_type = function_to_stream_type(function_name=function_name)

    if tvsize is None:
        tvsize = get_tv_size(stream_type=stream_type, function_name=function_name)

    num = tvcount
    if data is not None:
        num = int(math.ceil(data / float(tvsize)))

    if num is None:
        raise ValueError('Length of data not specified')

    if rounds is None:
        raise ValueError('Rounds is not defined')

    js = collections.OrderedDict()
    js['notes'] = 'Configuration generated by poly-verif-egen'
    js['seed'] = seed
    js['tv_size'] = tvsize
    js['tv_count'] = num
    if stdout:
        js['stdout'] = True
    if filename:
        js['file_name'] = filename

    stream_obj = collections.OrderedDict()
    stream_obj['type'] = STREAM_TYPES[stream_type]
    stream_obj['generator'] = generator
    stream_obj['algorithm'] = function_name
    stream_obj['round'] = rounds
    stream_obj['block_size'] = tvsize

    if stream_type == FUNCTION_BLOCK:
        stream_obj['init_frequency'] = str(init_frequency)
        stream_obj['key_size'] = BLOCK[function_name].key_size
        stream_obj['plaintext'] = {'type': plaintext_type}
        stream_obj['key'] = {'type': StreamCodes.RANDOM}
        stream_obj['iv'] = {'type': 'false_stream'}
        stream_obj['mode'] = mode

    elif stream_type == FUNCTION_ESTREAM:
        stream_obj['init_frequency'] = str(init_frequency)
        stream_obj['key_size'] = 16
        stream_obj['plaintext'] = {'type': plaintext_type}
        stream_obj['key'] = {'type': 'random'}
        stream_obj['iv'] = {'type': 'false_stream'}

    elif stream_type == FUNCTION_SHA3 or stream_type == FUNCTION_HASH:
        stream_obj['source'] = {'type': plaintext_type}
        stream_obj['hash_bitsize'] = 8 * tvsize

    else:
        raise ValueError('Unknown stream type')

    js['stream'] = stream_obj

    return js


def determine_stream(code):
    """
    Attempts to determine stream type from code
    :param code:
    :return:
    """
    if code is None or code == '0':
        return get_zero_stream()
    if code.startswith('rnd'):
        rnd_code = code[3:]
        return get_random_stream(None if not rnd_code else int(rnd_code))
    if code == 'sac':
        return get_sac_stream()
    if code == 'sac_step' or code == 'sacstep':
        return get_sac_step_stream()
    if code == 'ctr':
        return get_counter_stream()
    if code == 'rpcs':
        return get_rpcs_stream()
    if code.startswith('hw'):
        m = re.match(r'^hw([0-9]+)(r)?(s)?(i)?$', code)
        if not m:
            raise ValueError('Unknown hamming weight configuration: %s' % code)
        return get_hw_stream(hw=int(m.group(1)), increase_hw=m.group(4),
                             randomize_start=m.group(3), randomize_overflow=m.group(2))
    if code.startswith('xor-'):
        ob = get_xor_stream()
        ob['source'] = determine_stream(code[4:])
        return ob

    raise ValueError('Unknown stream code: %s' % code)


def determine_strategy(strategy, fgc, iv=None):
    """
    Parses strategy string
    :param strategy:
    :param fgc:
    :param iv:
    :return:
    """
    match = re.match(r'^(.+?-)?in(.+?)-k(.+?)(-iv.+?)?-ri(.+?)$', strategy)
    if not match:
        raise ValueError('Unrecognized strategy string: %s' % strategy)

    pref_group = match.group(1)
    if pref_group:
        if pref_group.startswith('xor-'):
            return xors_raw(determine_strategy(strategy[4:], fgc, iv))

        elif pref_group.startswith('rpcs-'):
            return rpcs_raw(determine_strategy(strategy[5:], fgc, iv))

        else:
            raise ValueError('Unrecognized strategy prefix %s' % pref_group)

    iv_in = match.group(4)
    input = match.group(2)
    key = match.group(3)
    reinit = int(match.group(5))
    iv = iv if iv_in is None else iv_in[3:]

    iv_s = determine_stream(iv)
    key_s = determine_stream(key)
    input_s = determine_stream(input)

    fun_cfg = get_function_config(fgc, src_input=input_s, src_key=key_s, src_iv=iv_s,
                                  init_frequency='every_vector' if reinit else None)

    return fun_cfg


def generate_config(args):
    """

    :param args:
    :return:
    """
    fnc = args.alg
    if not is_function_egen(fnc):
        raise ValueError('Function not known')

    fnc = normalize_function_name(fnc)
    params = ALL_FUNCTIONS[fnc] if fnc in ALL_FUNCTIONS else None

    fgc = FunctionGenConfig(fnc, rounds=args.round, data=args.size*1024*1024, params=params)

    input = args.input
    iv = args.iv
    key = args.key
    reinit = args.reinit

    strategy = args.strategy
    if strategy:
        fun_cfg = determine_strategy(strategy, fgc, iv)

    else:
        iv_s = determine_stream(iv)
        key_s = determine_stream(key)
        input_s = determine_stream(input)
        fun_cfg = get_function_config(fgc, src_input=input_s, src_key=key_s, iv=iv_s, init_frequency='every_vector' if reinit else None)

    seed = args.seed
    if args.seed_random:
        rand = random.Random()
        seed = '%016x' % rand.getrandbits(8*8)

    elif args.seed_code is not None:
        seed = common.generate_seed(args.seed_code)

    if seed is None:
        seed = common.generate_seed(0)

    config = get_config_header(fgc, stdout=True, stream=fun_cfg, seed=seed)
    return config


def main():
    """
    Generating test configurations
    :return:
    """
    parser = argparse.ArgumentParser(description='Generate test configurations')

    parser.add_argument('--type', dest='type', default=None,
                        help='Algorithm type')

    parser.add_argument('--alg', dest='alg', default=None, required=True,
                        help='Algorithm')

    parser.add_argument('--round', dest='round', default=1, type=int,
                        help='Round')

    parser.add_argument('--size', dest='size', default=1, type=int,
                        help='MB of data to generate')

    parser.add_argument('--in', dest='input', default=None,
                        help='Input')

    parser.add_argument('--key', dest='key', default=None,
                        help='key')

    parser.add_argument('--iv', dest='iv', default=None,
                        help='iv')

    parser.add_argument('--reinit', dest='reinit', default=None,
                        help='reinit')

    parser.add_argument('--seed', dest='seed', default=None,
                        help='seed')

    parser.add_argument('--seed-code', dest='seed_code', default=None, type=int,
                        help='seed index')

    parser.add_argument('--seed-random', dest='seed_random', default=False, action='store_const', const=True,
                        help='Randomize seed')

    parser.add_argument('--strategy', dest='strategy', default=None,
                        help='Strategy')

    args = parser.parse_args()

    config = generate_config(args)

    import json
    print(common.json_dumps(config, indent=2))


if __name__ == '__main__':
    main()


