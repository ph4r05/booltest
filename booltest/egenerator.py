#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import math

# Regex for parsing arg names from source codes
# if \(name == "(.+?)".+


class FunctionParams(object):
    __slots__ = ['block_size', 'key_size', 'iv_size', 'rounds', 'min_rounds']

    def __init__(self, block_size=None, key_size=None, rounds=None, min_rounds=None, iv_size=None):
        self.block_size = block_size
        self.key_size = key_size
        self.iv_size = iv_size
        self.rounds = rounds
        self.min_rounds = min_rounds

    def to_json(self):
        return dict(self.__dict__)

    def __repr__(self):
        return 'FunctionParams(block_size=%r, key_size=%r, iv_size=%r, rounds=%r, min_rounds=%r)' \
               % (self.block_size, self.key_size, self.iv_size, self.rounds, self.min_rounds)


FUNCTION_ESTREAM = 1
FUNCTION_SHA3 = 2
FUNCTION_BLOCK = 3


STREAM_TYPES = {
    FUNCTION_ESTREAM: 'estream',
    FUNCTION_SHA3: 'sha3',
    FUNCTION_BLOCK: 'block',
}


class StreamCodes:
    ZERO = 'false-stream'
    RANDOM = 'pcg32-stream'
    SAC = 'sac'
    SAC_STEP = 'sac-step'
    COUNTER = 'counter'
    XOR = 'xor-stream'
    RPCS = 'rnd-plt-ctx-stream'


# eStream
ESTREAM = {
    'ABC': None,
    'Achterbahn': None,
    'CryptMT': None,
    'DECIM': FunctionParams(rounds=8, iv_size=32),
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
    'Trivium': FunctionParams(rounds=9),
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
    'Tangle2': FunctionParams(rounds=80)
}


BLOCK = {
    'BLOWFISH': FunctionParams(8, 56, rounds=16),
    'TEA': FunctionParams(8, 16, rounds=64),
    'AES': FunctionParams(16, 16, rounds=10),
    'RC4': FunctionParams(16, 16),
    'SIMON': FunctionParams(16, 16, rounds=32),
    'SPECK': FunctionParams(16, 16, rounds=22),
    'SINGLE-DES': FunctionParams(8, 8, rounds=16),
    'TRIPLE-DES': FunctionParams(8, 24, rounds=16)
}

# Interesting rounds to test
ROUNDS = {
    'AES': [1, 3, 4, 10],
    'ARIRANG': [3, 4],
    'AURORA': [2, 3],
    'BLAKE': [1, 2],
    'Cheetah': [4, 5],
    'CubeHash': [0, 1],
    'DCH': [1, 2],
    'DECIM': [5, 6],
    'DynamicSHA': [7, 8],
    'DynamicSHA2': [11, 12],
    'ECHO': [1, 2],
    'Fubuki': [2, 3],
    'Grain': [2, 3],
    'Grostl': [2, 3],
    'Hamsi': [0, 1],
    'Hermes': [1, 2],
    'JH': [6, 7],
    'Keccak': [3, 4],
    'LEX': [3, 4],
    'Lesamnta': [2, 3],
    'Luffa': [7, 8],
    'MD6': [8, 9, 10],
    'Rabbit': [1, 2, 3],
    'SIMD': [0, 1],
    'Salsa20': [3, 4, 5],
    'SIMON': [1, 2, 3],
    'SPECK': [1, 2, 3],
    'TEA': [4, 5],
    'TSC-4': [12, 13, 14],
    'Tangle': [22, 23],
    'Twister': [6, 7],
}


# lower(function_name) -> function_name
FUNCTION_CASEMAP = {x.lower(): x for x in list(list(ESTREAM.keys()) + list(SHA3.keys()) + list(BLOCK.keys()))}


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
    return fnc in ROUNDS\
           or fnc in SHA3 \
           or fnc in ESTREAM\
           or fnc in BLOCK


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

    def __repr__(self):
        return 'FunctionGenConfig(fnc=%r, ftype=%r, tvsize=%r, rounds=%r, num=%r)' \
               % (self.function_name, self.stream_type, self.tvsize, self.rounds, self.num)


def get_scode(stream):
    if 'scode' in stream:
        return stream['scode']
    return 'na'


def get_zero_stream(**kwargs):
    return {'type': 'false-stream', 'scode': '0'}


def get_random_stream(di=None, **kwargs):
    return {'type': 'pcg32-stream', 'scode': 'rnd' if di is None else 'rnd%s' % di}


def get_sac_stream(**kwargs):
    return {'type': 'sac', 'scode': 'sac'}


def get_sac_step_stream(**kwargs):
    return {'type': 'sac-step', 'scode': 'sacstep'}


def get_counter_stream(**kwargs):
    return {'type': 'counter', 'scode': 'ctr'}


def get_xor_stream(**kwargs):
    return {'type': 'xor-stream', 'scode': 'xor'}


def get_rpcs_stream(**kwargs):
    return {'type': 'rnd-plt-ctx-stream', 'scode': 'rpcs'}


def get_hw_stream(hw=3, increase_hw=False, randomize_start=False, **kwargs):
    ob = collections.OrderedDict()
    ob['type'] = 'hw-counter'
    ob['hw'] = hw
    ob['scode'] = 'hw%d' % hw
    ob['increase_hw'] = increase_hw
    ob['randomize_start'] = randomize_start
    return ob


def column_stream(source, size=16):
    ob = collections.OrderedDict()
    ob['type'] = 'column-stream'
    ob['size'] = size
    ob['scode'] = 'col-' + get_scode(source)
    ob['source'] = source
    return ob


def get_function_config(func_cfg,
                        init_frequency='only-once', mode='ECB', generator='pcg32',
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
    if init_frequency == 'o':
        init_frequency = 'only-once'
    elif init_frequency == 'e':
        init_frequency = 'every-vector'

    stream_obj = collections.OrderedDict()
    stream_obj['type'] = STREAM_TYPES[func_cfg.stream_type]
    stream_obj['type_code'] = func_cfg.stream_type
    stream_obj['generator'] = generator
    stream_obj['algorithm'] = func_cfg.function_name
    stream_obj['round'] = func_cfg.rounds
    stream_obj['block-size'] = func_cfg.tvsize

    def_input = get_random_stream() if func_cfg.stream_type == FUNCTION_SHA3 else get_zero_stream()

    src_iv = src_iv if src_iv else get_zero_stream()
    src_input = src_input if src_input else def_input
    src_key = src_key if src_key else get_random_stream()

    stream_obj['gen_inp'] = src_input
    stream_obj['gen_key'] = src_key
    stream_obj['scode_inp'] = get_scode(src_input)
    stream_obj['scode_key'] = get_scode(src_key)

    if func_cfg.stream_type == FUNCTION_BLOCK and init_frequency == 'every-vector':
        init_frequency = 1

    if init_frequency == 'only-once':
        stream_obj['scode_init'] = '0'
    elif init_frequency == 'every-vector':
        stream_obj['scode_init'] = 'e'
    else:
        stream_obj['scode_init'] = init_frequency

    stream_obj['scode'] = 'tp%s-a%s-r%s-tv%s-in%s-k%s-ri%s' \
                          % (stream_obj['type'], func_cfg.function_name, func_cfg.rounds, func_cfg.tvsize,
                             stream_obj['scode_inp'], stream_obj['scode_key'], stream_obj['scode_init'])

    if func_cfg.stream_type == FUNCTION_BLOCK:
        stream_obj['init-frequency'] = str(init_frequency)
        stream_obj['key-size'] = BLOCK[func_cfg.function_name].key_size
        stream_obj['plaintext'] = src_input
        stream_obj['key'] = src_key
        stream_obj['iv'] = src_iv
        stream_obj['mode'] = mode

    elif func_cfg.stream_type == FUNCTION_ESTREAM:
        stream_obj['init-frequency'] = str(init_frequency)
        stream_obj['key-size'] = 16
        stream_obj['plaintext-type'] = src_input
        stream_obj['key-type'] = src_key
        stream_obj['iv-type'] = src_iv
        if func_cfg.params and func_cfg.params.iv_size:
            stream_obj['iv-size'] = func_cfg.params.iv_size

    elif func_cfg.stream_type == FUNCTION_SHA3:
        stream_obj['source'] = src_input
        stream_obj['hash-size'] = func_cfg.tvsize

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
    return get_function_config(func_cfg, init_frequency='every-vector',
                               src_input=get_zero_stream(), src_key=key, src_iv=get_zero_stream())


def rpcs_inp(func_cfg, key=None, **kwargs):
    ob = collections.OrderedDict()
    fcfg = get_function_config(func_cfg, src_input=get_zero_stream(), src_key=key)
    ob['type'] = StreamCodes.RPCS
    ob['scode'] = 'rpcs-' + get_scode(fcfg)
    ob['source'] = fcfg
    return ob


def rpcs_inp_xor(func_cfg, key=None, **kwargs):
    ob2 = rpcs_inp(func_cfg, key=key)

    ob = collections.OrderedDict()
    ob['type'] = StreamCodes.XOR
    ob['scode'] = 'xor-' + ob2['scode']
    ob['source'] = ob2
    return ob


def sac_inp(func_cfg, key=None, **kwargs):
    return get_function_config(func_cfg, src_input=get_sac_stream(), src_key=key)


def sac_xor_inp(func_cfg, key=None, **kwargs):
    ob = collections.OrderedDict()
    fcfg = sac_inp(func_cfg, key=key, **kwargs)
    ob['type'] = StreamCodes.XOR
    ob['scode'] = 'xor-' + get_scode(fcfg)
    ob['source'] = fcfg
    return ob


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
        tv_size *= 2

    js = collections.OrderedDict()
    js['notes'] = 'Configuration generated by poly-verif-egen'
    js['seed'] = seed
    js['function_name'] = func_cfg.function_name
    js['stream_type'] = func_cfg.stream_type
    js['tv-size'] = tv_size
    js['tv-count'] = func_cfg.num
    if stdout:
        js['stdout'] = True
    if filename:
        js['file-name'] = filename
    js['stream'] = stream
    return js


# noinspection PyUnusedLocal
def get_config(function_name, rounds=None, seed='1fe40505e131963c', stream_type=None,
               tvsize=None, tvcount=None, data=None,
               generator='pcg32', init_frequency='only-once', mode='ECB', plaintext_type='counter',
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
    js['tv-size'] = tvsize
    js['tv-count'] = num
    if stdout:
        js['stdout'] = True
    if filename:
        js['file-name'] = filename

    stream_obj = collections.OrderedDict()
    stream_obj['type'] = STREAM_TYPES[stream_type]
    stream_obj['generator'] = generator
    stream_obj['algorithm'] = function_name
    stream_obj['round'] = rounds
    stream_obj['block-size'] = tvsize

    if stream_type == FUNCTION_BLOCK:
        stream_obj['init-frequency'] = str(init_frequency)
        stream_obj['key-size'] = BLOCK[function_name].key_size
        stream_obj['plaintext'] = {'type': plaintext_type}
        stream_obj['key'] = {'type': 'pcg32-stream'}
        stream_obj['iv'] = {'type': 'false-stream'}
        stream_obj['mode'] = mode

    elif stream_type == FUNCTION_ESTREAM:
        stream_obj['init-frequency'] = str(init_frequency)
        stream_obj['key-size'] = 16
        stream_obj['plaintext'] = {'type': plaintext_type}
        stream_obj['plaintext-type'] = {'type': plaintext_type}
        stream_obj['key-type'] = {'type': 'random'}
        stream_obj['key-type'] = 'random'
        stream_obj['iv'] = {'type': 'zeros'}
        stream_obj['iv-type'] = 'zeros'

    elif stream_type == FUNCTION_SHA3:
        stream_obj['source'] = {'type': plaintext_type}
        stream_obj['hash-bitsize'] = 8 * tvsize

    else:
        raise ValueError('Unknown stream type')

    js['stream'] = stream_obj

    return js



