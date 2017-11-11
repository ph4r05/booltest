#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import math

# Regex for parsing arg names from source codes
# if \(name == "(.+?)".+


class FunctionParams:
    def __init__(self, block_size=None, key_size=None, rounds=None, min_rounds=None):
        self.block_size = block_size
        self.key_size = key_size
        self.rounds = rounds
        self.min_rounds = min_rounds


FUNCTION_ESTREAM = 1
FUNCTION_SHA3 = 2
FUNCTION_BLOCK = 3


STREAM_TYPES = {
    FUNCTION_ESTREAM: 'estream',
    FUNCTION_SHA3: 'sha3',
    FUNCTION_BLOCK: 'block'
}


# eStream
ESTREAM = {
    'ABC': None,
    'Achterbahn': None,
    'CryptMT': None,
    'DECIM': FunctionParams(rounds=8),
    'DICING': None,
    'Dragon': None,
    'Edon80': None,
    'F-FCSR': None,
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
    'Rabbit': None,
    'Salsa20': FunctionParams(rounds=12),
    'SFINKS': None,
    'SOSEMANUK': None,
    'TSC-4': FunctionParams(rounds=32),
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
    'CRUNCH': None,
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
    'LANE': None,
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
    'TEA': FunctionParams(8, 16, rounds=64),
    'AES': FunctionParams(16, 16, rounds=10),
    'RC4': FunctionParams(16, 16),
    'SINGLE-DES': FunctionParams(8, 8, rounds=16),
    'TRIPLE-DES': FunctionParams(8, 24, rounds=16)
}

# Interesting rounds to test
ROUNDS = {
    'DECIM': [5, 6, 7],
    'Fubuki': [2, 3, 4],
    'Grain': [2, 3, 4],
    'Hermes': [1, 2, 3],
    'LEX': [3, 4, 5],
    'Salsa20': [2, 3, 4],
    'TSC-4': [10, 11, 12, 13, 14],
    'ARIRANG': [2, 3, 4],
    'AURORA': [2, 3, 4, 5],
    'BLAKE': [0, 1, 2, 3],
    'Cheetah': [4, 5, 6, 7],
    'CubeHash': [0, 1, 2, 3],
    'DCH': [1, 2, 3],
    'DynamicSHA': [7, 8, 9],
    'DynamicSHA2': [11, 12, 13, 14],
    'ECHO': [1, 2, 3, 4],
    'Grostl': [2, 3, 4, 5],
    'Hamsi': [0, 1, 2, 3],
    'JH': [6, 7, 8],
    'Lesamnta': [2, 3, 4, 5],
    'Luffa': [6, 7, 8],
    'MD6': [8, 9, 10, 11],
    'SIMD': [0, 1, 2, 3],
    'Tangle': [22, 23, 24, 25],
    'Twister': [6, 7, 8, 9],
    'Keccak': [2, 3, 4],
    'AES': [2, 3, 4],
    'TEA': [4, 5, 6, 7, 8]
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
        return 16
    if stream_type == FUNCTION_SHA3:
        return 32
    if stream_type == FUNCTION_BLOCK:
        return BLOCK[normalize_function_name(function_name)].block_size
    return 16


# noinspection PyUnusedLocal
def get_config(function_name, rounds=None, seed='1fe40505e131963c', stream_type=None,
               tvsize=None, tvcount=None, data=None,
               generator='pcg32', init_frequency='only-once', mode='ECB', plaintext_type='counter',
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

    stream_obj = collections.OrderedDict()
    stream_obj['type'] = STREAM_TYPES[stream_type]
    stream_obj['generator'] = generator
    stream_obj['algorithm'] = function_name
    stream_obj['round'] = rounds
    stream_obj['block-size'] = tvsize

    if stream_type == FUNCTION_BLOCK:
        stream_obj['init-frequency'] = init_frequency
        stream_obj['key-size'] = BLOCK[function_name].key_size
        stream_obj['plaintext'] = {'type': plaintext_type}
        stream_obj['key'] = {'type': 'pcg32-stream'}
        stream_obj['iv'] = {'type': 'false-stream'}
        stream_obj['mode'] = mode

    elif stream_type == FUNCTION_ESTREAM:
        stream_obj['init-frequency'] = init_frequency
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



