#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections

# Regex for parsing arg names from source codes
# if \(name == "(.+?)".+


class FunctionParams:
    def __init__(self, block_size, key_size):
        self.block_size = block_size
        self.key_size = key_size


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
    'DECIM': None,
    'DICING': None,
    'Dragon': None,
    'Edon80': None,
    'F-FCSR': None,
    'Fubuki': None,
    'Grain': None,
    'HC-128': None,
    'Hermes': None,
    'LEX': None,
    'MAG': None,
    'MICKEY': None,
    'Mir-1': None,
    'Pomaranch': None,
    'Py': None,
    'Rabbit': None,
    'Salsa20': None,
    'SFINKS': None,
    'SOSEMANUK': None,
    'TEA': None,
    'TSC-4': None,
    'WG': None,
    'Zk-Crypt': None
}

# SHA3
SHA3 = {
    'Abacus': None,
    'ARIRANG': None,
    'AURORA': None,
    'BLAKE': None,
    'Blender': None,
    'BMW': None,
    'Boole': None,
    'Cheetah': None,
    'CHI': None,
    'CRUNCH': None,
    'CubeHash': None,
    'DCH': None,
    'DynamicSHA': None,
    'DynamicSHA2': None,
    'ECHO': None,
    'EDON': None,
    'ESSENCE': None,
    'Fugue': None,
    'Grostl': None,
    'Hamsi': None,
    'JH': None,
    'Keccak': None,
    'Khichidi': None,
    'LANE': None,
    'Lesamnta': None,
    'Luffa': None,
    'MCSSHA3': None,
    'MD6': None,
    'MeshHash': None,
    'NaSHA': None,
    'Sarmal': None,
    'Shabal': None,
    'SHAMATA': None,
    'SHAvite3': None,
    'SIMD': None,
    'Skein': None,
    'SpectralHash': None,
    'StreamHash'
    'Tangle': None,
    'Twister': None,
    'WaMM': None,
    'Waterfall': None,
    'Tangle2': None
}


BLOCK = {
    'TEA': FunctionParams(8, 16),
    'AES': FunctionParams(16, 16),
    'RC4': FunctionParams(16, 16),
    'SINGLE-DES': FunctionParams(8, 8),
    'TRIPLE-DES': FunctionParams(8, 24)
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
    'Twister': [6, 7, 8, 9]
}


# lower(function_name) -> function_name
FUNCTION_CASEMAP = {x.lower(): x for x in list(ESTREAM.keys() + SHA3.keys() + BLOCK.keys())}


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
        num = data // tvsize

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



