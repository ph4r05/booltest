"""
Provides basic cryptographic utilities for the Python EnigmaBridge client, e.g.,
generating random numbers, encryption, decryption, padding, etc...

For now we use PyCrypto, later we may use pure python implementations to minimize dependency count.
"""

import logging
import os
import base64
import types
import struct

from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Util.py3compat import *
from Crypto.Util.number import long_to_bytes, bytes_to_long, size, ceil_div

# Logging if needed
logger = logging.getLogger(__name__)


__author__ = 'dusanklinec'


#
# Utils
#


def to_bytes(x, blocksize=0):
    """
    Converts input to a byte string.
    Typically used in PyCrypto as an argument (e.g., key, iv)

    :param x: string (does nothing), bytearray, array with numbers
    :return:
    """
    if isinstance(x, bytearray):
        return left_zero_pad(''.join([bchr(y) for y in x]), blocksize)
    elif isinstance(x, basestring):
        return left_zero_pad(x, blocksize)
    elif isinstance(x, (list, tuple)):
        return left_zero_pad(''.join([bchr(y) for y in bytearray(x)]), blocksize)
    elif isinstance(x, (types.LongType, types.IntType)):
        return long_to_bytes(x, blocksize)
    else:
        raise ValueError('Unknown input argument type')


def to_long(x):
    """
    Converts input to a long number (arbitrary precision python long)
    :param x:
    :return:
    """
    if isinstance(x, types.LongType):
        return x
    elif isinstance(x, types.IntType):
        return long(x)
    else:
        return bytes_to_long(to_bytes(x))


def to_bytearray(x):
    """
    Converts input to byte array.
    If already a byte array, return directly.

    :param x:
    :return:
    """
    if isinstance(x, bytearray):
        return x
    else:
        return bytearray(x)


def to_hex(x):
    """
    Converts input to the hex string
    :param x:
    :return:
    """
    if isinstance(x, bytearray):
        return x.decode('hex')
    elif isinstance(x, basestring):
        return base64.b16encode(x)
    elif isinstance(x, (list, tuple)):
        return bytearray(x).decode('hex')
    else:
        raise ValueError('Unknown input argument type')


def from_hex(x):
    """
    Converts hex-coded (b16 encoding) string to the byte string.
    :param x:
    :return:
    """
    return base64.b16decode(x, True)


def long_bit_size(x):
    return size(x)


def long_byte_size(x):
    return ceil_div(long_bit_size(x), 8)


def get_zero_vector(numBytes):
    """
    Generates a zero vector of a given size

    :param numBytes:
    :return:
    """
    return bytearray([0] * numBytes).decode('ascii')


def left_zero_pad(s, blocksize):
    """
    Left padding with zero bytes to a given block size

    :param s:
    :param blocksize:
    :return:
    """
    if blocksize > 0 and len(s) % blocksize:
        s = (blocksize - len(s) % blocksize) * b('\000') + s
    return s


def str_equals(a, b):
    """
    Constant time string equals method - no time leakage
    :param a:
    :param b:
    :return:
    """
    al = len(a)
    bl = len(b)
    match = True
    for i in xrange(0, min(al, bl)):
        match &= a[i] == b[i]
    return match


def bytes_replace(byte_str, start_idx, stop_idx, replacement):
    """
    Replaces given portion of the byte string with the replacement, returns new array
    :param bytes:
    :param start_idx:
    :param stop_idx:
    :param replacement:
    :return:
    """
    return byte_str[:start_idx] + replacement + byte_str[stop_idx:]


def bytes_transform(byte_str, start_idx, stop_idx, fction):
    """
    Takes portion of the byte array and passes it to the function for transformation.
    Result is replaced in the byte string, new one is created.
    :param bytes:
    :param start_idx:
    :param stop_idx:
    :param fction:
    :return:
    """
    return bytes_replace(byte_str, start_idx, stop_idx, fction(byte_str[start_idx:stop_idx]))


def bytes_to_short(byte, offset=0):
    return struct.unpack('>H', byte[offset:offset+2])[0]


def short_to_bytes(short):
    return struct.pack('>H', int(short))


def bytes_to_byte(byte, offset=0):
    return struct.unpack('>B', byte[offset:offset+1])[0]


def byte_to_bytes(byte):
    return struct.pack('>B', int(byte) & 0xFF)


#
# Randomness
#


def get_random_vector(numBytes):
    #return Random.get_random_bytes(numBytes)
    return os.urandom(numBytes)


def get_random_integer(N, randfunc=None):
    """getRandomInteger(N:int, randfunc:callable):long
    Return a random number with at most N bits.

    If randfunc is omitted, then Random.new().read is used.

    This function is for internal use only and may be renamed or removed in
    the future.
    """
    if randfunc is None:
        randfunc = Random.new().read

    S = randfunc(N>>3)
    odd_bits = N % 8
    if odd_bits != 0:
        char = ord(randfunc(1)) >> (8-odd_bits)
        S = bchr(char) + S
    value = bytes_to_long(S)
    return value


def get_random_range(a, b, randfunc=None):
    """getRandomRange(a:int, b:int, randfunc:callable):long
    Return a random number n so that a <= n < b.

    If randfunc is omitted, then Random.new().read is used.

    This function is for internal use only and may be renamed or removed in
    the future.
    """
    range_ = b - a - 1
    bits = size(range_)
    value = get_random_integer(bits, randfunc)
    while value > range_:
        value = get_random_integer(bits, randfunc)
    return a + value


#
# Padding
#


class Padding(object):
    """
    Basic Padding methods
    """
    @staticmethod
    def pad(data, *args, **kwargs):  # pragma: no cover
        """Pads data with given padding.

        :returns: Padded data.
        :rtype: list

        """
        raise NotImplementedError()

    @staticmethod
    def unpad(data, *args, **kwargs):  # pragma: no cover
        """UnPads data with given padding.

        :returns: unpaded data
        :rtype: list

        """
        raise NotImplementedError()


class EmptyPadding(Padding):
    @staticmethod
    def unpad(data, *args, **kwargs):
        return data

    @staticmethod
    def pad(data, *args, **kwargs):
        return data


class PKCS7(Padding):
    @staticmethod
    def unpad(data, *args, **kwargs):
        return data[:-ord(data[len(data)-1:])]

    @staticmethod
    def pad(data, *args, **kwargs):
        bs = kwargs.get('bs', 16)
        return data + (bs - len(data) % bs) * chr(bs - len(data) % bs)


class PKCS15(Padding):
    @staticmethod
    def unpad(data, *args, **kwargs):
        bs = kwargs.get('bs', 256 if len(args) == 0 else args[0])
        bt = kwargs.get('bt', 2 if len(args) <= 1 else args[1])

        prefix = b("\x00") + bchr(bt)
        if data[0:2] != prefix:
            raise ValueError('Padding error')

        # Not needed in the client
        raise NotImplementedError()

    @staticmethod
    def pad(data, *args, **kwargs):
        bs = kwargs.get('bs', 256 if len(args) == 0 else args[0])
        bt = kwargs.get('bt', 2 if len(args) <= 1 else args[1])

        data = to_bytes(data)
        blb = len(data)
        if blb+3 > bs:
            raise ValueError('Input data too long')

        ps_len = bs - 3 - blb
        padding_str = bchr(0x00) # tmp
        if bt == 0:
            padding_str = bchr(0x00) * ps_len
        elif bt == 1:
            padding_str = bchr(0xFF) * ps_len
        elif bt == 2:
            arr = [int(get_random_range(1, 0x100)) for _ in range(ps_len)]
            padding_str = to_bytes(arr)
        else:
            raise ValueError('Unknown padding type')

        return b("\x00") + bchr(bt) + padding_str + b("\x00") + data


#
# Encryption
#

def aes_cbc(key):
    """
    Returns AES-CBC instance that can be used for [incremental] encryption/decryption in ProcessData.
    Uses zero IV.

    :param key:
    :return:
    """
    return AES.new(key, AES.MODE_CBC, get_zero_vector(16))


def aes(encrypt, key, data):
    """
    One-pass AES-256-CBC used in ProcessData. Zero IV (don't panic, IV-like random nonce is included in plaintext in the
    first block in ProcessData).

    Does not use padding (data has to be already padded).

    :param encrypt:
    :param key:
    :param data:
    :return:
    """
    cipher = AES.new(key, AES.MODE_CBC, get_zero_vector(16))
    if encrypt:
        return cipher.encrypt(data)
    else:
        return cipher.decrypt(data)


def aes_enc(key, data):
    return aes(True, key, data)


def aes_dec(key, data):
    return aes(False, key, data)


def cbc_mac(key, data):
    """
    AES-265-CBC-MAC on the data used in ProcessData.
    Does not use padding (data has to be already padded).

    :param key:
    :param data:
    :return:
    """
    engine = AES.new(key, AES.MODE_CBC, get_zero_vector(16))
    return engine.encrypt(data)[-16:]


def rsa_enc(data, modulus, exponent):
    """
    Simple RAW RSA encryption method, returns byte string.
    Returns byte string of the same size as the modulus (left padded with 0)

    :param data:
    :param modulus:
    :param exponent:
    :return:
    """
    modulus = to_long(modulus)
    exponent = to_long(exponent)
    data = to_long(data)

    return long_to_bytes(pow(data, exponent, modulus), long_byte_size(modulus))

