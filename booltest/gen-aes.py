#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from crypto_util import aes_ctr, get_zero_vector


def main():
    parser = argparse.ArgumentParser(description='AES-CTR generator')

    parser.add_argument('--size', dest='size', default=16,
                        help='Number of bytes to generate')

    args = parser.parse_args()
    aes = aes_ctr(get_zero_vector(16))
    print(aes.encrypt(get_zero_vector(int(args.size))))


# Launcher
app = None
if __name__ == "__main__":
    main()


