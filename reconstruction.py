#!/usr/bin/env python3


import bitarray
import numpy as np
import os
import sys
import collections
import itertools
from os import stat
from bidict import bidict
import numpy as np
from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from numpy import ones
from math import gcd
from functools import reduce
from numpy import ones,zeros, pi, cos, exp, sign
from scipy import integrate
from  scipy.io.wavfile import read as wavread
from scipy import signal
from scipy.fftpack import dct, idct
import queue as Queue
import time
import base64
from PIL import Image
import glob

import collections
import itertools

from bidict import bidict
import numpy as np
import collections
import itertools
def GIF_save(path, framerate):
    os.system("ffmpeg -r {:d} -i {:s}frame_%2d.tiff -compression_level 0 -plays 0 -f apng {:s}animation.png".format(framerate, path,path))


EOB = (0, 0)
ZRL = (15, 0)

def encode_huffman(value, layer_type):
    """Encode the Huffman coding of value.
    Arguments:
        value {int or tuple} -- "dc" (int) or run-length "ac" (tuple).
        layer_type {"y" or "c"} -- Specify the layer type of
            value.
    Raises:
        ValueError -- When the value is out of the range.
    Returns:
        str -- Huffman encoded bit array.
    """

    def index_2d(table, target):
        for i, row in enumerate(table):
            for j, element in enumerate(row):
                if target == element:
                    return (i, j)
        raise ValueError('Cannot find the target value in the table.')

    if not isinstance(value, collections.Iterable):  # "dc"
        if value <= -2048 or value >= 2048:
            raise ValueError(
                '"dc" should be within [-2047, 2047].'
            )

        size, fixed_code_idx = index_2d(HUFFMAN_CATEGORIES, value)

        if size == 0:
            return HUFFMAN_CATEGORY_CODEWORD["dc"][layer_type][size]
        return (HUFFMAN_CATEGORY_CODEWORD["dc"][layer_type][size]
                + '{:0{padding}b}'.format(fixed_code_idx, padding=size))
    else:   # "ac"
        value = tuple(value)
        if value == EOB or value == ZRL:
            return HUFFMAN_CATEGORY_CODEWORD["ac"][layer_type][value]

        run, nonzero = value
        if nonzero == 0 or nonzero <= -1024 or nonzero >= 1024:
            raise ValueError(
                '"ac" coefficient nonzero should be within [-1023, 0) '
                'or (0, 1023].'
            )

        size, fixed_code_idx = index_2d(HUFFMAN_CATEGORIES, nonzero)
        return (HUFFMAN_CATEGORY_CODEWORD["ac"][layer_type][(run, size)]
                + '{:0{padding}b}'.format(fixed_code_idx, padding=size))


def decode_huffman(bit_seq, dc_ac, layer_type):
    """Decode a bit sequence encoded by JPEG baseline Huffman table.
    Arguments:
        bit_seq {str} -- The encoded bit sequence.
        dc_ac {"dc" or "ac"} -- The type of coefficient.
        layer_type {"y" or "c"} -- The layer type of bit sequence.
    Raises:
        IndexError -- When there is not enough bits in bit sequence to decode
            DIFF value codeword.
        KeyError -- When not able to find any prefix in current slice of bit
            sequence in Huffman table.
    Returns:
        Generator -- A generator and its item is decoded value which could be an
            integer (differential "dc") or a tuple (run-length-encoded "ac").
    """

    def diff_value(idx, size):
        if idx >= len(bit_seq) or idx + size > len(bit_seq):
            raise IndexError('There is not enough bits to decode DIFF value '
                             'codeword.')
        fixed = bit_seq[idx:idx + size]
        return int(fixed, 2)

    current_idx = 0
    while current_idx < len(bit_seq):
        #   1. Consume next 16 bits as `current_slice`.
        #   2. Try to find the `current_slice` in Huffman table.
        #   3. If found, yield the corresponding key and go to step 4.
        #      Otherwise, remove the last element in `current_slice` and go to
        #      step 2.
        #   4. Consume next n bits, where n is the category (size) in returned
        #      key yielded in step 3. Use those info to decode the data.
        remaining_len = len(bit_seq) - current_idx
        current_slice = bit_seq[
            current_idx:
            current_idx + (16 if remaining_len > 16 else remaining_len)
        ]
        err_cache = current_slice
        while current_slice:
            if (current_slice in
                    HUFFMAN_CATEGORY_CODEWORD[dc_ac][layer_type].inv):
                key = (HUFFMAN_CATEGORY_CODEWORD[dc_ac][layer_type]
                       .inv[current_slice])
                if dc_ac == "dc":  # "dc"
                    size = key
                    if size == 0:
                        yield 0
                    else:
                        yield HUFFMAN_CATEGORIES[size][diff_value(
                            current_idx + len(current_slice),
                            size
                        )]
                else:  # "ac"
                    run, size = key
                    if key == EOB or key == ZRL:
                        yield key
                    else:
                        yield (run, HUFFMAN_CATEGORIES[size][diff_value(
                            current_idx + len(current_slice),
                            size
                        )])

                current_idx += len(current_slice) + size
                break
            else:
                current_slice = current_slice[:-1]
        else:
            print(current_slice)
            raise KeyError(
                current_slice,'Cannot find prefix in Huffman table.'
            )

HUFFMAN_CATEGORIES = (
    (0, ),
    (-1, 1),
    (-3, -2, 2, 3),
    (*range(-7, -4 + 1), *range(4, 7 + 1)),
    (*range(-15, -8 + 1), *range(8, 15 + 1)),
    (*range(-31, -16 + 1), *range(16, 31 + 1)),
    (*range(-63, -32 + 1), *range(32, 63 + 1)),
    (*range(-127, -64 + 1), *range(64, 127 + 1)),
    (*range(-255, -128 + 1), *range(128, 255 + 1)),
    (*range(-511, -256 + 1), *range(256, 511 + 1)),
    (*range(-1023, -512 + 1), *range(512, 1023 + 1)),
    (*range(-2047, -1024 + 1), *range(1024, 2047 + 1)),
    (*range(-4095, -2048 + 1), *range(2048, 4095 + 1)),
    (*range(-8191, -4096 + 1), *range(4096, 8191 + 1)),
    (*range(-16383, -8192 + 1), *range(8192, 16383 + 1)),
    (*range(-32767, -16384 + 1), *range(16384, 32767 + 1))
)

HUFFMAN_CATEGORY_CODEWORD = {
    "dc": {
        "y": bidict({
            0:  '00',
            1:  '010',
            2:  '011',
            3:  '100',
            4:  '101',
            5:  '110',
            6:  '1110',
            7:  '11110',
            8:  '111110',
            9:  '1111110',
            10: '11111110',
            11: '111111110'
        }),
        "c": bidict({
            0:  '00',
            1:  '01',
            2:  '10',
            3:  '110',
            4:  '1110',
            5:  '11110',
            6:  '111110',
            7:  '1111110',
            8:  '11111110',
            9:  '111111110',
            10: '1111111110',
            11: '11111111110'
        })
    },
    "ac": {
        "y": bidict({
            EOB: '1010',  # (0, 0)
            ZRL: '11111111001',  # (F, 0)

            (0, 1):  '00',
            (0, 2):  '01',
            (0, 3):  '100',
            (0, 4):  '1011',
            (0, 5):  '11010',
            (0, 6):  '1111000',
            (0, 7):  '11111000',
            (0, 8):  '1111110110',
            (0, 9):  '1111111110000010',
            (0, 10): '1111111110000011',

            (1, 1):  '1100',
            (1, 2):  '11011',
            (1, 3):  '1111001',
            (1, 4):  '111110110',
            (1, 5):  '11111110110',
            (1, 6):  '1111111110000100',
            (1, 7):  '1111111110000101',
            (1, 8):  '1111111110000110',
            (1, 9):  '1111111110000111',
            (1, 10): '1111111110001000',

            (2, 1):  '11100',
            (2, 2):  '11111001',
            (2, 3):  '1111110111',
            (2, 4):  '111111110100',
            (2, 5):  '1111111110001001',
            (2, 6):  '1111111110001010',
            (2, 7):  '1111111110001011',
            (2, 8):  '1111111110001100',
            (2, 9):  '1111111110001101',
            (2, 10): '1111111110001110',

            (3, 1):  '111010',
            (3, 2):  '111110111',
            (3, 3):  '111111110101',
            (3, 4):  '1111111110001111',
            (3, 5):  '1111111110010000',
            (3, 6):  '1111111110010001',
            (3, 7):  '1111111110010010',
            (3, 8):  '1111111110010011',
            (3, 9):  '1111111110010100',
            (3, 10): '1111111110010101',

            (4, 1):  '111011',
            (4, 2):  '1111111000',
            (4, 3):  '1111111110010110',
            (4, 4):  '1111111110010111',
            (4, 5):  '1111111110011000',
            (4, 6):  '1111111110011001',
            (4, 7):  '1111111110011010',
            (4, 8):  '1111111110011011',
            (4, 9):  '1111111110011100',
            (4, 10): '1111111110011101',

            (5, 1):  '1111010',
            (5, 2):  '11111110111',
            (5, 3):  '1111111110011110',
            (5, 4):  '1111111110011111',
            (5, 5):  '1111111110100000',
            (5, 6):  '1111111110100001',
            (5, 7):  '1111111110100010',
            (5, 8):  '1111111110100011',
            (5, 9):  '1111111110100100',
            (5, 10): '1111111110100101',

            (6, 1):  '1111011',
            (6, 2):  '111111110110',
            (6, 3):  '1111111110100110',
            (6, 4):  '1111111110100111',
            (6, 5):  '1111111110101000',
            (6, 6):  '1111111110101001',
            (6, 7):  '1111111110101010',
            (6, 8):  '1111111110101011',
            (6, 9):  '1111111110101100',
            (6, 10): '1111111110101101',

            (7, 1):  '11111010',
            (7, 2):  '111111110111',
            (7, 3):  '1111111110101110',
            (7, 4):  '1111111110101111',
            (7, 5):  '1111111110110000',
            (7, 6):  '1111111110110001',
            (7, 7):  '1111111110110010',
            (7, 8):  '1111111110110011',
            (7, 9):  '1111111110110100',
            (7, 10): '1111111110110101',

            (8, 1):  '111111000',
            (8, 2):  '111111111000000',
            (8, 3):  '1111111110110110',
            (8, 4):  '1111111110110111',
            (8, 5):  '1111111110111000',
            (8, 6):  '1111111110111001',
            (8, 7):  '1111111110111010',
            (8, 8):  '1111111110111011',
            (8, 9):  '1111111110111100',
            (8, 10): '1111111110111101',

            (9, 1):  '111111001',
            (9, 2):  '1111111110111110',
            (9, 3):  '1111111110111111',
            (9, 4):  '1111111111000000',
            (9, 5):  '1111111111000001',
            (9, 6):  '1111111111000010',
            (9, 7):  '1111111111000011',
            (9, 8):  '1111111111000100',
            (9, 9):  '1111111111000101',
            (9, 10): '1111111111000110',
            # A
            (10, 1):  '111111010',
            (10, 2):  '1111111111000111',
            (10, 3):  '1111111111001000',
            (10, 4):  '1111111111001001',
            (10, 5):  '1111111111001010',
            (10, 6):  '1111111111001011',
            (10, 7):  '1111111111001100',
            (10, 8):  '1111111111001101',
            (10, 9):  '1111111111001110',
            (10, 10): '1111111111001111',
            # B
            (11, 1):  '1111111001',
            (11, 2):  '1111111111010000',
            (11, 3):  '1111111111010001',
            (11, 4):  '1111111111010010',
            (11, 5):  '1111111111010011',
            (11, 6):  '1111111111010100',
            (11, 7):  '1111111111010101',
            (11, 8):  '1111111111010110',
            (11, 9):  '1111111111010111',
            (11, 10): '1111111111011000',
            # C
            (12, 1):  '1111111010',
            (12, 2):  '1111111111011001',
            (12, 3):  '1111111111011010',
            (12, 4):  '1111111111011011',
            (12, 5):  '1111111111011100',
            (12, 6):  '1111111111011101',
            (12, 7):  '1111111111011110',
            (12, 8):  '1111111111011111',
            (12, 9):  '1111111111100000',
            (12, 10): '1111111111100001',
            # D
            (13, 1):  '11111111000',
            (13, 2):  '1111111111100010',
            (13, 3):  '1111111111100011',
            (13, 4):  '1111111111100100',
            (13, 5):  '1111111111100101',
            (13, 6):  '1111111111100110',
            (13, 7):  '1111111111100111',
            (13, 8):  '1111111111101000',
            (13, 9):  '1111111111101001',
            (13, 10): '1111111111101010',
            # E
            (14, 1):  '1111111111101011',
            (14, 2):  '1111111111101100',
            (14, 3):  '1111111111101101',
            (14, 4):  '1111111111101110',
            (14, 5):  '1111111111101111',
            (14, 6):  '1111111111110000',
            (14, 7):  '1111111111110001',
            (14, 8):  '1111111111110010',
            (14, 9):  '1111111111110011',
            (14, 10): '1111111111110100',
            # F
            (15, 1):  '1111111111110101',
            (15, 2):  '1111111111110110',
            (15, 3):  '1111111111110111',
            (15, 4):  '1111111111111000',
            (15, 5):  '1111111111111001',
            (15, 6):  '1111111111111010',
            (15, 7):  '1111111111111011',
            (15, 8):  '1111111111111100',
            (15, 9):  '1111111111111101',
            (15, 10): '1111111111111110'
        }),
        "c": bidict({
            EOB: '00',  # (0, 0)
            ZRL: '1111111010',  # (F, 0)

            (0, 1):  '01',
            (0, 2):  '100',
            (0, 3):  '1010',
            (0, 4):  '11000',
            (0, 5):  '11001',
            (0, 6):  '111000',
            (0, 7):  '1111000',
            (0, 8):  '111110100',
            (0, 9):  '1111110110',
            (0, 10): '111111110100',

            (1, 1):  '1011',
            (1, 2):  '111001',
            (1, 3):  '11110110',
            (1, 4):  '111110101',
            (1, 5):  '11111110110',
            (1, 6):  '111111110101',
            (1, 7):  '1111111110001000',
            (1, 8):  '1111111110001001',
            (1, 9):  '1111111110001010',
            (1, 10): '1111111110001011',

            (2, 1):  '11010',
            (2, 2):  '11110111',
            (2, 3):  '1111110111',
            (2, 4):  '111111110110',
            (2, 5):  '111111111000010',
            (2, 6):  '1111111110001100',
            (2, 7):  '1111111110001101',
            (2, 8):  '1111111110001110',
            (2, 9):  '1111111110001111',
            (2, 10): '1111111110010000',

            (3, 1):  '11011',
            (3, 2):  '11111000',
            (3, 3):  '1111111000',
            (3, 4):  '111111110111',
            (3, 5):  '1111111110010001',
            (3, 6):  '1111111110010010',
            (3, 7):  '1111111110010011',
            (3, 8):  '1111111110010100',
            (3, 9):  '1111111110010101',
            (3, 10): '1111111110010110',

            (4, 1):  '111010',
            (4, 2):  '111110110',
            (4, 3):  '1111111110010111',
            (4, 4):  '1111111110011000',
            (4, 5):  '1111111110011001',
            (4, 6):  '1111111110011010',
            (4, 7):  '1111111110011011',
            (4, 8):  '1111111110011100',
            (4, 9):  '1111111110011101',
            (4, 10): '1111111110011110',

            (5, 1):  '111011',
            (5, 2):  '1111111001',
            (5, 3):  '1111111110011111',
            (5, 4):  '1111111110100000',
            (5, 5):  '1111111110100001',
            (5, 6):  '1111111110100010',
            (5, 7):  '1111111110100011',
            (5, 8):  '1111111110100100',
            (5, 9):  '1111111110100101',
            (5, 10): '1111111110100110',

            (6, 1):  '1111001',
            (6, 2):  '11111110111',
            (6, 3):  '1111111110100111',
            (6, 4):  '1111111110101000',
            (6, 5):  '1111111110101001',
            (6, 6):  '1111111110101010',
            (6, 7):  '1111111110101011',
            (6, 8):  '1111111110101100',
            (6, 9):  '1111111110101101',
            (6, 10): '1111111110101110',

            (7, 1):  '1111010',
            (7, 2):  '111111110000',
            (7, 3):  '1111111110101111',
            (7, 4):  '1111111110110000',
            (7, 5):  '1111111110110001',
            (7, 6):  '1111111110110010',
            (7, 7):  '1111111110110011',
            (7, 8):  '1111111110110100',
            (7, 9):  '1111111110110101',
            (7, 10): '1111111110110110',

            (8, 1):  '11111001',
            (8, 2):  '1111111110110111',
            (8, 3):  '1111111110111000',
            (8, 4):  '1111111110111001',
            (8, 5):  '1111111110111010',
            (8, 6):  '1111111110111011',
            (8, 7):  '1111111110111100',
            (8, 8):  '1111111110111101',
            (8, 9):  '1111111110111110',
            (8, 10): '1111111110111111',

            (9, 1):  '111110111',
            (9, 2):  '1111111111000000',
            (9, 3):  '1111111111000001',
            (9, 4):  '1111111111000010',
            (9, 5):  '1111111111000011',
            (9, 6):  '1111111111000100',
            (9, 7):  '1111111111000101',
            (9, 8):  '1111111111000110',
            (9, 9):  '1111111111000111',
            (9, 10): '1111111111001000',
            # A
            (10, 1):  '111111000',
            (10, 2):  '1111111111001001',
            (10, 3):  '1111111111001010',
            (10, 4):  '1111111111001011',
            (10, 5):  '1111111111001100',
            (10, 6):  '1111111111001101',
            (10, 7):  '1111111111001110',
            (10, 8):  '1111111111001111',
            (10, 9):  '1111111111010000',
            (10, 10): '1111111111010001',
            # B
            (11, 1):  '111111001',
            (11, 2):  '1111111111010010',
            (11, 3):  '1111111111010011',
            (11, 4):  '1111111111010100',
            (11, 5):  '1111111111010101',
            (11, 6):  '1111111111010110',
            (11, 7):  '1111111111010111',
            (11, 8):  '1111111111011000',
            (11, 9):  '1111111111011001',
            (11, 10): '1111111111011010',
            # C
            (12, 1):  '111111010',
            (12, 2):  '1111111111011011',
            (12, 3):  '1111111111011100',
            (12, 4):  '1111111111011101',
            (12, 5):  '1111111111011110',
            (12, 6):  '1111111111011111',
            (12, 7):  '1111111111100000',
            (12, 8):  '1111111111100001',
            (12, 9):  '1111111111100010',
            (12, 10): '1111111111100011',
            # D
            (13, 1):  '11111111001',
            (13, 2):  '1111111111100100',
            (13, 3):  '1111111111100101',
            (13, 4):  '1111111111100110',
            (13, 5):  '1111111111100111',
            (13, 6):  '1111111111101000',
            (13, 7):  '1111111111101001',
            (13, 8):  '1111111111101010',
            (13, 9):  '1111111111101011',
            (13, 10): '1111111111101100',
            # E
            (14, 1):  '11111111100000',
            (14, 2):  '1111111111101101',
            (14, 3):  '1111111111101110',
            (14, 4):  '1111111111101111',
            (14, 5):  '1111111111110000',
            (14, 6):  '1111111111110001',
            (14, 7):  '1111111111110010',
            (14, 8):  '1111111111110011',
            (14, 9):  '1111111111110100',
            (14, 10): '1111111111110101',
            # F
            (15, 1):  '111111111000011',
            (15, 2):  '1111111111110110',
            (15, 3):  '1111111111110111',
            (15, 4):  '1111111111111000',
            (15, 5):  '1111111111111001',
            (15, 6):  '1111111111111010',
            (15, 7):  '1111111111111011',
            (15, 8):  '1111111111111100',
            (15, 9):  '1111111111111101',
            (15, 10): '1111111111111110'
        })
    }
}









def reconstruct(binfile, outfile):
    # Read in data and settings
    uid = int(binfile.split('.')[0])
    SOI = 0
    recon = []
        
    with open(binfile, 'rb') as fh:
        SOI = fh.read(2)

        ## For you to modify
        print(outfile)
        print(SOI)
        if SOI != bytes.fromhex("FFD1"):
            raise Exception("Start of File marker not found!")
        M = int.from_bytes(fh.read(2), "big")
        N = int.from_bytes(fh.read(2), "big")
        recon = []#np.zeros(2,M,N,3)
        temp = []
        quality = int.from_bytes(fh.read(2), "big")
        rate = int.from_bytes(fh.read(2), "big")
        SOI = fh.read(2)
        count = 0
        up = 4
        while SOI != bytes.fromhex("FFD2"):
            if SOI != bytes.fromhex("FFD8"):
                raise Exception("Start of Image marker not found!")
            bits = ()
            L = 0
            if SOI != bytes.fromhex("FFDA"):
                SOI = fh.read(2)
            for _ in range(5):
                ba = bitarray.bitarray()
                for b in iter(lambda: fh.read(2), bytes.fromhex("FFDA")):
                    #print(b)
                    ba.frombytes(b)
                    #if L <= 0:
                    #    print(L,'\n\nreading bits\n:',ba,'\n')
                bits = (*bits, ba)
                #print('run:',L,'inside loop\n\n\n',bits)
                L += 1
            ba = bitarray.bitarray()
            for b in iter(lambda: fh.read(2), bytes.fromhex("FFD9")):
                ba.frombytes(b)
            bits = (*bits, ba)
            #print('lenth of bits is :',len(bits))
            #print('in run:',count,'the bits are:\n',bits,'\n\n\n')
            #for i in range(len(bits)):
            #    print(i,' run:  ',bits[i],'\n\n\n\n\n')
            x = decode_image(bits, M, N, quality)
            temp.append(x)
            #x = Image.fromarray(x.astype(np.uint8))
            count += 1
#             if count%2 == 0:
#                 h = (x+temp[count-2])/2
#                 h = signal.resample_poly(h,up,1,axis=0,padtype='mean')
#                 h = signal.resample_poly(h,up,1,axis=1,padtype='mean')
#                 recon.append(Image.fromarray(h.astype(np.uint8)))
            x = signal.resample_poly(x,up,1,axis=0,padtype='mean')
            x = signal.resample_poly(x,up,1,axis=1,padtype='mean')
            recon.append(Image.fromarray(x.astype(np.uint8))) #Image.fromarray(np_im)
            SOI = fh.read(2)
    #print('\n:second:',recon.shape)
    
    # signal.resample_poly(Y, up, 1, padtype='mean')
    for i in range(len(recon)):
        im_ = recon[i]
        rec_fname = "frame_{:02d}.tiff".format(i+1)
        im_.save(rec_fname,save_all=True)
    #recon[0].save(outfile,save_all=True, append_images=recon[1:])
    GIF_save(path = '', framerate = 6)
    
    
    
    ##########3
    with open(binfile, 'rb') as f:
        B = fh.read()

    # Check length
    if len(B) == 0:
        print("Empty data!")
        msg = "empty data component"
        ffail = str(uid) + '.fail'
        with open(ffail, 'w') as f:
            f.write(msg)
        return ffail
    if (len(B) - 4) != np.frombuffer(B[:4], dtype='<u4')[0]:
        print("Length of received bytes ({}) != length header ({})".format(
            len(B) - 4, np.frombuffer(B[:4], dtype='<u4')[0]))
        msg = "length mismatch"
        ffail = str(uid) + '.fail'
        with open(ffail, 'w') as f:
            f.write(msg)
        return ffail
    data_out = B[4:]  # Drop uint containing bit length

    ## Save as the given filename
    # Write output
    with open(outfile+'1', 'wb') as f:
        # Drop initial uint containing bit length
        f.write(data_out)
    return outfile

def YCbCr2RGB(im_ycbcr):
    # Input:  a 3D float array, im_ycbcr, representing a YCbCr image in range [-128.0,127.0]
    # Output: a 3D float array, im_rgb, representing an RGB image in range [0.0,255.0]
    
    # Your code here
    im_rgb = np.empty(im_ycbcr.shape, dtype=np.float64)
    im_rgb[:,:,0] = im_ycbcr[:,:,0] + 128. + 1.402*im_ycbcr[:,:,2]
    im_rgb[:,:,1] = im_ycbcr[:,:,0] + 128. - 0.344136*im_ycbcr[:,:,1] - 0.714136*im_ycbcr[:,:,2]
    im_rgb[:,:,2] = im_ycbcr[:,:,0] + 128. + 1.772*im_ycbcr[:,:,1]
    
    im_rgb[im_rgb<0.] = 0.
    im_rgb[im_rgb>255.] = 255.
    
    return im_rgb
        
def chroma_upsample(C2):
    # Input:  an (M/2)x(N/2) array, C2, of downsampled chroma values
    # Output: an MxN array, C, of chroma values
    
    # Your code here
    M, N = C2.shape
    C = np.array(Image.fromarray(C2).resize((N*2,M*2), resample=Image.BILINEAR))
    
    return C
        
def idct2(block_c):
    # Input:  a 2D array, block_c, of DCT coefficients
    # Output: a 2D array, block, representing an image block
    
    # Your code here
    block = idct(block_c, type=2, norm='ortho', axis=1)
    block = idct(block, type=2, norm='ortho', axis=0)
    
    return block
        
        
def unquantize(block_cq, mode="y", quality=75):
    # Input:  a 2D int array, block_cq, of quantized DCT coefficients
    #         a string, mode, ("y" for luma quantization, "c" for chroma quantization)
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: a 2D float array, block_c, of "unquantized" DCT coefficients (they will still be quantized)
    
    if mode is "y":
        Q = np.array([[ 16,  11,  10,  16,  24,  40,  51,  61 ],
                      [ 12,  12,  14,  19,  26,  58,  60,  55 ],
                      [ 14,  13,  16,  24,  40,  57,  69,  56 ],
                      [ 14,  17,  22,  29,  51,  87,  80,  62 ],
                      [ 18,  22,  37,  56,  68,  109, 103, 77 ],
                      [ 24,  36,  55,  64,  81,  104, 113, 92 ],
                      [ 49,  64,  78,  87,  103, 121, 120, 101],
                      [ 72,  92,  95,  98,  112, 100, 103, 99 ]])
    elif mode is "c":
        Q = np.array([[ 17,  18,  24,  47,  99,  99,  99,  99 ],
                      [ 18,  21,  26,  66,  99,  99,  99,  99 ],
                      [ 24,  26,  56,  99,  99,  99,  99,  99 ],
                      [ 47,  66,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ],
                      [ 99,  99,  99,  99,  99,  99,  99,  99 ]]) 
    else:
        raise Exception("String argument must be 'y' or 'c'.")
            
    if quality < 1 or quality > 100:
        raise Exception("Quality factor must be in range [1,100].")
        
    scalar = 5000 / quality if quality < 50 else 200 - 2 * quality # formula for scaling by quality factor
    Q = Q * scalar / 100. # scale the quantization matrix
    Q[Q<1.] = 1. # do not divide by numbers less than 1
    
    # Your code here
    block_c = (block_cq * Q).astype(float)
    
    return block_c 
        
def unzigzag(block_cqz):
    # Input:  a list, block_cqz, of zig-zag reordered quantized DCT coefficients
    # Output: a 2D array, block_cq, of conventionally ordered quantized DCT coefficients
    
    idx = [0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 
           43, 9, 11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38,
           46, 51, 55, 60, 21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63]
    
    # Your code here
    block_cq = np.array(block_cqz)[idx].reshape((8,8))
    
    return block_cq        

def unzrle(block_cqzr):
    # Input:  a list, block_cqzr, of zero-run-length encoded quantized DCT coefficients
    # Output: a list, block_cqz, of zig-zag reordered quantized DCT coefficients
    
    # Your code here
    block_cqz = [block_cqzr[0]] # initialize list with DC value
    for ind in range(1, len(block_cqzr)-1):
        run, val = block_cqzr[ind] # unpack (run length, nonzero value) tuple
        zer = [0] * run # list of zeros
        block_cqz.extend(zer) # extend with list of zeros
        block_cqz.append(val) # append nonzero value
    zer = [0] * (64-len(block_cqz)) # list of zeros to pad at end
    block_cqz.extend(zer)
    
    return block_cqz  
def decode_block(dc_gen, ac_gen, mode="y", quality=75):
    # Inputs: a generator, dc_gen, that yields decoded Huffman DC coefficients
    #         a generator, ac_gen, that yields decoded Huffman AC coefficients
    #         a string, mode, ("y" for luma, "c" for chroma)
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: a 2D array, block, decoded by and yielded from the two generators
    
    block_cqzr = [next(dc_gen)] # initialize list by yielding from DC generator
    while block_cqzr[-1] != (0,0):
        block_cqzr.append(next(ac_gen)) # append to list by yielding from AC generator until (0,0) is encountered
    block_cqz = unzrle(block_cqzr)
    block_cq = unzigzag(block_cqz)
    block_c = unquantize(block_cq, mode, quality)
    block = idct2(block_c)
    
    return block
def decode_image(bits, M, N, quality=75):
    # Inputs: a tuple, bits, containing the following:
    #              a bitarray, Y_dc_bits, the Y component DC bitstream
    #              a bitarray, Y_ac_bits, the Y component AC bitstream
    #              a bitarray, Cb_dc_bits, the Cb component DC bitstream
    #              a bitarray, Cb_ac_bits, the Cb component AC bitstream
    #              a bitarray, Cr_dc_bits, the Cr component DC bitstream
    #              a bitarray, Cr_ac_bits, the Cr component AC bitstream
    #         ints, M and N, the number of rows and columns in the image
    #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
    # Output: a 3D float array, img, representing an RGB image in range [0.0,255.0]
    
    Y_dc_bits, Y_ac_bits, Cb_dc_bits, Cb_ac_bits, Cr_dc_bits, Cr_ac_bits = bits # unpack bits tuple
    M_orig = M # save original image dimensions
    N_orig = N
    M = M_orig + ((16 - (M_orig % 16)) % 16) # dimensions of padded image
    N = N_orig + ((16 - (N_orig % 16)) % 16)
    num_blocks = M * N // 64 # number of blocks
    
    # Y component
    Y_dc_gen = decode_huffman(Y_dc_bits.to01(), "dc", "y")
    Y_ac_gen = decode_huffman(Y_ac_bits.to01(), "ac", "y")
    Y = np.empty((M, N))
    for b in range(num_blocks):
        #print('this is run:', b)
        #print('\n\n\n\n:',Y_dc_bits,'\n\n\n\n')
        block = decode_block(Y_dc_gen, Y_ac_gen, "y", quality)
        r = (b*8 // N)*8 # row index (top left corner)
        c = b*8 % N # column index (top left corner)
        Y[r:r+8, c:c+8] = block

    # Cb component
    Cb_dc_gen = decode_huffman(Cb_dc_bits.to01(), "dc", "c")
    Cb_ac_gen = decode_huffman(Cb_ac_bits.to01(), "ac", "c")
    Cb2 = np.empty((M//2, N//2))
    for b in range(num_blocks//4):
        block = decode_block(Cb_dc_gen, Cb_ac_gen, "c", quality)
        r = (b*8 // (N//2))*8 # row index (top left corner)
        c = b*8 % (N//2) # column index (top left corner)
        Cb2[r:r+8, c:c+8] = block

    # Cr component
    Cr_dc_gen = decode_huffman(Cr_dc_bits.to01(), "dc", "c")
    Cr_ac_gen = decode_huffman(Cr_ac_bits.to01(), "ac", "c")
    Cr2 = np.empty((M//2, N//2))
    for b in range(num_blocks//4):
        block = decode_block(Cr_dc_gen, Cr_ac_gen, "c", quality)
        r = (b*8 // (N//2))*8 # row index (top left corner)
        c = b*8 % (N//2) # column index (top left corner)
        Cr2[r:r+8, c:c+8] = block

    Cb = chroma_upsample(Cb2)
    Cr = chroma_upsample(Cr2)

    img = YCbCr2RGB(np.stack((Y,Cb,Cr), axis=-1))
    
    img = img[0:M_orig,0:N_orig,:] # crop out padded parts

    return img
def ee123_bitarr_to_matrix(bits: bitarray.bitarray):
    bN = np.uint32(len(bits)).tobytes()
    Bytes = bits.tobytes()
    base64_ = base64.b64encode(Bytes)
    result = base64.b64decode(base64_)
    return result

def main():
    datadir = sys.argv[1]
    os.chdir(datadir)  # cd into data dir
    binfile = sys.argv[2]
    outfile = sys.argv[3]
    fname = reconstruct(binfile, outfile)
    print("Reconstructed", fname)
    return 0


if __name__ == "__main__":
    main()

