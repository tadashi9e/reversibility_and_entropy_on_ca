# -*- mode:python;coding:utf-8 -*-
import argparse
from enum import IntEnum
import random
import sys
import cupy as cp
import matplotlib
import matplotlib.pyplot as plt

BIT_E = cp.uint8(1 << 0)
BIT_W = cp.uint8(1 << 1)
BIT_BW = cp.uint8(1 << 2)
BITS_EW = BIT_E | BIT_W

def rotate_and_flip(field : cp.ndarray) -> cp.ndarray:
    forward = field & BITS_EW
    back = (
        ((field & BIT_E) != 0).astype(cp.uint8) * BIT_W |
        ((field & BIT_W) != 0).astype(cp.uint8) * BIT_E)
    rotated = cp.where((field & BIT_BW) == 0,
                       forward,
                       back)
    flip = ((field & BITS_EW) != 0).astype(cp.uint8) * BIT_BW
    flipped = (field & BIT_BW) ^ flip
    return rotated | flipped
def rev_rotate_and_flip(field : cp.ndarray) -> cp.ndarray:
    forward = field & BITS_EW
    back = (
        ((field & BIT_E) != 0).astype(cp.uint8) * BIT_W |
        ((field & BIT_W) != 0).astype(cp.uint8) * BIT_E)
    rotated = cp.where((field & BIT_BW) != 0,
                       forward,
                       back)
    flip = ((field & BITS_EW) != 0).astype(cp.uint8) * BIT_BW
    flipped = (field & BIT_BW) ^ flip
    return rotated | flipped
def forward(field : cp.ndarray) -> cp.ndarray:
    return (
        ((cp.roll(field, -1, axis = 0) & BIT_E) != 0).astype(cp.uint8) * BIT_E |
        ((cp.roll(field, 1, axis = 0) & BIT_W) != 0).astype(cp.uint8) * BIT_W |
        cp.bitwise_and(field, BIT_BW))
def rev_forward(field : cp.ndarray) -> cp.ndarray:
    return (
        ((cp.roll(field, 1, axis = 0) & BIT_E) != 0).astype(cp.uint8) * BIT_E |
        ((cp.roll(field, -1, axis = 0) & BIT_W) != 0).astype(cp.uint8) * BIT_W |
        cp.bitwise_and(field, BIT_BW))

def update_field(field : cp.ndarray) -> None:
    field2 = forward(rotate_and_flip(field))
    cp.copyto(field, field2, casting = 'safe')
def rev_update_field(field : cp.ndarray) -> None:
    field2 = rev_rotate_and_flip(rev_forward(field))
    cp.copyto(field, field2, casting = 'safe')

def calc_entropy_bw(field : cp.ndarray) -> float:
    ncells = field.size
    c_b = cp.sum((field & BIT_BW) != 0)
    p_b = c_b / ncells
    p_w = 1.0 - p_b
    entropy = -(cp.where(p_b > 0, p_b * cp.log2(p_b), 0) +
                cp.where(p_w > 0, p_w * cp.log2(p_w), 0))
    return float(entropy)

def update(count : int, reverse_count : int,
           field : cp.ndarray,
           result : cp.ndarray) -> float:
    if reverse_count >= 0 and count < reverse_count:
        update_field(field)
    else:
        rev_update_field(field)
    entropy_bw = calc_entropy_bw(field)
    #s = ''.join([str(n) for n in field])
    #print(f'{s} {entropy_bw}')
    result[count, :] = ((field & BIT_BW) != 0).astype(cp.uint8)
    return entropy_bw

def simulate(field : cp.ndarray,
             max_count : int,
             reverse_count : int) -> tuple[cp.ndarray, list[float]]:
    result = cp.zeros((max_count, field.size), dtype = int)
    result_entropy : list[float] = []
    for count in range(max_count):
        if count % 10 == 0:
            sys.stdout.write(f'{count} / {max_count}\r')
        entropy = update(count, reverse_count, field, result)
        result_entropy.append(entropy)
    print(f'{max_count}')
    return result, result_entropy

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--width",
                    type = int,
                    default = 128,
                    help = "Space width (default 128)")
parser.add_argument("-c", "--count",
                    type = int,
                    default = 3000,
                    help = "Max step count (default 3000)")
parser.add_argument("-r", "--reverse",
                    type = int,
                    default = -1,
                    help = "Reversing start count (default -1: NO REVERSE)")
parser.add_argument("-s", "--seed",
                    type = int,
                    default = 8,
                    help = "Random seed")
parser.add_argument("-n", "--n_ants",
                    type = int,
                    default = 4,
                    help = "N ants (default 4)")
args = parser.parse_args()
random.seed(args.seed)

field = cp.zeros(args.width, dtype = cp.uint8)
for n in range(1, args.n_ants):
    x = random.randint(0, args.width - 1)
    while field[x] != 0:
        x = random.randint(0, args.width - 1)
    field[x] = 1 << random.randint(0, 1)
result, result_entropy = simulate(
    field, args.count, args.reverse)
fig, (ax1, ax2) = plt.subplots(
    2,1, gridspec_kw={'height_ratios': [5, 1]},
    tight_layout = True)
ax1.set_ylabel('Space')
ax1.imshow(result.swapaxes(0,1).get(), cmap = 'binary',
           interpolation = 'nearest', aspect = 'auto')
ax2.set_ylabel('Entropy')
ax2.set_xlabel('Time')
ax2.set_xmargin(0)
ax2.plot(result_entropy)
plt.show()
print('done')
