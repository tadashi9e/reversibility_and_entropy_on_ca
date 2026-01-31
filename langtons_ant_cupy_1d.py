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

def calc_entropy_bw(field : cp.ndarray) -> float:
    ncells = field.size
    c_b = cp.sum((field & BIT_BW) != 0)
    p_b = c_b / ncells
    p_w = 1.0 - p_b
    entropy = -(cp.where(p_b > 0, p_b * cp.log2(p_b), 0) +
                cp.where(p_w > 0, p_w * cp.log2(p_w), 0))
    return float(entropy)

def rotate_and_flip(field : cp.ndarray) -> cp.ndarray:
    forward = field & BITS_EW
    back = (
        ((field & BIT_E) != 0).astype(cp.uint8) * BIT_W |
        ((field & BIT_W) != 0).astype(cp.uint8) * BIT_E)
    # when cell is white(0) -> go forward
    # when cell is black(1) -> go back
    rotated = cp.where((field & BIT_BW) == 0,
                       forward,
                       back)
    # flip black & white
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
    field2 = rotate_and_flip(rev_forward(field))
    cp.copyto(field, field2, casting = 'safe')

def update(count : int, max_count : int, reverse_count : int,
           field : cp.ndarray,
           result : cp.ndarray) -> tuple[bytes, float]:
    if count < reverse_count:
        update_field(field)
    else:
        rev_update_field(field)
    entropy_bw = calc_entropy_bw(field)
    #s = ''.join(['0' if (n & BIT_BW) == 0 else '1'
    #             for n in field])
    #print(f'{s} {entropy_bw}')
    sys.stdout.write(f'{count}\r')
    result[count, :] = ((field & BIT_BW) != 0).astype(cp.uint8)
    return cp.asnumpy(field).tobytes(), entropy_bw

def simulate(field : cp.ndarray,
             max_count : int,
             reverse_count : int) -> tuple[cp.ndarray, list[float], int, int]:
    loop_detection : set[bytes] = set()
    history : list[bytes] = []
    result = cp.zeros((max_count, field.size), dtype = int)
    result_entropy : list[float] = []
    for i in range(max_count):
        s, entropy = update(i, max_count, reverse_count, field, result)
        result_entropy.append(entropy)
        if s in loop_detection:
            return result, result_entropy, history.index(s), i
        loop_detection.add(s)
        history.append(s)
    return result, result_entropy, -1, max_count

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--width",
                    type = int,
                    default = 100,
                    help = "Space width (default 100)")
parser.add_argument("-c", "--count",
                    type = int,
                    default = 30000,
                    help = "Max step count (default 30000)")
parser.add_argument("-r", "--reverse",
                    type = int,
                    default = 1000,
                    help = "Reversing start count (default 1000)")
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
    field[x] = 1 << random.randint(0, 2)
result, result_entropy, loop_start, n = simulate(
    field, args.count, args.reverse)
fig, (ax1, ax2) = plt.subplots(
    2,1, gridspec_kw={'height_ratios': [5, 1]})
ax1.set_ylabel('Space')
ax1.set_xlabel('Time')
ax1.imshow(result[:n].swapaxes(0,1).get(), cmap = 'binary',
           interpolation = 'nearest', aspect = 'auto')
ax2.set_ylabel('Entropy')
ax2.set_xlabel('Time')
ax2.plot(result_entropy)
ax2.axvline(loop_start, color = 'r')
plt.show()
print('done')
