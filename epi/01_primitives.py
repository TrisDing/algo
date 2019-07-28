import sys
import math
import random
import collections

"""
Bit-wise operators
------------------
6 & 4       # 0110 & 0100 = 0100 (4)                     AND
1 | 2       # 0001 | 0010 = 0011 (3)                     OR
15 ^ 1      # 00001111 ^ 00000001 = 00001110 (14)        XOR
8 >> 1      # 00001000 >> 1 = 00000100 (4)               x >> y = x // 2^y
1 << 10     # 000000000001 << 10 = 010000000000 (1024)   x << y = x *  2^y
-16 >> 2    # 11110000 >> 2 = 11111100 (-4)              negative right shifting
-16 << 2    # 11110000 << 2 = 11000000 (-64)             negative left shifting
~0          # ~0000 = 1111 (-1)                          ~x = -x - 1

Bit Operation Tricks
--------------------
x & 1             # Extract the last bit
(x >> k) & 1      # Extract the Kth bit
x |= 1            # Set the last bit
x |= (1 << k)     # Set the Kth bit
x ^= 1            # Flip the last bit
x ^= (1 << k)     # Flip the Kth bit
x & x - 1         # Drop the lowest set bit of x
x & ~(x - 1)      # Extract the lowest set bit of x
"""

# debugging util
def fb(num):
    return "{0:b}".format(num)

""" 4.0 COUNT BITS

    Count the number of bits that are set to 1 in a nonnegative integer.
"""
def count_bits(x):
    # Time complexity: O(n), where n is the word size
    num_bits = 0
    while x:
        num_bits += x & 1
        x >>= 1
    return num_bits

""" 4.1 COMPUTING THE PARITY OF A WORD

    The parity of a binary word is 1 if number of 1s in the word is odd;
    otherwise, it is 0. For example, the parity of 1011 is 1, and the parity
    of 10001000 is 0.
"""
def parity(x):
    # Brute-force
    # Time complexity: O(n), where n is the word size
    num_bits = 0
    while x:
        num_bits += x & 1
        x >>= 1
    return num_bits % 2

def parity2(x):
    # Don't have to add them up, just XOR
    # Time complexity: O(n), where n is the word size
    result = 0
    while x:
        result ^= x & 1
        x >>= 1
    return result

def parity3(x):
    # Drop the lowest set bit of x
    # Time complexity: O(k), where k is the number of bit set to 1
    result = 0
    while x:
        result ^= 1
        x &= x - 1
    return result

def parity4(x):
    # For a very large 64-bit number, use 16-bit cache
    # Time complexity: O(n/L), where L is the width of the cache

     # 8 bit cache example: 00101001 <(00),(01),(10),(11)>
    PRECOMPUTED_PARITY = {
        '00': 0,
        '10': 1,
        '01': 1,
        '11': 0,
    }
    MASK_SIZE = 16
    BIT_MASK = 0xFFFF
    # break into 4 16-bit cache and use precomputed parity
    return (PRECOMPUTED_PARITY[(x >> (3 * MASK_SIZE)) & BIT_MASK] ^ # 1st 16-bits
            PRECOMPUTED_PARITY[(x >> (2 * MASK_SIZE)) & BIT_MASK] ^ # 2nd 16-bits
            PRECOMPUTED_PARITY[(x >> (1 * MASK_SIZE)) & BIT_MASK] ^ # 3rd 16-bits
            PRECOMPUTED_PARITY[(x >> (0 * MASK_SIZE)) & BIT_MASK])  # 4th 16-bits

def parity5(x):
    # God-like solution
    # Time complexity: O(log n)
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 0x1

""" 4.2 SWAP BITS

    Take a 64-bit integer and swap the bits at indices i and j, assuming
    i and j are both [0-63]. For example, the result of swapping the 8-bit
    integer 73 (01001001) at indices 1 and 6 is 11 (00001011).
"""
def swap_bits(x, i, j):
    # Time complexity: O(1)
    # Extract the i-th and j-th bits, and see if they differ
    if (x >> i) & 1 != (x >> j) & 1:
        # i-th and j-th bits differ, swap them by flipping their values.
        bit_mask = (1 << i) | (1 << j)
        x ^= bit_mask # Flip the bit
    return x

""" 4.3 REVERSE BITS

    Take a 64-bit unsigned integer and return the 64-bit unsigned integer
    consisting of the bits of the input in reverse order. For example, if the
    input is (1110000000000001), the output should be (1000000000000111).
"""
def reverse_bits(x):
    # Brute-force
    # Time complexity: O(n)
    result = 0
    while x:
        result |= (x & 1)
        result <<= 1
        x >>= 1
    return result

def reverse_bits2(x):
    # For a very large 64-bit number, use 16-bit cache
    # Time complexity: O(n/L), where L is the width of the cache

    # 8-bit cache example: 00101001 <(00),(01),(10),(11)>
    PRECOMPUTED_REVERSE = {
        '00': '00',
        '10': '01',
        '01': '10',
        '11': '11',
    }
    MASK_SIZE = 16
    BIT_MASK = 0xFFFF
    return (PRECOMPUTED_REVERSE[(x >> (0 * MASK_SIZE)) & BIT_MASK] << (3 * MASK_SIZE) |
            PRECOMPUTED_REVERSE[(x >> (1 * MASK_SIZE)) & BIT_MASK] << (2 * MASK_SIZE) |
            PRECOMPUTED_REVERSE[(x >> (2 * MASK_SIZE)) & BIT_MASK] << (1 * MASK_SIZE) |
            PRECOMPUTED_REVERSE[(x >> (3 * MASK_SIZE)) & BIT_MASK] << (0 * MASK_SIZE))

""" 4.4 FIND A CLOSEST INTEGER WITH THE SAME WEIGHT

    Define the weight of a nonnegative integer x to be the number of bits that
    are set to 1 in its binary representation. For example, since 92 in base-2
    equals (01011100), the weight of 92 is 4.

    Takes a nonnegative integer x and returns a number y which is not equal to
    x, but has the same weight as x and their difference, |y-x|, is as small as
    possible. You are assume x is not 0, or all 1s. For example, if x = 6, you
    should return 5. You can assume the integer fits in 64 bits.
"""
def closest_int_same_bit_count(x):
    # Brute-force (using count_bits in 4.0)
    # Time complexity: O(n2)
    y = x - 1
    while y and count_bits(y) != count_bits(x):
        y -= 1
    return y

def closest_int_same_bit_count2(x):
    # God-like solution: swap 2 rightmost consecutive bits that differ
    # Time complexity: O(n)
    NUM_UNSINGED_BITS = 64
    for i in range(NUM_UNSINGED_BITS - 1):
        # compare 2 consecutive bits
        if (x >> i) & 1 != (x >> (i + 1)) & 1:
            x ^= (1 << i) | (1 << (i + 1)) # swap ith bit and (i-1)th bit
            return x
    # Raise error if all bits of x are 0 or 1.
    raise ValueError('All bits are 0 or 1')

""" 4.5 COMPUTE X * Y

    Multiplies two nonnegative integers. The only operators allowed are:
    * assignment
    * the bitwise operators >>, <<, &, ~, ^ and
    * equality checks and Boolean combinations thereof.
"""
def add(x, y):
    ''' Half adder truth table

        Input | Output
        --------------
        A   B   C   S
        --------------
        0   0   0   0
        0   1   0   1
        1   0   0   1
        1   1   1   0

        Carry = A & B
        Sum = A ^ B
    '''

    # Iterate until there is no carry
    while y:
        # Carry contains common set bits of x and y
        carry = x & y
        # Sum of bits of x and y where at least one of the bits is not set
        x = x ^ y
        # Carry is shifted by 1 so that adding it to x gives the sum
        y = carry << 1
    return x

def substract(x, y):
    ''' Half subtractor truth table

        Input | Output
        --------------
        A   B   D   B
        --------------
        0   0   0   0
        0   1   1   1
        1   0   1   0
        1   1   0   0

        Borrow = (~A) & B
        Diff = A ^ B
    '''

    # Iterate until there is no borrow
    while y:
        # Borrow contains common set bits of y and unset bits of x
        borrow = (~x) & y
        # Subtraction of bits of x and y where at least one of the bits is not set
        x = x ^ y
        # Borrow is shifted by 1 so that subtracting it from x gives the diff
        y = borrow << 1
    return x

def minus_one(x):
    k = 1
    # Flip all the bits until we find 1
    while not (x & k):
        x ^= k
        k = k << 1
    # Finally, flip back the Kth bit
    x ^= k
    return x

def add2(a, b):
    result, carry_in = 0, 0
    ta, tb, k = a, b, 1
    while ta or tb:
        # extract the Kth bit of a and b
        ak, bk = a & k, b & k
        # add up the result
        result |= ak ^ bk ^ carry_in
        # any carry-out bit?
        carry_out = (ak & bk) | (ak & carry_in) | (bk & carry_in)
        # carry-out is the next carry-in
        carry_in = carry_out << 1
        # move on to the next bit
        ta, tb, k = ta >> 1, tb >> 1, k << 1
    return result | carry_in

def multiply(x, y):
    # Initialize the result to 0 and iterate through the bits of x, adding
    # (2^k)y to the result if the Kth bit of x is 1
    # Time complexity: O(2^n)
    running_sum = 0
    while x:
        if x & 1:
            running_sum = add2(running_sum, y)
        x, y = x >> 1, y << 1
    return running_sum

""" 4.6 COMPUTE X / Y

    Given two positive integers, compute their quotient, using only the
    addition, substraction and shifting operators.
"""
def divide(x, y):
    # Time complexity: O(n), assuming individual shift and add operations take
    # o(1) time.
    result, power = 0, 32
    y_power = y << power
    while x >= y:
        while y_power > x:
            y_power >>= 1
            power -= 1
        result += 1 << power
        x -= y_power
    return result

""" 4.7 COMPUTE X ^ Y

    Take a double x and an integer y and returns x^y. You can ignore the
    overflow and underflow.
"""
def pow(x, y):
    # Time complexity: O(n)
    result, power = 1.0, y
    if y < 0:
        x, power = 1.0 / x, -power
    while power:
        if power & 1:
            result *= x
        x, power = x * x, power >> 1
    return result

""" 4.8 REVERSE BITS

    Take an integer and return the integer corresponding to the digits of the
    input written in reverse order. For example, the reverse of 42 is 24, and
    the reverse of -314 is -413. You may not convert the integer to string.
"""
def reverse(x):
    # Time complexity: O(n)
    result, remaining_x = 0, abs(x)
    while remaining_x:
        result = result * 10 + remaining_x % 10
        remaining_x //= 10
    return -result if x < 0 else result

""" 4.9 CHECK IF A DECIMAL INTEGER IS A PALINDROME

    A palindromic string is one which reads the same forwards and backwards,
    e.g., "redivider". You are to write a program which determines if the
    decimal representation of an integer is a palindromic string. For example,
    your program should return true for the inputs 0, 1, 7, 11, 121, 333 and
    214747412, and false for the inputs -1, 12, 100, and 2147483647.
"""
def is_palindrome_number(x):
    # Using reverse() from 4.8
    # Time complexity: O(n)
    if x <= 0:
        return x == 0

    return reverse(x) == x

def is_palindrome_number2(x):
    # Time complexity: O(n)
    if x <= 0:
        return x == 0

    n = math.floor(math.log10(x)) + 1 # get number of digits
    msd_mask = 10**(n - 1)
    for _ in range(n // 2):
        if x // msd_mask != x % 10:
            return False
        x %= msd_mask      # Remove the most significant digit of x
        x //= 10           # Remove the least significant digit of x
        msd_mask //= 100   # adjust the most significant digit mask
    return True

""" 4.10 GENERATE UNIFORM RANDOM NUMBERS

    How would you implement a random number generator that generates a random
    integer i between a and b, inclusive, given a random number generator that
    produces zero or one with equal probability? All values in [a,b] should be
    equally likely.
"""
def uniform_random(lower_bound, upper_bound):
    # Time complexity: O(log(b - a + 1))
    p = upper_bound - lower_bound + 1
    while True:
        result, i = 0, 0
        while (1 << i) < p:
            result = (result << 1) | random.getrandbits(1)
            i += 1
        if result < p:
            break
    return result + lower_bound

""" 4.11 RECTANGLE INTERSECTION

    The problem is concerned with rectangles whose sides are parallel to the
    X-axis and Y-axis. Write a program which tests if two rectangles have a
    nonempty intersection. If intersection is nonempty, return the rectangle
    formed by their intersection.
"""
Rectangle = collections.namedtuple('Rectangle', ('x', 'y', 'width', 'height'))

def is_intersect(R1, R2):
    return (R1.x + R1.width  >= R2.x and
            R2.x + R2.width  >= R1.x and
            R1.y + R1.height >= R2.y and
            R2.y + R2.height >= R1.y)

def intersect_rectangle(R1, R2):
    if not is_intersect(R1, R2):
        return Rectangle(0, 0, -1, -1) #  no intersection
    return Rectangle(
        max(R1.x, R2.x),
        max(R1.y, R2.y),
        min(R1.x + R1.width , R2.x + R2.width ) - max(R1.x, R2.x),
        min(R1.y + R1.height, R2.y + R2.height) - max(R1.y, R2.y))
