import sys
import math
import random
import collections

# debugging util
def fb(num):
    return "{0:b}".format(num)

# Numbers boundary
# Integers in Python is unbound
sys.maxsize    # 2**63 - 1 (64 bit machine)
sys.float_info

# Bit-wise operators
6 & 4    # 0110 & 0100 = 0100 (4)                     AND
1 | 2    # 0001 | 0010 = 0011 (3)                     OR
15 ^ 1   # 00001111 ^ 00000001 = 00001110 (14)        XOR
8 >> 1   # 00001000 >> 1 = 00000100 (4)               x >> y = x // 2**y
1 << 10  # 000000000001 << 10 = 010000000000 (1024)   x << y = x *  2**y
-16 >> 2 # 11110000 >> 2 = 11111100 (-4)              negative shifting, convert to 2's complement
-16 << 2 # 11110000 << 2 = 11000000 (-64)             negative shifting, convert to 2's complement
~0       # ~0000 = 1111 (-1)                          ~x = -x - 1

# Tricks
x, i = 0xFFFF, 1
x & 1             # Extract the last bit
x = x | 1         # Set the last bit
x = x ^ 1         # Flip the last bit
(x >> i) & 1      # Extract the ith bit
x = x | (1 << i)  # Set the ith bit
x = x ^ (1 << i)  # Flip the ith bit
x & x - 1         # Drop the lowest set bit of x
x & ~(x - 1)      # Extract the lowest set bit of x

# Key methods for numeric types
abs(-34.5)
math.ceil(2.17)
math.floor(3.14)
min(-1, 3, 10)
max(-1, 4, 5)
pow(2.71, 3.14)  # same as **

# Interconvert integers and strings
str(42)
int('42')
str(3.14)
float('3.14')

# Float infinity
float('inf')
float('-inf')

# comparing float point values
math.isclose(1e-09, 1e-09+1)

# Key methods in random
random.random()                       # Random float x, 0.0 <= x < 1.0
random.randrange(0, 101, 2)           # Even integer from 0 to 100
random.randint(1, 10)                 # Integer from 1 to 10, endpoints included
random.shuffle([1, 2, 3, 4, 5, 6, 7]) # shuffle a list
random.choice('abcdefghij')           # Choose a random element
random.sample([1, 2, 3, 4, 5],  3)    # Choose 3 elements
random.getrandbits(1)                 # Generate random bit (0/1)

# 4.0 Count Bits
def count_bits(x):
    """
    Count the number of bits that are set to 1
    in a nonnegative integer.
    Time complexity: O(n), where n is the word size
    """
    num_bits = 0
    while x:
        num_bits += x & 1
        x >>= 1
    return num_bits

# 4.1 COMPUTING THE PARITY OF A VERY LARGE NUMBER OF 64-BIT WORD
def parity(x):
    """
    Time complexity: O(n), where n is the word size
    """
    num_bits = 0
    while x:
        num_bits += x & 1
        x >>= 1
    return num_bits % 2

def parity2(x):
    """
    Time complexity: O(n), where n is the word size
    """
    result = 0
    while x:
        result ^= x & 1 # Don't have to add them up, just XOR
        x >>= 1
    return result

def parity3(x):
    """
    Time complexity: O(k), where k is the number of bit set to 1
    """
    result = 0
    while x:
        result ^= 1
        x &= x - 1 # Drop the lowest set bit of x
    return result

def parity4(x):
    """
    For a very large 64-bit number, use 16-bit cache
    Time complexity: O(n/L), where L is the width of the cache
    """
    # 8 bit example: 00101001 <(00),(01),(10),(11)>
    PRECOMPUTED_PARITY = {
        '00': 0,
        '10': 1,
        '01': 1,
        '11': 0,
    }
    MASK_SIZE = 16
    BIT_MASK = 0xFFFF
    # break into 4 16-bit cache and use precomputed parity
    return (PRECOMPUTED_PARITY[(x >> (3 * MASK_SIZE)) & BIT_MASK] ^
            PRECOMPUTED_PARITY[(x >> (2 * MASK_SIZE)) & BIT_MASK] ^
            PRECOMPUTED_PARITY[(x >> (1 * MASK_SIZE)) & BIT_MASK] ^
            PRECOMPUTED_PARITY[(x >> (0 * MASK_SIZE)) & BIT_MASK])

def parity5(x):
    """
    Time complexity: O(log n)
    """
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 0x1

# 4.2 SWAP BITS
def swap_bits(x, i, j):
    """
    Assuming: x = 64-bit integer, i and j = [0-63]
    Time complexity: O(1)
    """
    if i != j and (x >> i & 1) != (x >> j & 1):
        BIT_MASK = (1 << i) | (1 << j)
        x ^= BIT_MASK # Flip the bit
    return x

# 4.3 REVERSE BITS
def reverse_bits(x):
    """
    Time complexity: O(n)
    """
    result = 0
    while x:
        result |= (x & 1)
        result <<= 1
        x >>= 1
    return result

def reverse_bits2(x):
    """
    For a very large 64-bit number, use 16-bit cache
    Time complexity: O(n/L), where L is the width of the cache
    """
    # 8 bit example: 00101001 <(00),(01),(10),(11)>
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

# 4.4 FIND A CLOSEST INTEGER WITH THE SAME WEIGHT
def closest_int_same_bit_count(x):
    """
    Time complexity: O(n2)
    """
    y = x - 1
    while count_bits(y) != count_bits(x):
        y -= 1
    return y

def closest_int_same_bit_count2(x):
    """
    Swap 2 rightmost consecutive bits that differ
    Time complexity: O(n)
    """
    NUM_UNSINGED_BITS = 64
    for i in range(NUM_UNSINGED_BITS - 1):
        if (x >> i) & 1 != (x >> (i + 1)) & 1:
            x ^= (1 << i) | (1 << (i + 1)) # swap bit-i and bit-(i+1)
            return x
    # Raise error if all bits of x are 0 or 1.
    raise ValueError('All bits are 0 or 1')

# 4.5 COMPUTE X*Y
def multiply(x, y):
    """
    x, y are non negative integers
    only allow to use bitwise operations
    Time complexity: O(n2)
    """
    def add(a, b):
        running_sum, carry_in, k, temp_a, temp_b = 0, 0, 1, a, b
        while temp_a or temp_b:
            ak, bk = a & k, b & k # extract k-th bit
            carry_out = (ak & bk) | (ak & carry_in) | (bk & carry_in)
            running_sum |= ak ^ bk ^ carry_in
            carry_in, k, temp_a, temp_b = (carry_out << 1, k << 1, temp_a >> 1, temp_b >> 1)
        return running_sum | carry_in

    running_sum = 0
    while x: # Extract each bit of x
        if x & 1:
            running_sum = add(running_sum, y)
        x >>= 1
        y <<= 1
    return running_sum

# 4.6 COMPUTE X/Y
def divide(x, y):
    """
    x, y are non-negative integers
    only allow to use addition, substraction and shifting
    """
    print(fb(x), fb(y))
    result, power = 0, 32
    y_power = y << power
    print("y_power =", fb(y_power))
    while x >= y:
        print("x =", fb(x))
        while y_power > x:
            print("shifting y_power")
            y_power >>= 1
            power -= 1
        result += 1 << power
        x -= y_power
        print("result =", fb(result))
    return result

# 4.7 COMPUTE X^Y
def pow(x, y):
    """
    x is double, y is integer
    ignore overflow and underflow
    Time complexity: O(n)
    """
    result, power = 1.0, y
    if y < 0:
        x, power = 1.0 / x, -power
    while power:
        if power & 1:
            result *= x
        x, power = x * x, power >> 1
    return result

# 4.8 REVERSE BITS
def reverse(x):
    """
    Time complexity: O(n)
    """
    result, remaining_x = 0, abs(x)
    while remaining_x:
        result = result * 10 + remaining_x % 10
        remaining_x //= 10
    return -result if x < 0 else result

# 4.9 CHECK PALINDROME
def is_palindrome_number(x):
    """
    Time complexity: O(n)
    """
    if x <= 0:
        return x == 0
    reverse_x, remaining_x = 0, abs(x)
    while remaining_x:
        reverse_x = reverse_x * 10 + remaining_x % 10
        remaining_x //= 10
    return reverse_x == x

def is_palindrome_number2(x):
    """
    Time complexity: O(n)
    """
    if x <= 0:
        return x == 0
    num_digits = math.floor(math.log10(x)) + 1
    msd_mask = 10**(num_digits - 1)
    for i in range(num_digits // 2):
        if x // msd_mask != x % 10:
            return False
        x %= msd_mask # Remove the most significant digit of x
        x //= 10 # Remove the least significant digit of x
        msd_mask //= 100
        i = i # no meaning
    return True

# 4.10 GENERATE UNIFORM RANDOM NUMBERS
def uniform_random(lower_bound, upper_bound):
    """
    Time complexity: O(log(b - a + 1))
    """
    possibilities = upper_bound - lower_bound + 1
    while True:
        result, i = 0, 0
        while (1 << i) < possibilities:
            result = (result << 1) | random.getrandbits(1)
            i += 1
        if result < possibilities:
            break
    return result + lower_bound

# 4.11 RECTANGLE INTERSECTION
Rectangle = collections.namedtuple('Rectangle', ('x', 'y', 'width', 'height'))

def intersect_rectangle(R1, R2):
    def is_intersect(R1, R2):
        return (R1.x + R1.width  >= R2.x and 
                R2.x + R2.width  >= R1.x and
                R1.y + R1.height >= R2.y and
                R2.y + R2.height >= R1.y)
    if not is_intersect(R1, R2):
        return Rectangle(0, 0, -1, -1) #  no intersection
    return Rectangle(
        max(R1.x, R2.x),
        max(R1.y, R2.y),
        min(R1.x + R1.width , R2.x + R2.width ) - max(R1.x, R2.x),
        min(R1.y + R1.height, R2.y + R2.height) - max(R1.y, R2.y))

R1 = Rectangle(0, 0, 2, 2)
R2 = Rectangle(2, 2, 2, 2)
print(intersect_rectangle(R1, R2))