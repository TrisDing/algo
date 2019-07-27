from functools import reduce
from itertools import groupby
import string

"""
String Constants
----------------
>>> string.digits      # The string '0123456789'.
>>> string.hexdigits   # The string '0123456789abcdefABCDEF'.
>>> string.octdigits   # The string '01234567'

Key Operators
-------------
>>> s[3]
>>> len(s)
>>> s + t
>>> s * 3
>>> s[2:4]
>>> s in t
>>> s.strip([chars])
>>> s.startswith(prefix)
>>> s.endswith(prefix)
>>> s.slipt(delimiter)
>>> s.lower()
>>> s.upper()
>>> 'Name {name}, Rank {rank}'.format(name='Archimedes', rank=3)

Useful
------
# Given a string representing one Unicode character, return an integer
# representing the Unicode code point of that character. For example,
# ord('a') returns the integer 97
>>> ord(c)

# Return the string representing a character whose Unicode code point is
# the integer i. For example, chr(97) returns the string 'a'
>>> chr(i)

# functools.reduce() takes the first two elements A and B returned by the
# iterator and calculates func(A, B). It then requests the third element, C,
# calculates func(func(A, B), C), combines this result with the fourth element
# returned, and continues until the iterable is exhausted.
>>> functools.reduce(func, iter, [initial_value])
"""

""" 6.0 IS PALINDROM

    A palindrom string is one which reads the same when it is reversed. Rather
    than creating a new string for the reverse of the input string, it traverses
    the input string forwards and backwards, thereby saving space.
"""
def is_palindrom(s):
    # Note that s[~i] for i in [0, len(s) - 1] is s[-(i+1)].
    # This uniformly handles even and odd length string.
    # Time complexity: O(n), Space complexity: O(1)
    return all(s[i] == s[~i] for i in range(len(s) // 2))

""" 6.1 INTERCONVERT STRINGS AND INTEGERS

    You are to implement methods that a string representing an integer and
    return the corresponding integer, and vice versa. Your code should handle
    negative integers. You cannot use library functions like int() or str().
"""
def int_to_string(x):
    # Time complexity: O(n), Space complexity: O(1)
    signed = False
    if x < 0:
        signed = True
        x = -x

    s = []
    while x != 0:
        s.append(chr(ord('0') + x % 10))
        x //= 10

    return ('-' if signed else '') + ''.join(reversed(s))

def string_to_int(s):
    # Time complexity: O(n), Space complexity: O(1)
    return reduce(lambda res, c: res * 10 + string.digits.index(c),
        s[s[0] == '-':], 0) * (-1 if s[0] == '-' else 1)

""" 6.2 BASE CONVERSION

    Write a program that performs base conversion. The input is a string, an
    integer b1, and another integer b2. The string represents an integer in base
    b1. The output should be the string representing the integer in base b2.
    Assume b1 >= 2, b2 <= 16. Use "A" to represent 10, "B" to represent 11,...,
    and "F" to represent 15.
"""
def convert_base(s, b1, b2):
    # Time complexity: O(n), Space complexity: O(1)
    signed = True if s[0] == '-' else False
    s = s[s[0] == '-':]

    total = reduce(lambda res, tup: res + string.hexdigits.index(tup[1].lower()) *
        (b1 ** (len(s) - tup[0] - 1)), enumerate(s), 0)

    result = []
    for i in range(len(s)):
        op = b2 ** (len(s) - i - 1)
        result.append(string.hexdigits[total // op].upper())
        total = total % op

    return ('-' if signed else '') + ''.join(result)

def convert_base2(s, b1, b2):
    # Time complexity: O(n), Space complexity: O(1)
    def construct_from_base(num, base):
        return ('' if num == 0 else
            construct_from_base(num // base, base) + string.hexdigits[num % base].upper())

    signed = s[0] == '-'
    num = reduce(lambda x, c: x * b1 + string.hexdigits.index(c.lower()), s[signed:], 0)
    return ('-' if signed else '') + ('0' if num == 0 else construct_from_base(num, b2))

""" 6.3 COMPUTE THE SPREADSHEET COLUMN ENCODING

    Implement a function that converts a spreadsheet column id to the corresponding
    integer, with "A" corresponding to 1. For example, you should return 4 for "D",
    27 for "AA", 702 for "ZZ", etc. How would you test your code?
"""
def decode_spreadsheet_column(column):
    # Time complexity: O(n), Space complexity: O(1)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = 0
    for i in range(len(column)):
        idx = alphabet.index(column[i].upper()) + 1
        result += idx * 26 ** (len(column) - 1 - i)

    return result

def decode_spreadsheet_column2(column):
    # Time complexity: O(n), Space complexity: O(1)
    return reduce(lambda res, c: res * 26 + ord(c) - ord('A') + 1, column, 0)

""" 6.4 REPLACE AND REMOVE

    Write a program which takes as input an array of characters, and removes each
    'b' and replaces each 'a' by two 'd's, Specifically, along with the array,
    you are provided an integer-valued size. Size donates the number of entries
    of array that the operation is to be applied to. You do not have to worry
    about preserving subsequent entries. For example, if the array is <a,b,a,c,_>
    and the size is 4, then you can return <d,d,d,d,c>. You can assume there is
    enough space in the array to hold the final result.
"""
def replace_and_remove(size, s):
    # Time complexity: O(n), Space complexity: O(n)
    result = []
    for i in range(len(s)):
        if s[i] == 'b':
            continue
        elif s[i] == 'a':
            result.append('d')
            result.append('d')
        else:
            result.append(s[i])
    return result

def replace_and_remove2(size, s):
    # Time complexity: O(n), Space complexity: O(1)

    # forward interation: remove 'b's and count the number of 'a's
    prev, count_a = 0, 0
    for i in range(size):
        if s[i] != 'b':
            s[prev] = s[i]
            prev += 1
        if s[i] == 'a':
            count_a += 1

    # backward interation: replace 'a's with 'dd's starting from the end
    curr = prev - 1
    prev += count_a - 1
    final_size = prev + 1
    while curr >=0 :
        if s[curr] == 'a':
            s[prev - 1 : prev + 1] = 'dd'
            prev -= 2
        else:
            s[prev] = s[curr]
            prev -= 1
        curr -= 1

    return final_size

""" 6.5 TEST PALINDROME

    Implement a function which takes as input a string s and returns true if s
    is a palindromic string. The string could contain upper case, lower case
    and nonalphanumeric characters.
"""
def test_palindrome(s):
    # Time complexity: O(n), Space complexity: O(1)
    i, j = 0, len(s) - 1
    while i < j:
        if not s[i].isalnum() and i < j:
            i += 1
        if not s[j].isalnum() and i < j:
            j -= 1
        if s[i].lower() != s[j].lower():
            return False
        i, j = i + 1, j - 1
    return True

def test_palindrome2(s):
    return all(a == b for a, b in zip(
        map(str.lower, filter(str.isalnum, s)),
        map(str.lower, filter(str.isalnum, reversed(s)))))

""" 6.6 REVERSE ALL WORDS IN A SENTENCE

    Implement a function for reversing the words in string s seperated by
    whitespace. For example, "Alice likes Bob" tranforms to "Bob likes Alice".
    We do not need to keep the original string.
"""
def reverse_words(s):
    # First, reverse the whole string.
    s.reverse()

    def reverse_range(s, start, finish):
        while start < finish:
            s[start], s[finish] = s[finish], s[start]
            start, finish = start + 1, finish - 1

    start = 0
    while True:
        finish = s.find(b' ', start)
        if finish < 0:
            break
        # Reverse each word in the string.
        reverse_range(s, start, finish - 1)
        start = finish + 1

    # Reverse the last word
    reverse_range(s, start, len(s) - 1)

""" 6.7 COMPUTE ALL MNEMONICS FOR A PHONE NUMBER

    Write a program which takes as input a phone number, specified a string of
    digits, and returns all possible character sequences that correspond to the
    phone number. The cell phone keypad is specified by a mapping that takes a
    digit and returns the corresponding set of characters. The character
    sequences do not have to be legal words or phrases.
"""
def phone_mnemonic(phone):
    MAPPING = ('0', '1', 'ABC', 'DEF', 'GHI', 'JKL', 'MNO', 'PQR', 'TUV', 'WXYZ')
    def iter(digit):
        if digit == len(phone):
            # All digits are processed, so add partial to mnemonics.
            # We add a copy since subsequent calls modify partial.
            mnemonics.append(''.join(partial))
        else:
            # Try all possible characters for this digit.
            for c in MAPPING[int(phone[digit])]:
                partial[digit] = c
                iter(digit + 1)

    mnemonics, partial = [], [0] * len(phone)
    iter(0)
    return mnemonics

""" 6.8 THE LOOK-AND-SAY PROBLEM

    Write a program that takes as input an integer n and returns the nth integer
    in the look-and-say sequence. Return the result as a string. For example,
    the first eight numbers in the look-and-say sequence are <1,11,21,1211,111221,
    312211,13112221,1113213211>
"""
def look_and_say(n):
    if n < 1:
        return None

    def iter(s, n):
        if n <= 1:
            return s
        next_s, i = [], 0
        while i < len(s):
            count = 1
            while i + 1 < len(s) and s[i] == s[i + 1]:
                i += 1
                count += 1
            next_s.append(str(count) + s[i])
            i += 1
        return iter(''.join(next_s), n - 1)

    return iter('1', n)

def look_and_say2(n):
    s = '1'
    for _ in range(n - 1):
        s = ''.join(str(len(list(group))) + key for key, group in groupby(s))
    return s

""" 6.9 CONVERT FROM ROMAN TO DECIMAL

    Write a program which takes as input a valid Roman number string s and returns
    the integer corresponding to. For example, the strings "XXXXXIIIIIIIII",
    "LVIIII" and "LIX" are valid Roman number string representing 59. The shortest
    valid complex Roman number string corresponding to the integer 59 is "LIX"
"""
def roman_to_integer(s):
    T = { 'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'M':1000 }

    return reduce(lambda val, i: val + (-T[s[i]]) if T[s[i]] < T[s[i+1]] else T[s[i]],
        reversed(range(len(s) - 1)), T[s[-1]])

""" 6.10 COMPUTE ALL VALID IP ADDRESS

    Write a program that determines where to add periods to a decimal string so
    that the resulting string is a valid IP address. There maybe more than one
    valid IP address corresponding to a string, in which ase you should print
    all possibilities.
"""
def valid_ip_address(s):
    if len(s) < 4 or len(s) > 12:
        return None

    def is_valid_part(s):
        # '00', '000', '01', etc. are not valid, but '0' is valid.
        return len(s) == 1 or (s[0] != '0' and int(s) <= 255)

    result, parts = [], [None] * 4
    for i in range(1, min(len(s), 4)):
        parts[0] = s[:i]
        if is_valid_part(parts[0]):
            for j in range(1, min(len(s) - i, 4)):
                parts[1] = s[i:i+j]
                if is_valid_part(parts[1]):
                    for k in range(1, min(len(s) - i - j, 4)):
                        parts[2] = s[i+j:i+j+k]
                        parts[3] = s[i+j+k:]
                        if is_valid_part(parts[2]) and is_valid_part(parts[3]):
                            result.append('.'.join(parts))
    return result

""" 6.11 WRITE A STRING SINUSOIDALLY

    Define the snake string of s to be the left-right top-to-bottom sequence in
    which characters appear when s is written in sinusoidal fashion. For example,
    the snake string for "Hello_World!" is "e lHloWrdlo!". Write a program which
    takes as input a string s and returns the snake string of s.
"""
def snake_string(s):
    top, middle, bottom = [], [], []
    direction = [0, 1, 0, -1] * (len(s) // 4)
    for i in range(len(s)):
        if direction[i] > 0:
            top.append(s[i])
        elif direction[i] == 0:
            middle.append(s[i])
        elif direction[i] < 0:
            bottom.append(s[i])
        print(top, middle, bottom)
    return ''.join(top) + ''.join(middle) + ''.join(bottom)

def snake_string2(s):
    result = []
    for i in range(1, len(s), 4):
        result.append(s[i]) # top, s[1], s[5], s[9], ...
    for i in range(0, len(s), 2):
        result.append(s[i]) # middle: s[0], s[2], s[4], ...
    for i in range(3, len(s), 4):
        result.append(s[i]) # bottom: s[3], s[7], s[11], ...
    return ''.join(result)

def snake_string3(s):
    return s[1::4] + s[::2] + s[3::4]

""" 6.12 IMPLEMENT RUN-LENGTH ENCODING

    Implement run-length encoding and decoding functions. Assume the string to
    be encoded consists of letters of the alphabet with no digits, and the string
    to be decoded is a valid encoding.
"""
def encode(s):
    result = []
    count, i = 1, 0
    while i < len(s):
        if i < len(s) - 1 and s[i] == s[i+1]:
            count += 1
        else:
            result.append(str(count) + s[i])
            count = 1
        i += 1
    return ''.join(result)

def decode(s):
    result = []
    count = 0
    for c in s:
        if c.isdigit():
            count = int(c)
        else:
            result.append(c * count)
            count = 0
    return ''.join(result)

""" 6.13 FIND THE FIRST OCCURRENCE OF A SUBSTRING

    Given two string s(the search string) and t(the text), find the first
    occurrence of s in t.
"""
def search(t, s):
    if len(s) > len(t):
        return -1

    search_idx, i = 0, 0
    for i in range(len(t)):
        search_idx = search_idx + 1 if t[i] == s[search_idx] else 0
        if search_idx == len(s) - 1:
            return i - search_idx + 1
    return -1

def rabin_karp(t, s):
    if len(s) > len(t):
        return -1

    BASE = 26
    # Hash codes for the substring of t and s
    t_hash = reduce(lambda h, c: h * BASE + ord(c), t[:len(s)], 0)
    s_hash = reduce(lambda h, c: h * BASE + ord(c), s, 0)
    power_s = BASE**max(len(s) - 1, 0) # BASE^|s-1|.

    print(t_hash, s_hash, power_s)

    for i in range(len(s), len(t)):
        # Checks the two substrings are actually equal or not, to protect
        # against hash collision.
        if t_hash == s_hash and t[i - len(s):i] == s:
            return i - len(s) # Found a match.

        # Uses rolling hash to compute the hash code.
        t_hash -= ord(t[i - len(s)]) * power_s
        t_hash = t_hash * BASE + ord(t[i])

    # Tries to match s and t[-len(s):].
    if t_hash == s_hash and t[-len(s):] == s:
        return len(t) - len(s)

    return -1 # s is not a substring of t.