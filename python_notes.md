# Python Notes

## 1. Syntax

### Comments
```py
# single line comments

"""
Documentation
"""
```

### Built-in Constants
```py
False
True
None
NotImplemented
Ellipsis
```

### Build-in Types
```py
int, float, complex   # Number Types
iter, next, yield     # Iterator Types
list, tuple, range    # Sequence Types
str                   # Text Sequance Types
bytes, bytearray      # Binary Sequence Types
set, frozenset        # Set Types
dict                  # Map Types
```

### Literals
```py
1                     # int
1.34                  # float
0b1010                # binary
100                   # decimal 
0o310                 # octal
0x12c                 # hexadecimal

"string"              # string
'c'                   # char
"""multiline"""       # multiline str
u"\u00dcnic\u00f6de"  # unicode
r"raw \n string"      # raw string

[1, 'a', False]       # list
(1, 2, 3)             # tuple
{'a': 1, 'b': 2}      # dict
{'a', 'b', 'c'}       # set
```

```py
type(1)               # <class 'int'>
type(1.0)             # <class 'float'>
type('')              # <class 'str'>
type("")              # <class 'str'>
type(True)            # <class 'bool'>
type(False)           # <class 'bool'>
type(None)            # <class 'NoneType'>
type([])              # <class 'list'>
type({})              # <class 'dict'>
type(())              # <class 'tuple'>
type({''})            # <class 'set'>
```

### Comparisions
```py
x < y                 # strictly less than
x <= y                # less than or equal
x > y                 # strictly greater than
x >= y                # greater than or equal
x == y                # equal
x != y                # not equal
x is y                # object identity
x is not y            # negated object identity
```

### Number Operations
```py
x + y                 # sum of x and y
x - y                 # difference of x and y
x * y                 # product of x and y
x ** y                # x to the power y
x / y                 # quotient of x and y
x // y                # floored quotient of x and y
x % y                 # remainder of x / y
-x                    # x negated
+x                    # x unchanged
abs(x)	              # absolute value or magnitude of x
int(x)	              # x converted to integer
float(x)              # x converted to floating point
divmod(x, y)	      # the pair (x // y, x % y)
pow(x, y)             # x to the power y
```

### Bitwise Operations
```py
x | y	              # bitwise 'or' of x and y
x ^ y	              # bitwise 'exclusive or' of x and y
x & y	              # bitwise 'and' of x and y
x << n	              # x shifted left by n bits
x >> n	              # x shifted right by n bits
~x                    # the bits of x inverted
```

### Sequence Operations
```py
x in s	              # True if an item of s is equal to x, else False
x not in s            # False if an item of s is equal to x, else True
s + t	              # the concatenation of s and t
s * n or n * s	      # equivalent to adding s to itself n times
s[i]	              # ith item of s, origin 0
s[i:j]	              # slice of s from i to j
s[i:j:k]              # slice of s from i to j with step k
len(s)	              # length of s
min(s)	              # smallest item of s
max(s)	              # largest item of s
s.count(x)            # total number of occurrences of x in s
```

### References
```py
a = 0                 # assignment
a, b = 1, 2           # multiple assignments
a, b = b, a           # swap reference of a and b
a = b = 1             # a, b have different reference (primitives are immutable)
a = b = []            # a, b have same reference (lists are mutable)
```

## 2. Functions

### Function Definition

```py
def least_difference(a, b, c):
    """
    Return the smallest difference between any two numbers among a, b and c.

    Parameters
    ----------
    a: int
        first number
    b: int
        second number
    c: int
        third number
    
    Returns
    -------
    y: the smallest difference.

    Examples
    --------
    >>> least_difference(1, 5, -5)
    4
    >>> least_difference(1, 10, 10)
    0
    """
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    return min(diff1, diff2, diff3)

# Function Calls
least_difference(5, 6, 7)

# Getting Help
help(least_difference)
```

### Default and optional arguments
```py
def greet(who="Colin"):
    print("Hello,", who)

greet()              # Hello Colin
greet(who="Kaggle")  # Hello Kaggle
greet("world")       # Hello World
```

### Functions are objects
```py
def f(x):
    return x * 2

type(f)  # <class 'function'>

def call(fn, arg):
    return fn(arg)

def squared_call(fn, arg):
    return fn(fn(arg))

call(f, 1)           # 2
squared_call(f, 1)   # 4

# Function returns None if no return statements
print(print("Sam"))  # None
```

### Lambda functions
```py
mod_5 = lambda x: x % 5
mod_5(101)     # 1

abs_diff = lambda a, b: abs(a-b)
abs_diff(5, 7) # 2

always_3 = lambda: 3
always_3()     # 3

names = ['jacques', 'Ty', 'Mia', 'pui-wa']
max(names, key=len)                           # jacques
sorted(names, key=lambda name: name.lower())  # Names sorted case insensitive
```

### Print
```py
print('line-1', 'line-2', sep='\n')

print('The story of {0}, {1}, and {other}.'.format('Bill', 'Manfred', other='Georg'))

for x in range(1, 11):
    print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x))
```

### Build-in functions
```py
# Arithmetic
abs()
divmod()
pow()
round()

# Types
type()
bool()
bytearray()
bytes()
complex()
dict()
float()
list()
str()
int()
set()
frozenset()
tuple()
slice()
object()

# Conversions
ascii()
bin()
chr()
hex()
oct()
hash()
repr()
format()

# List & Iteration
len()
iter()
next()
range()
enumerate()
ord()
all()
any()
filter()
map()
reversed()
sorted()
zip()
sum()
max()
min()

# Class & functions
classmethod
staticmethod
property()
super()
id()
callable()
help()
delattr()
getattr()
setattr()
hasattr()
isinstance()
issubclass()
vars()

# System & Program
dir()
open()
print()
input()
locals()
memoryview()
compile()
eval()
```

## 3. Conditions
```py
3.0 == 3               # True
'3' == 3               # False
True and False         # False
True or False          # True
not True               # False
True or True and False # False

# if elif else
def inspect(x):
    if x == 0:
        print(x, "is zero")
    elif x > 0:
        print(x, "is positive")
    elif x < 0:
        print(x, "is negative")
    else:
        print(x, "is unlike anything I've ever seen...")

bool(1)      # all numbers are treated as true, except 0
bool(0)      # zero is false
bool("asf")  # all strings are treated as true, except the empty string ""
bool("")     # empty string is false

# python version of cond ? x : y
outcome = 'failed' if grade < 60 else 'passed' 
```

## 4. Lists

### 
```py
primes = [3, 5, 2, 7]

planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

hands = [
    ['J', 'Q', 'K'],
    ['2', '2', '2'],
    ['6', 'A', 'K'],
]
hands = [['J', 'Q', 'K'], ['2', '2', '2'], ['6', 'A', 'K']]

my_favourite_things = [32, 'raindrops on roses', help,]

planets[0]   # Mercury
planets[1]   # Venus
planets[-1]  # Neptune
planets[-2]  # Uranus
```

### Slicing
```py
planets[0:3]   # ['Mercury', 'Venus', 'Earth']
planets[:3]    # ['Mercury', 'Venus', 'Earth']
planets[3:]    # ['Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planets[1:-1]  # ['Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus']
planets[-3:]   # ['Saturn', 'Uranus', 'Neptune']
```

### Mutating
```py
planets[3] = 'Malacandra'
# ['Mercury', 'Venus', 'Earth', 'Malacandra', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

planets[:3] = ['Mur', 'Vee', 'Ur']
# ['Mur', 'Vee', 'Ur', 'Malacandra', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

planets[:4] = ['Mercury', 'Venus', 'Earth', 'Mars']
# ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
```

### List Functions
```py
len(primes)     # 4
sum(primes)     # 17
max(primes)     # 7
min(primes)     # 2
sorted(primes)  # [2, 3, 5, 7]
```

### List Methods
```py
planets.append('Pluto')
planets.pop()              # Pluto
planets.index('Earth')     # 2
planets.index('Pluto')     # ValueError: 'Pluto' is not in list
"Earth" in planets         # True
"Calbefraques" in planets  # False
```

### Tuples (immutable lists)
```py
t = (1, 2, 3)
t = 1, 2, 3    # parentheses are optional
t[0] = 100     # TypeError: 'tuple' object does not support item assignment

x = 0.125
x.as_integer_ratio()  # (1, 8)

numerator, denominator = x.as_integer_ratio()
numerator / denominator  # 0.125
```

## 5. Loops

### Lists
```py
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
for planet in planets:
    print(planet, end=' ')
```

### Tuples
```py
multiplicands = (2, 2, 2, 3, 3, 5)
product = 1
for mult in multiplicands:
    product = product * mult
```

### Strings
```py
s = 'steganograpHy is the practicE of conceaLing a file, message, image, or video within another fiLe, message, image, Or video.'
msg = ''
for char in s:
    if char.isupper():
        print(char, end='')
```

### Range
```py
for i in range(5):
    print("Doing important work. i =", i)

list(range(5)) # [0, 1, 2, 3, 4]

nums = [1, 2, 4, 8, 16]
for i in range(len(nums)): # Iterate over indices
    nums[i] = nums[i] * 2
```

### Enumerate
```py
def double_odds(nums):
    for i, num in enumerate(nums):
        if num % 2 == 1:
            nums[i] = num * 2

x = list(range(10))
double_odds(x)
print(x)  # [0, 2, 2, 6, 4, 10, 6, 14, 8, 18]

list(enumerate(['a', 'b']))  # [(0, 'a'), (1, 'b')]

nums = [
    ('one', 1, 'I'),
    ('two', 2, 'II'),
    ('three', 3, 'III'),
    ('four', 4, 'IV'),
]
for word, integer, roman_numeral in nums:
    print(integer, word, roman_numeral, sep=' = ', end='; ')
# 1 = one = I; 2 = two = II; 3 = three = III; 4 = four = IV;
```

### While
```py
i = 0
while i < 10:
    print(i, end=' ')
    i += 1
# 0 1 2 3 4 5 6 7 8 9 
```

### List comprehensions
```py
squares = [n**2 for n in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

short_planets = [planet for planet in planets if len(planet) < 6]
print(short_planets)  # ['Venus', 'Earth', 'Mars']

loud_short_planets = [planet.upper() + '!' for planet in planets if len(planet) < 6]
print(loud_short_planets)  # ['VENUS!', 'EARTH!', 'MARS!']

def count_negatives(nums):
    return len([num for num in nums if num < 0])
print(count_negatives([5, -1, -2, 0, 3]))  # 2
```

## 6. Strings

### String syntax
```py
x = 'Pluto is a planet'
y = "Pluto is a planet"
x == y  # True

"Pluto's a planet!"
'My dog is named "Pluto"'
'Pluto\'s a planet!'

a = "hello\nworld"
b = """hello
world"""
a == b # True
```

### Strings are sequences
```py
# Indexing
planet = 'Pluto'
planet[0]    # 'P'

# Slicing
planet[-3:]  # 'uto'

# Length
len(planet)  # 5

# Looping
[char+'! ' for char in planet] # ['P! ', 'l! ', 'u! ', 't! ', 'o! ']

# String is immutable
planet[0] = 'B'  # 'str' object does not support item assignment
planet.append    # AttributeError: 'str' object has no attribute 'append'
```

### String methods
```py
claim = "Pluto is a planet!"

claim.upper()            # 'PLUTO IS A PLANET!'
claim.lower()            # 'pluto is a planet!'
claim.index('plan')      # 11
claim.startswith(planet) # True
claim.endswith('dwarf')  # False

# slipt and join
datestr = '1956-01-31'
year, month, day = datestr.split('-')
'/'.join([month, day, year])  # 01/31/1956

# unicode characters in string literals
' 👏 '.join([word.upper() for word in claim.split()]) # 'PLUTO 👏 IS 👏 A 👏 PLANET!'

pos = 9 
planet + ", you'll always be the " + pos + "th planet to me."  # TypeError: can only concatenate str (not "int") to str
planet + ", you'll always be the " + str(pos) + "th planet to me."

# Building strings with .format()
"{}, you'll always be the {}th planet to me.".format(planet, pos)

pluto_mass = 1.303 * 10**22
earth_mass = 5.9722 * 10**24
population = 52910390

# 2 decimal points, 3 decimal points, format as percent and separate with commas
"{} weighs about {:.2} kilograms ({:.3%} of Earth's mass). It is home to {:,} Plutonians.".format(
    planet, pluto_mass, pluto_mass / earth_mass, population,
)

# Referring to format() arguments by index, starting from 0
s = """Pluto's a {0}.
No, it's a {1}.
{0}!
{1}!""".format('planet', 'dwarf planet')
```

## 7. Dictionaries
```py
numbers = {'one': 1, 'two': 2, 'three': 3}
numbers['one']            # 1
numbers['eleven'] = 11    # numbers => {'one': 1, 'two': 2, 'three': 3, 'eleven': 11}
numbers['one'] = 'Pluto'  # numbers => {'one': 'Pluto', 'two': 2, 'three': 3, 'eleven': 11}

# Same list comprehensions
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planet_to_initial = {planet: planet[0] for planet in planets}

'Saturn' in planet_to_initial     # True
'Betelgeuse' in planet_to_initial # False

# loop over its keys
for k in numbers:
    print("{} = {}".format(k, numbers[k]))

# Get all the initials, sort them alphabetically, and put them in a space-separated string.
' '.join(sorted(planet_to_initial.values())) # 'E J M M N S U V'

# dict.items() lets us iterate over the keys and values of a dictionary simultaneously.
for planet, initial in planet_to_initial.items():
    print("{} begins with \"{}\"".format(planet.rjust(10), initial))
```