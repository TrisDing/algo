### Chapter 1 Arrays and Strings ###

# 1.1 UNIQUE CHARS
def is_unique_chars(S):
    """
    Time = O(nlogn), Space = O(n)
    """
    if len(S) == 0:
        return False
    A = sorted(S)
    for i in range(len(A) - 1):
        if A[i] == A[i + 1]:
            return False
    return True

# Solution
def is_unique_chars2(S):
    """
    Assuming the string is ASCII
    Time = O(n), Space = O(1)
    """
    S = S.replace(' ', '')
    if len(S) > 256:
        return False
    char_set = [False] * 256
    for c in S:
        i = ord(c)
        if char_set[i]:
            return False
        char_set[i] = True
    return True

# Python One-Liner
# (1) str.count: >>> 'aaabb'.count('a') # 3
# (2) set(str): >>> set('aabbcc') # ['a', 'b', 'c']

# 1.2 REVERSE STRING
def reverse_string(S):
    """
    Time = O(n), Space = O(1)
    """
    if len(S) == 0:
        return S
    A = list(S)
    i, j = 0, len(S) - 1
    while i < j:
        A[i], A[j] = A[j], A[i]
        i += 1
        j -= 1
    return ''.join(A)

# Python One-Liner
# (1) list slicing: >>> list('abcde')[::-1] # ['e', 'd', 'c', 'b', 'a']

# 1.3 PERMUTATION
def is_permutation(S1, S2):
    """
    Depending on the sort complexity
    Time = O(?), Space = O(?)
    """
    S1 = S1.replace(' ', '').lower()
    S2 = S2.replace(' ', '').lower()
    if len(S1) != len(S2) or S1 == S2:
        return False
    return sorted(S1) == sorted(S2)

# Solution
def is_permutation2(S1, S2):
    """
    Time = O(n), Space = O(1)
    """
    if len(S1) != len(S2):
        return False
    for char in S1:
        if S2.find(char) == -1:
            return False
        else:
            S2 = S2.replace(char, '', 1)
    return True

# 1.4 REPLACE SPACE
def replace_space(S, true_length):
    """
    Time = O(n), Space = O(n)
    """
    S = S.rstrip()
    if len(S) == 0:
        return ''
    space_count = S.count(' ')
    res = [' '] * (true_length + space_count * 2) 
    A = list(S)
    j = 0
    for i in range(len(A)):
        if A[i] == ' ':
            res[j], res[j+1], res[j+2] = '%', '2', '0'
            j += 3
        else:
            res[j] = A[i]
            j += 1
    return ''.join(res)

# Python One-Liner
# (1) str split and join: >>> '%20'.join('Mr John Smith  '.split()) # 'Mr%20John%20Smith'
# (2) str replace: >>> 'Mr John Smith  '[:13].replace(' ', '%20') # 'Mr%20John%20Smith'

# 1.5 COMPRESS STRING
def compress_string(S):
    """
    String concatenation takes O(n2) and a lot of spaces
    Time = O(n2), Space = O(?)
    """
    A = list(S)
    compressed = ''
    first, count = A[0], 1
    for i in range(1, len(A)):
        if A[i] == first:
            count += 1
        else:
            compressed += first + str(count)
            first = A[i]
            count = 1 # reset
    compressed += first + str(count)
    return compressed if len(compressed) <= len(S) else S

# Solution
def compress_string1(S):
    """
    Time = O(n), Space = O(n)
    """
    if type(S) is not type(''):
        return None
    if len(S) is 0:
        return ''
    A = list(S)
    output = []
    first = S[0]
    count = 0
    for char in A:
        if char == first:
            count += 1
        else:
            output.append(first)
            output.append(str(count))
            first = char
            count = 1
    output.append(first)
    output.append(str(count))
    if len(S) <= len(output):
        return S
    else:
        return ''.join(output)

# Python One-Liner
# (1) itertools.groupby(): [list(g) for k, g in itertools.groupby('aaabbc')] # [[a, a, a], [b, b], [c]]

# 1.6 ROTATE IMAGE (MATRIX)
def rotate_image(M):
    """
    Time = O(n2), Space = O(1)
    """
    if type(M) is not type([[]]):
        return None
    if len(M) != len(M[0]):
        raise Exception('input must be a N x N matrix')
    N = len(M)
    for layer in range(N):
        i = j = layer
        length = N - layer - 1
        while i < length:
            print(layer, i, j, M)
            d = length - i
            temp = M[i][j]
            # move up
            M[i][j] = M[i + d][j + i]
            # move right
            M[i + d][j + i] = M[d][j + i + d]
            # move down
            M[d][j + i + d] = M[0][j + d]
            # move left
            M[0][j + d] = temp
            # next
            i += 1
    return M

# Solution
def rotate_image2(M, N):
    """
    Time = O(n2), Space = O(1)
    """
    for layer in range(N//2):
        first = layer
        last = N - 1 - layer
        for i in range(first, last):
            offset = i - first
            # save top
            top = M[first][i]
            # left -> top
            M[first][i] = M[last - offset][first]
            # bottom -> left
            M[last - offset][first] = M[last][last - offset]
            # right -> bottom
            M[last][last - offset] = M[i][last]
            # top -> right
            M[i][last] = top
    return M

# Python One-Liner
# M = [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12], [13, 14, 15, 16]]
# zip() >>> zip(*M[::-1]) # [(13, 9, 5, 1), (14, 10, 6, 2), (15, 11, 7, 3), (16, 12, 8, 4)]

# 1.7 SET ZEROS
def set_zeros(M):
    """
    Time = O(MN), Space = O(n)
    """
    m, n = len(M), len(M[0])
    rows = [False] * m
    cols = [False] * n
    for i in range(m):
        for j in range(n):
            if M[i][j] == 0:
                rows[i] = True
                cols[j] = True
    for r in range(m):
        if rows[r]:
            for j in range(n):
                M[r][j] = 0
    for c in range(n):
        if cols[c]:
            for i in range(m):
                M[i][c] = 0
    return M

# 1.8 IS ROTATION
def is_rotation(s1, s2):
    """
    s1 = xy, s2 = yx, s1s1 = xyxy
    xy and yx is always a substring of xyxy
    Time = O(1), Space = O(1)
    """
    def isSubstring(s1, s2):
        return s1.find(s2) > -1

    if type(s1) is not type('') or type(s2) is not type(''):
        raise Exception('input must be strings')
    if len(s2) != len(s1):
        return False
    return isSubstring(s1 + s1, s2)
