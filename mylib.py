from collections import defaultdict
import bisect

import sys
input = sys.stdin.readline
inpl = lambda: list(map(int,input().split()))

class CountUp:
    def __init__(self, start=0):
        self.index = start-1

    def __call__(self):
        self.index += 1
        return self.index

ui = defaultdict(CountUp())

class UnionFind:
    def __init__(self, N=None):
        if N is None or N < 1:
            self.parent = defaultdict(lambda: -1)
        else:
            self.parent = [-1]*int(N)

    def root(self, n):
        if self.parent[n] < 0:
            return n
        else:
            m = self.root(self.parent[n])
            self.parent[n] = m
            return m

    def merge(self, m, n):
        rm = self.root(m)
        rn = self.root(n)
        if rm != rn:
            if -self.parent[rm] < -self.parent[rn]:
                rm, rn = rn, rm
            self.parent[rm] += self.parent[rn]
            self.parent[rn] = rm

    def size(self, n):
        return -self.parent[self.root(n)]
    
    def connected(self, m, n):
        return self.root(m) == self.root(n)

    def groups_num(self):
        if isinstance(self.parent,list):
            return len(list(filter(lambda x: x<0, self.parent)))
        else: # self.parent: defaultdict
            return len(list(filter(lambda x: x<0, self.parent.values())))

    def elements(self):
        if isinstance(self.parent,list):
            return range(len(self.parent))
        else:
            return self.parent.keys()


from collections import defaultdict
class ParityUnionFind:
    def __init__(self, N=None):
        if N is None or N < 1:
            self.parent = defaultdict(lambda: -1)
            self.parity = defaultdict(int)
        else:
            self.parent = [-1]*int(N)
            self.parity = [0]*int(N)

    def root(self, n):
        if self.parent[n] < 0:
            return n, 0
        else:
            m, p = self.root(self.parent[n])
            self.parent[n] = m
            self.parity[n] = (p + self.parity[n]) % 2
            return m, self.parity[n]

    def merge(self, m, n, parity):
        rm, pm = self.root(m)
        rn, pn = self.root(n)
        if rm != rn:
            if -self.parent[rm] < -self.parent[rn]:
                rm, rn = rn, rm
            self.parent[rm] += self.parent[rn]
            self.parent[rn] = rm
            self.parity[rn] = (pm + pn + parity) % 2
            return True
        elif (pm + pn + parity) % 2 == 0:
            return True
        else:
            return False

    def size(self, n):
        return -self.parent[self.root(n)[0]]
    
    def connected(self, m, n):
        return self.root(m)[0] == self.root(n)[0]
    
    def samesign(self, m, n):
        return self.root(m) == self.root(n)

    def diffsign(self,m,n):
        rm, pm = self.root(m)
        rn, pn = self.root(n)
        return rm == rn and pm != pn

# bisect functions with key
def bisect_left_withkey(a, x, lo=0, hi=None, key=lambda x: x, keyvalue_x=False):
    left = lo
    right = len(a) if hi is None else hi
    kx = x if keyvalue_x else key(x)
    while left + 1 < right:
        new = (left + right)//2
        kn = key(a[new])
        if kx <= kn:
            right = new
        else:
            left = new
    return right

def bisect_right_withkey(a, x, lo=0, hi=None, key=lambda x: x, keyvalue_x=False):
    left = lo
    right = len(a) if hi is None else hi
    kx = x if keyvalue_x else key(x)
    while left + 1 < right:
        new = (left + right)//2
        kn = key(a[new])
        if kx < kn:
            right = new
        else:
            left = new
    return right


from fractions import gcd
from functools import reduce
# from math import gcd
def gcd_list(numbers):
    return reduce(gcd, numbers)

def my_gcd(a, b):
    if a < 0: a = -a
    if b < 0: b = -b
    while b > 0:
        (a, b) = (b, a % b)
    return a


def get_sequence(arr):
    if len(arr) == 0:
        return [], []
    elements = []
    nums = []
    n = 1
    prev = arr[0]
    for c in arr[1:]:
        if c == prev:
            n += 1
        else:
            elements.append(prev)
            nums.append(n)
            prev = c
            n = 1
    elements.append(prev)
    nums.append(n)
    return elements, nums


def primes(N):
    P = []
    for i in range(2,N):
        sieve = 1
        for p in P:
            if p*p > i:
                break
            elif i % p == 0:
                sieve = 0
                break
        if sieve == 0:
            continue
        else:
            P.append(i)
    return P

import math
def factorize(n, P=None):
    if P is None:
        P = primes(int(math.sqrt(n))+1)
    factor = []
    power = []
    for p in P:
        if p * p > n:
            break
        elif n % p == 0:
            k = 0
            while n % p == 0:
                n //= p
                k += 1
            factor.append(p)
            power.append(k)
    if n > 1:
        factor.append(n)
        power.append(1)
    return factor, power

def divisors(n, sort=True, reverse=False):
    sqn = math.sqrt(n)
    sqn_int = int(sqn)
    div_list = []
    if sqn == sqn_int:
        div_list.append(sqn_int)
        loop = range(1,sqn_int)
    else:
        loop = range(1,sqn_int + 1)
    for i in loop:
        if n % i == 0:
            div_list.append(i)
            div_list.append(n//i)
    if sort:
        div_list.sort(reverse=reverse)
    return div_list

import math
class Factorizer:
    def __init__(self):
        self.primes_ceil = 5
        self.primes = [2,3]

    def find_primes(self, primes_ceil):
        if primes_ceil <= self.primes_ceil:
            return self.primes

        prev_primes_ceil, self.primes_ceil = self.primes_ceil, primes_ceil
        r, m = prev_primes_ceil//6, prev_primes_ceil%6
        if m <= 1:
            i = r*6 + 1
            mod_flag = True
        else:
            i = r*6 + 5
            mod_flag = False

        while i < self.primes_ceil:
            sieve = True
            for p in self.primes:
                if p*p > i:
                    break
                elif i % p == 0:
                    sieve = False
                    break
            if sieve:
                self.primes.append(i)
            if mod_flag:
                i += 4
                mod_flag = False
            else:
                i += 2
                mod_flag = True

        return self.primes

    def __call__(self, n):
        if (self.primes_ceil-1)**2 < n:
            self.find_primes(int(math.sqrt(n))+1)
        factors = []
        powers = []
        for p in self.primes:
            if p * p > n:
                break
            elif n % p == 0:
                n //= p
                k = 1
                while n % p == 0:
                    n //= p
                    k += 1
                factors.append(p)
                powers.append(k)
        if n > 1:
            factors.append(n)
            powers.append(1)
        return factors, powers


def inv_mod(a, p=10**9+7):
    def inv_mod_sub(a, p):
        if a == 1:
            return 1, 0
        else:
            d, r = p//a, p%a
            x, y = inv_mod_sub(r, a)
            return y-d*x, x
    if p < 0: p = -p
    a %= p
    return inv_mod_sub(a,p)[0] % p

def comb_mod(n,k,p):
    ans = 1
    k = min(k,n-k)
    for i in range(k):
        ans = ans * (n-i) * inv_mod(k-i,p) % p
    return ans

def pow_mod(x, n, p):
    ans = 1 % p
    while n > 0:
        if n % 2:
            ans = ans * x % p
        n //= 2
        x = x * x % p
    return ans

class SegmentTree:
    def __init__(self, value, N=0, comp=lambda x,y: x<=y, reverse=False):
        M = max(len(value),N)
        N = 2**(len(bin(M))-3)
        if N < M: N *= 2
        self.N = N
        self.node = [0] * (2*N-1)
        for i in range(N):
            self.node[N-1+i] = i
        self.value = [None] * N
        for i, v in enumerate(value):
            self.value[i] = v
        self.comp = lambda x, y: True if y is None else False if x is None else comp(x,y)^reverse
        for i in range(N-2,-1,-1):
            left_i, right_i = self.node[2*i+1], self.node[2*i+2]
            left_v, right_v = self.value[left_i], self.value[right_i]
            if self.comp(left_v, right_v):
                self.node[i] = left_i
            else:
                self.node[i] = right_i

    def set_input(self, n, v):
        self.value[n] = v
        i = (self.N-1) + n
        while i > 0:
            i = (i-1)//2
            left_i, right_i = self.node[2*i+1], self.node[2*i+2]
            left_v, right_v = self.value[left_i], self.value[right_i]
            if self.comp(left_v, right_v):
                new_i = left_i
            else:
                new_i = right_i
            if new_i == self.node[i] and new_i != n:
                break
            else:
                self.node[i] = new_i

    def get_input(self, n):
        if n is None:
            return None
        else:
            return self.value[n]

    def get_output(self, a=0, b=-1):
        return self.get_input(self.get_output_index(a,b))

    def get_output_index(self,
                         a=0, b=-1,
                         k=0, l=0, r=-1):
        if b < 0:
            b = self.N
        if r < 0:
            r = self.N
        if a <= l and r <= b:
            return self.node[k]
        elif r <= a or b <= l:
            return None
        else:
            left_i = self.get_output_index(a,b,2*k+1,l,(l+r)//2)
            right_i = self.get_output_index(a,b,2*k+2,(l+r)//2,r)
            left_v, right_v = self.get_input(left_i), self.get_input(right_i)
            if left_v is None and right_v is None:
                return None
            elif self.comp(left_v, right_v):
                return left_i
            else:
                return right_i

class SegAccumCalc:
    def __init__(self, value, N=0, calc=lambda x,y: x+y):
        M = max(len(value),N)
        N = 2**(len(bin(M))-3)
        if N < M: N *= 2
        self.N = N
        self.node = [None] * (2*N-1)
        for i, v in enumerate(value):
            self.node[i+N-1] = v
        self.calc = lambda x, y: x if y is None else y if x is None else calc(x,y)
        for i in range(N-2,-1,-1):
            left, right = self.node[2*i+1], self.node[2*i+2]
            self.node[i] = self.calc(left, right)

    def set_input(self, n, v):
        i = (self.N-1) + n
        self.node[i] = v
        while i > 0:
            i = (i-1)//2
            left, right = self.node[2*i+1], self.node[2*i+2]
            new = self.calc(left,right)
            if self.node[i] == new:
                break
            else:
                self.node[i] = new
    
    def get_input(self, n):
        return self.node[(self.N-1)+n]

    def get_output(self,
                   a=0, b=-1,
                   k=0, l=0, r=-1):
        if b < 0:
            b = self.N
        if r < 0:
            r = self.N
        if a <= l and r <= b:
            return self.node[k]
        elif r <= a or b <= l:
            return None
        else:
            left = self.get_output(a,b,2*k+1,l,(l+r)//2)
            right = self.get_output(a,b,2*k+2,(l+r)//2,r)
            return self.calc(left,right)

## Incomplete Codes... ###
class LinkedList:
    def __init__(self, record, next=None):
        self.car = record
        self.cdr = next

    @classmethod
    def fromList(cls, arr, start=0):
        if start < len(arr)-1:
            return None
        else:
            return cls(arr[start],cls.fromList(arr,start+1))

    def last(self):
        if self.cdr == None:
            return self
        else:
            return self.cdr.last()

    def push(self, v):
        self.last().cdr = LinkedList(v)
        return v
    
#    def pop(self):
