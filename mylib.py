from collections import defaultdict
import bisect

import sys
input = lambda: sys.stdin.readline().rstrip()
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

    def root_rec(self, n):
        if self.parent[n] < 0:
            return n
        else:
            m = self.root(self.parent[n])
            self.parent[n] = m
            return m

    def root(self, n):
        stack = []
        while n >= 0:
            stack.append(n)
            n = self.parent[n]
        m = stack.pop()
        while stack:
            self.parent[stack.pop()] = m
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

    def groups(self):
        if isinstance(self.parent,list):
            return list(filter(lambda i: self.parent[i]<0, range(len(self.parent))))
        else: # self.parent: defaultdict
            return list(filter(lambda i: self.parent[i]<0, self.parent.keys()))
 
    def groups_num(self):
        return len(self.groups())

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


class MinMaxUnionFind:
    def __init__(self, N):
        self.parent = [-1]*int(N)
        self.grouprange = [(i,i) for i in range(N)]

    def root(self, n):
        stack = []
        while n >= 0:
            stack.append(n)
            n = self.parent[n]
        m = stack.pop()
        while stack:
            self.parent[stack.pop()] = m
        return m

    def merge(self, m, n):
        rm = self.root(m)
        rn = self.root(n)
        if rm != rn:
            if -self.parent[rm] < -self.parent[rn]:
                rm, rn = rn, rm
            self.parent[rm] += self.parent[rn]
            self.parent[rn] = rm
            self.grouprange[rm] = (
                min(self.grouprange[rm][0],self.grouprange[rn][0]),
                max(self.grouprange[rm][1],self.grouprange[rn][1]),
            )

    def min(self, n):
        return self.grouprange[self.root(n)][0]
    
    def max(self, n):
        return self.grouprange[self.root(n)][1]

    def size(self, n):
        return -self.parent[self.root(n)]
    
    def connected(self, m, n):
        return self.root(m) == self.root(n)

    def groups(self):
        return list(filter(lambda i: self.parent[i]<0, range(len(self.parent))))
 
    def groups_num(self):
        return len(self.groups())

    def elements(self):
        return range(len(self.parent))


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
    if key(a[left]) == kx:
        return left
    else:
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
    for i in range(1,len(arr)):
        c = arr[i]
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

from math import sqrt, ceil
def divisors(n, sort=True, reverse=False):
    sqn = int(ceil(sqrt(n)))
    div_list = []
    if sqn*sqn == n:
        div_list.append(sqn)
    for i in range(1,sqn):
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


def inv_mod_rec(a, p=10**9+7):
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

def inv_mod(a, p=10**9+7):
    p = abs(p)
    a %= p
    stack = []
    p0 = p
    while a > 1:
        d, a, p = p//a, p%a, a
        stack.append(d)
    x, y = 1, 0
    while stack:
        d = stack.pop()
        x, y = y-d*x, x
    return x % p0

def comb_mod(n,k,p):
    ans = 1
    k = min(k,n-k)
    if k < 0:
        return 0
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
    def __init__(self, value=[], N=0, comp=lambda x,y: x<=y, reverse=False):
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
            self.node[i] = left_i if self.comp(left_v, right_v) else right_i

    def __setitem__(self, n, v):
        self.update(n,v)

    def __getitem__(self, n):
        return self.at(n)

    def update(self, n, v):
        self.value[n] = v
        i = (self.N-1) + n
        while i > 0:
            i = (i-1)//2
            left_i, right_i = self.node[2*i+1], self.node[2*i+2]
            left_v, right_v = self.value[left_i], self.value[right_i]
            new_i = left_i if self.comp(left_v, right_v) else right_i
            if new_i == self.node[i] and new_i != n:
                break
            else:
                self.node[i] = new_i

    def at(self, n):
        if n is None:
            return None
        else:
            return self.value[n]

    def query(self, l=0, r=-1):
        return self.at(self.query_index(l,r))

    def query_index(self, l=0, r=-1):
        if r < 0: r = self.N
        L = l + self.N; R = r + self.N
        s = None
        while L < R:
            if R & 1:
                R -= 1
                if self.comp(self.at(self.node[R-1]), self.at(s)):
                    s = self.node[R-1]
            if L & 1:
                if self.comp(self.at(self.node[L-1]), self.at(s)):
                    s = self.node[L-1]
                L += 1
            L >>= 1; R >>= 1
        return s

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

    def __setitem__(self, n, v):
        self.update(n,v)

    def __getitem__(self, n):
        return self.at(n)

    def update(self, n, v):
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
    
    def at(self, n):
        return self.node[(self.N-1)+n]

    def query(self, l=0, r=-1):
        if r < 0: r = self.N
        L = l + self.N; R = r + self.N
        s = None
        while L < R:
            if R & 1:
                R -= 1
                s = self.calc(self.node[R-1], s)
            if L & 1:
                s = self.calc(self.node[L-1], s)
                L += 1
            L >>= 1; R >>= 1
        return s


from heapq import heappop, heappush
from collections import defaultdict
def dijkstra(start, goal, edges,
             start_distance=0,
             plus_func=lambda D,d: D+d):
    '''
    goal: int or list or set
    edges: list or dict
        edges[x] = (y, distance)
    '''
    q = [(start_distance, start)]
    done = set()
    if isinstance(goal, int):
        goal_set = {goal}
    else:
        goal_set = set(goal)
    ret = defaultdict(lambda: None)
    minimum = defaultdict(lambda: None)

    while q:
        D, x = heappop(q)
        if x in done:
            continue
        elif x in goal_set:
            ret[x] = D
            goal_set.remove(x)
            if len(goal_set) == 0:
                return ret
        done.add(x)
        for y, d in edges[x]:
            if not y in done:
                m = minimum[y]
                p = plus_func(D,d)
                if m is None or m > p:
                    heappush(q, (p, y))
                    minimum[y] = p
    return ret

def bellman_ford(start, goal, N, edges,
                 get_route=False,
                 negative_loop=None,
                 start_distance=0,
                 plus_func=lambda D,d: D+d):
    '''
    goal: int or list or set
    edges: list or dict
        edges[x] = {(y0, distance0), ...}
    '''
    distance = [None]*N
    predecessor = [None]*N
    if isinstance(goal, int):
        goal = [goal]
    elif isinstance(goal, set):
        goal = list(goal)
    M = len(goal)
    goalable = [[False]*N for _ in range(M)]
    for m in range(M):
        goalable[m][goal[m]] = True
    distance[start] = start_distance
    for _ in range(N-1):
        for i in range(N):
            if distance[i] is None:
                continue
            for j, d in edges[i]:
                for m in range(M):
                    if goalable[m][j]:
                        goalable[m][i] = True
                p = plus_func(distance[i], d)
                if distance[j] is None or distance[j] > p:
                    distance[j] = p
                    predecessor[j] = i

    goal_valid = [True]*M
    for i in range(N):
        if distance[i] is None:
            continue
        for j, d in edges[i]:
            if distance[j] is not None and plus_func(distance[i], d) < distance[j]:
                # Negative loop detected.
                for m in range(M):
                    if goal_valid[m] and goalable[m][i]:
                        distance[goal[m]] = negative_loop
                        goal_valid[m] = False
    if get_route:
        return distance, predecessor
    else:
        return distance

# scipy.sparse.csgraph.floyd_warshall()
def warshall_floyd(N, edges,
                   nopath=None,
                   get_route=False,
                   start_distance=0,
                   plus_func=lambda D,d: D+d):
    '''
    edges: list of set
        edges = [{(y0, distance0), (y1, distance1), ...}, ...]
    '''
    ret = [[nopath]*N for _ in range(N)]
    predecessor = [[None]*N for _ in range(N)]
    for i in range(N):
        ret[i][i] = start_distance
    for i in range(N):
        for j, d in edges[i]:
            if 0 <= j < N:
                ret[i][j] = plus_func(start_distance, d)
                predecessor[i][j] = i
    for k in range(N):
        for i in range(N):
            if i == k: continue
            for j in range(N):
                if j == k or ret[i][k] == nopath or ret[k][j] == nopath:
                    continue
                d = plus_func(ret[i][k], ret[k][j])
                if ret[i][j] == nopath or d < ret[i][j]:
                    ret[i][j] = d
                    predecessor[i][j] = predecessor[k][j]
    if get_route:
        return ret, predecessor
    else:
        return ret


def combinations(N,k):
    if 0 <= k <= N:
        c = [0]*k
        pool = [(-1,-1)]
        while pool:
            i, n = pool.pop()
            if i >= 0:
                c[i] = n
            if i >= k-1:
                yield tuple(c)
            else:
                for m in range(i+N-k+1,n,-1):
                    pool.append((i+1, m))

def permutations(N, k=-1):
    if k < 0:
        k = N
    if 0 <= k <= N:
        c = [0]*k
        pool = [(-1,-1)]
        S = set(range(N))
        d = -1
        while pool:
            i, n = pool.pop()
            if i >= 0:
                if i <= d:
                    S |= set(c[i:d+1])
                d = i
                c[d] = n
                S.remove(n)
            if i >= k-1:
                yield tuple(c)
            else:
                for m in S:
                    pool.append((d+1, m))


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


if __name__ == '__main__':
    #print(divisors(384,reverse=True))
#    sg = SegmentTree([1,2,-4,3,4],reverse=False)
#    print(sg.query(0,-1))

    for i in range(1,15):
        print(inv_mod_rec(i), inv_mod(i))