

# 競技プログラミング用マイライブラリ [改訂2版]

2019.12 AtamaokaC

（作者が水色のため、同程度かそれ未満の人向けです。）

## 競プロ（AtCoder）用の自作既成コード

何でもそうですが、**よく出てくる典型的な処理**、というのはあるものです。競プロも例外ではありません。こういうものには、**自作の関数／クラスのストック**を持っていると便利なことがあります。<font color="Cyan">僕</font>も少しだけ作って時々使っているので、今回はそれを紹介します。  
（大抵は失敗をきっかけに関数等を追加しているので、紹介する問題例は僕がコンテストで解き損ねたものが多いです。また大部分の関数・クラスはC++版も作っています。）

**注意その1（良くないところ）**  
ライブラリ化は良くないところもあります。そもそもプログラミングの勉強・練習・能力という意味では、やや極論気味ですが**その都度実装できるに越したことはない**です。少なくとも<font color="Cyan">水色</font>程度までなら、というよりABCであれば、あると便利・有利なことはあっても、必須ということはそうはありません。
どちらかというとドーピング／バッドノウハウの類かもしれません。AtCoderの<font color="Red">chokudai</font>社長（「高橋君」）によれば、<font color="Blue">青</font>まではライブラリなしでいいそうです。

（ただ逆に競プロ本番の即興スピード勝負とは違い、アルゴリズムを勉強してコードを改良していく作業は、別の意味の勉強にはとてもなると思います。）

なので別にこういうものを作っておくのがお奨め！とかではなく、こういう処理や書き方もある、こういうものが使える場合もある、くらいに見ていただけると幸いです。それでいつでもすらすら書けるようになれば一番いいです。  
ほか、もっとこうした方がいいよ！とかもあればぜひ教えてください。

**注意その2（バグなど）**  
初版に載せた関数には複数のバグや非効率部分がその後見つかりました。それにより僕自身コンテストでWAやTLEを出してしまったものもあります。それはそれで悪いことばかりでもないのですが、いずれにせよあまり信用はせずに、自分で確認したコードを使いましょう。

## 入力関係

### ベクトル入力関数inpl()

【対象：AtCoder 100点問題（ABC-A）以上】

整数のリストを一行から読み込む関数。  
**殆どの問題に使用する最頻出関数**です。頻出すぎるため僕はテンプレートに入れていますが、いつでも書けることは絶対必要と言ってもよいです。

```python
inpl = lambda: list(map(int,input().split()))
# 使用例：
# A, B, C = inpl()
# x = inpl()  ：xは配列（リスト）になる。
```

### input()関数再定義

【対象：AtCoder 300点問題（ABC-C）以上】

入力が大量に繰り返される場合、標準の`input()`よりもこのように再定義した方が速いようです。内部的に何が違うのかは知りません。

最後の`.rstrip()`が地味に大事で、無くても問題ない場合もあるが、文字列の最後に改行を含めてしまい事故を起こすことがあります。僕はかつてやってしまいました。

```python
import sys
input = lambda: sys.stdin.readline().rstrip()
```

## 割り算丸め関連

【対象：AtCoder 200点問題（ABC-B）以上？ A問題でも出るかもしれない】

### 割り算のceil

「`num/denom`以上で最小の整数」です。これが必要な機会は結構あります。  
なおfloorは単に`//`でOK。四捨五入は`round(num/denom)`。

実際にはこれも必ずしも関数化しなくてよいし、ライブラリ化よりも頭の中に入れておいた方がいいです。これが空気になると上達した気分になれます。応用も色々と効きます。

```python
div_ceil = lambda num, denom: (num-1)//denom + 1
# もちろんmath.ceil()を使っても悪くはないが。。
# from math import ceil
# div_ceil = lambda num, denom: int(ceil(num/denom))
```

## 最大公約数

【対象：AtCoder 300点問題（ABC-C）以上】

###  gcd() 再実装

AtCoderのPython 3.4.3（やや古い）では`gcd()`は`fractions`パッケージにあり、`math`にはないです。
勿論ちゃんと覚えていれば何の問題もないのですが、**いざとなればユークリッド互除法くらい、いつでもさっと実装できる**ことも大事ではあります。短時間でバグなく、は慣れないと難しいものですが。

```python
# from fractions import gcd # これの代わりに
def gcd(a, b):
    while b != 0:
        (a, b) = (b, a % b)
    return a
    # return abs(a) # 符号を気にする場合
```

### 多数の整数の最大公約数 gcd_list()

`gcd()`関数は引数2つだけのため、多数の最大公約数を一気に求めたい場合は`reduce()`でこれを順々に適用していきます。`numbers`は整数の配列。もちろん単にループを回してもよいですが。
（注：本関数は[こちら](https://note.nkmk.me/python-gcd-lcm/)からの借用。実際のところプライオリティが生じるほど高級なものではないが、変数名などここからのコピーなので。）

```python
# from math import gcd : Python 3.5以降
from fractions import gcd
from functools import reduce

def gcd_list(numbers):
    return reduce(gcd, numbers)
```
一度計算するだけなら、これでよいでしょう。繰り返し計算するなどの場合は、色々と工夫の余地がありますが、それは基本的にはコンテスト中に考えればよいです。
（例として、セグメント木を使う暴力的な方法を本記事の最後で紹介します。）

## 配列中連続要素数の取り出し get_sequence()
【対象：AtCoder 300点問題（ABC-C）以上】

`[A,A,A,B,B,A,A]` -> `[A,B,A],[3,2,2]`のように変換する関数です。

```python
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
```

特段重い処理ではない（$\mathcal{O}(N)$）ですが、これを使うとコード全体がやや野暮ったくなる場合は多い気がします。でも**実戦的にはそこそこ便利**なので使ってしまいます。


## mod逆元関数 inv_mod()
【対象：AtCoder 400点問題（ABC-D）以上】

$p$を法とするとき、掛け算による$a$の逆元、すなわち
$$
a \times a^{-1} \equiv 1\;\; (\mathrm{mod}\;p)
$$
となるような$p$未満の正整数 $a^{-1}$ を求める関数。（$a$と$p$は互素であることが必要。）
整数の割り算（割り切れることはわかっている）をしてmodを求める場合に、割り算を$a^{-1}$の掛け算で置き換えることで処理を高速化できます。

日常生活で使うことは比較的少ないと思いますが、AtCoderでは ${}_nC_k \;\mathrm{mod} \;p$ の計算や、「$A/B$の代わりに$A\equiv BR \;(\mathrm{mod} \; p)$なる$R$を出力せよ」などを求められることがままあり、割り算の処理に活躍します。

（関数名について：英語的にはmodを先に置く方が自然です（modular inverse → mod_inv）。ただエディタの補完機能等でmodのついた候補がずらずら出たりするのが微妙なのでmodを後にしています。このあたりはお好みで。）

### 実装その1（再帰関数版）

アルゴリズムは上でも出てきたユークリッド互除法なので高速ですが、少し複雑になるので**再帰関数**を使います。このコードでは再帰関数`inv_mod_sub()`を、インターフェース関数`inv_mod()`の内部関数として定義しています。

```python
# inv_mod()の実装その1（再帰関数版）
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
```

### 実装その2（非再帰版）

上の再帰関数をスタックを使ったループで置き換えたバージョン。特に**PyPyで実行する場合に高速**になります。

```python
# inv_mod()の実装その2（非再帰版）
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
```

### 実装その3（フェルマーの小定理版）

フェルマーの小定理
$$
a^{p-1}\equiv 1\quad (\mathrm{mod}\; p)
$$
（$p$は素数、$a$は$p$の倍数でない任意の整数）は、単純ながら恐ろしい定理。この式を変形すると
$$
a\cdot a^{p-2}\equiv 1\quad (\mathrm{mod}\; p)
$$
となり、よくよく見ると（$a^{-1}$の定義により）
$$
a^{-1} \equiv a^{p-2}\quad (\mathrm{mod}\; p)
$$
であることがわかります。これを愚直に使えば、$a^{-1}$を計算することができます。

実装は下記の通り。この場合、$p$は素数に限られ、理論上の計算量もユークリッド互除法よりもやや大きい（$O(\log p)$）です。ただ所詮はlogな上に組み込み関数に投げる（定数倍は速い）ので、ほぼ気にしなくていいです。また**何といっても実装が簡単**なのがポイントです。丸暗記しておけるレベルですね。

```python
# inv_mod()の実装その3（フェルマーの小定理版）
def inv_mod(a, p=10**9+7): # pは素数であること
    return pow(a, p-2, p)  
```

また$p$がaと互いに素ではあるが素数でない場合、フェルマーの小定理の一般化である**オイラーの定理**を用いて計算が可能です。が、$p$を素因数分解する必要があるので、そこに計算量をとられる場合はお勧めできません。素因数分解器を持ち出すくらいならユークリッド互除法の方が無難ではあります。

```python
# inv_mod()の実装その3'（オイラーの定理版）
def euler_phi(factor, power): # オイラー関数
    phi = 1
    for i in range(len(factor)):
        phi *= (factor[i]-1)*pow(factor[i],power[i]-1)
    return phi

def inv_mod(a, p=10**9+7, phi=None):
    if phi is None:
        phi = p-1
    return pow(a, phi-1, p)

# Usage:
#	fct = Factorizer() # 素因数分解器クラスFactorizerは本稿後の方で定義
#   phi = euler_phi(*fct(p))
#   a_inv = inv_mod(a, p, phi)
```

### 組み合わせ数のmod

上の`inv_mod()`（実装はどれでもよい）を使い、${}_nC_k \;\mathrm{mod} \;p$を一度計算するだけならこれだけです：

```python
def comb_mod(n,k,p=10**9+7):
    ans = 1
    k = min(k,n-k)
    if k < 0:
        return 0
    for i in range(k):
        ans = ans * (n-i) * inv_mod(k-i,p) % p
    return ans
```
階乗の割り算部分を順次掛け算に置き換えることで、その都度$p$の剰余だけを考えればよくなります。巨大な階乗同士の割り算に比べて劇的な高速化が可能です。  
[ABC145-D](https://atcoder.jp/contests/abc145/tasks/abc145_d)はまさにこれを使う問題です（普通に計算すると2秒では絶対できない）。

いくつも計算する場合は`inv_mod()`や階乗を何度も計算するのは無駄なので、表を作ったり累積計算をする方がいいです。

### 冪乗のmod
逆元と関係ないけど、冪乗のmodを求めたい場合、手動で書くとこんな感じ：

```python
def pow_mod(x, n, p):
    ans = 1
    while n > 0:
        if n % 2:
            ans = ans * x % p
        n //= 2
        x = x * x % p
    return ans
```
自乗の自乗の自乗、、、と分解することで、$O(\log n)$の計算量で求められます。
（これ自体は実は標準関数の`pow(x,n,p)`でできる、と<font color="Blue">nadare</font>さんに教えてもらいました。ただアルゴリズムの参考にはなるかと思います。）

## キー関数による二分探索 bisect\_left\_withkey()
【対象：AtCoder 500点問題（ABC-E）以上】

二分探索は`bisect`パッケージで行えるけど、比較関数を渡すオプションがありません。仕方なく自作したのがこちら。[Exawizards2019-C](https://atcoder.jp/contests/exawizards2019/tasks/exawizards2019_c)で作ったものです。（コンテスト時間内には解けませんでしたが。）
他にも[ABC144-E](https://atcoder.jp/contests/abc144/tasks/abc144_e)などがこれを使って解ける問題です。

```python
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
```

## カウンタクラス CountUp
【対象：AtCoder 300点問題（ABC-C）以上】

（ここから後はクラスを使う実装なので、Pythonのクラスに関する知識が若干必要。）

呼ぶごとに`0`,`1`,`2`,...を順番に返す関数オブジェクトになる。`collections.defaultdict`と組み合わせて使うと便利。実際に頻繁に使っています。  

```python
class CountUp:
    def __init__(self, start=0):
        self.index = start-1

    def __call__(self):
        self.index += 1
        return self.index

# 使用例：
# from collections import defaultdict
# ui = defaultdict(CountUp())
```
`ui`は与えた要素に自動的に通し番号をつけてくれます。例えば、

```python
A = [100, 200, 300, 200, 150]
for a in A:
    print(ui[a])
```
とすると、`0 1 2 1 3`の順に出力されます。

[ABC113-C](https://atcoder.jp/contests/abc113/tasks/abc113_c)での使用例は[こちら](https://atcoder.jp/contests/abc113/submissions/8558627)。

## 素数探索 primes() と素因数分解器 Factorizer
【対象：基本的にAtCoder 400点問題（ABC-D）以上。ただし300点でも使える場合がある】

頻出処理。[ABC125-C](https://atcoder.jp/contests/abc125/tasks/abc125_c) で痛い目にあったので、書いておくことにしたもの。
（問題自体は素因数分解よりももっと良い解法があり、解説参照。また後のセグ木のところでもさらに別解を示します。）

### primes()
「N未満の素数を全て求める」関数。計算量は$\mathcal{O}(N\log N)$くらい？

```python
def primes(N):
    P = [2]
    for i in range(3,N,2):
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
```

### Factorizer
こちらは素因数分解器。関数オブジェクトとして使う。

`primes()`と同様のアルゴリズムで素数集合を必要なだけ追加計算して、`self.primes`として（上限`self.primes_ceil`とともに）記憶しておきます。
これを何度も再利用することで、多数の素因数分解を行う場合に計算量を低減できます。
（それでも重い処理なので、TLEの原因にはなりやすいです。**ご利用は計画的に**。）

```python
# 使用例：
# fct = Factorizer()
# print(fct(84))  ->  ([2, 3, 7], [2, 1, 1]) を出力。

from math import sqrt
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
            self.find_primes(int(sqrt(n))+1)
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
```

###  divisors()：約数リスト

与えられた数`n`の約数全部をリストにして返す。ときどき必要になります。

```python
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
```

## UnionFind木

【対象：AtCoder 400点問題（ABC-D）以上】

（ここから先は有名アルゴリズムの実装と、ちょっとした拡張です。<font color="Brown">茶色</font>程度以上かつ、ABCのD問題程度まで解説を読めばわかる、くらいでないと少し難しいかも。）

### UnionFindクラス
UnionFind木により、**グラフの連結成分**を高速に構成・判定するクラスです。
[ABC120-D](https://atcoder.jp/contests/abc120/tasks/abc120_d)で実装が間に合わず、懲りて作りました（僕がコードをストックし始めたきっかけ）。[ABC126-E](https://atcoder.jp/contests/abc126/tasks/abc126_e)では再び類題が出題されました（こちらの方が簡単で、本クラスさえあれば秒殺）。
（なおグラフ頂点の構成が不明な時は`defaultdict`を使えばいい、というのも<font color="Blue">nadare</font>さんに教えてもらった。）

基本的なアルゴリズムは単純ですが、`root()`メソッド中最後の`self.parent[n]=m`がポイント（再帰実装を見てください。非再帰版も処理内容は同じ）。ルートノードを求めて木を辿っていくのですが、答えを返す直前、次回は延々辿る必要がないように、自らをルート直下に付け替えています。

```python
from collections import defaultdict
class UnionFind:
    def __init__(self, N=None):
        if N is None or N < 1:
            self.parent = defaultdict(lambda: -1)
        else:
            self.parent = [-1]*int(N)

    # 非再帰実装
    def root(self, n):
        stack = []
        while n >= 0:
            stack.append(n)
            n = self.parent[n]
        m = stack.pop()
        while stack:
            self.parent[stack.pop()] = m
        return m

#    # root()の再帰実装
#    def root(self, n):
#        if self.parent[n] < 0:
#            return n
#        else:
#            m = self.root(self.parent[n])
#            self.parent[n] = m
#            return m

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
```

### 拡張：ParityUnionFindクラス
こちらは余談に近いですが。

UnionFind木をパリティ付き（与える辺が頂点同士の偶奇関係を持ち、各連結成分内のグラフ頂点を2群に分類する）に拡張したもの。
[ABC126-D](https://atcoder.jp/contests/abc126/tasks/abc126_d)を解く際に作ったものがベースです。（上の`UnionFind`のコードを持っていたからできた。）
[ABC126-E](https://atcoder.jp/contests/abc126/tasks/abc126_d)の拡張版（矛盾発見／全容解明）もこれで解けます。

```python
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
            # 再帰実装。もちろんここも非再帰に直せる。
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

    def groups(self):
        if isinstance(self.parent,list):
            return list(map(lambda x: x<0, self.parent)).count(True)
        else: # self.parent: defaultdict
            return list(map(lambda x: x<0, self.parent.values())).count(True) 
```

## セグメント木
【対象：AtCoder 500点問題（ABC-E）以上】

### セグメント木： SegmentTreeクラス

いわゆる**セグ木** (segment tree)。1次元配列`value`を半分また半分と区切っていった領域ごとの最小値を記憶し、値を更新しながら任意の範囲 $[a,b)$ 内の最小値をいつでも取り出せる（`query(a,b)`）もの。隣同士を比較して勝ち上がっていくトーナメント表を記憶しています。

普通、「配列`value`の最小値を求めろ」と言われたら、

```python
ans = Infinity
for v in value:
	if v < ans:
		ans = v
```
みたいにすると思います。これを優勝者を決めるトーナメントだと思うと、後の人に順番に挑戦していくパラマス式トーナメント。そう思うと少しいびつです。
これに対して普通の公平なトーナメント表を実現しているのがセグ木。一人の対戦数が少ない（突出して多い人がいない）ので、出場者を差し替えた場合の結果予測がパラマス式（最悪$\mathcal{O}(N)$）よりもずっと高速（$\mathcal{O}(\log N)$）に行えます。

概念的に難しいものではないけれど、コンテスト中にバグなくさっと書くのは僕のレベルだと容易でないです。即興フルスクラッチで動作させる方々も当然おられます。クラスライブラリ化するとコード量はどうしても多くなるし、問題に応じた最低限のコードをすぐに書けるのが理想ではあります。

[M-Solutions2019-D](https://atcoder.jp/contests/m-solutions2019/tasks/m_solutions2019_d)は、このセグ木で捻じ伏せることもできます。（最小のオーダーを持つ結節を順番に除いていく。この問題自体は木を辿るだけでも解けるが、グラフに関する条件がない場合でも解けるのが強み。）

```python
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

    def query_index(self, l=0, r=-1): # 再帰実装もあるが、非再帰の方が速い
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
```
各ノードには値でなくインデックスを記憶して、優勝記録（`query()`）だけでなく優勝者が誰か（`query_index()`）、もわかるようにしています。引き分けはインデックスの小さい方の勝ちです。

### セグ木による累積計算：SegAccumCalcクラス

少し改造することで、最大最小に限らず累積計算の結果（全ての和、積、最大公約数、最小公倍数など、**結合則**の成り立つ演算について）を高速に計算するクラスを作れます。こちらの場合は`self.node[]`には値をそのまま書き込みます。

```python
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
```

例えば`sac=SegAccumCalc(A,calc=min)`とすれば、`SegmentTree`と同じになります（最小を与えるインデックスを得ることはできない）。

応用の例として、`sac=SegAccumCalc(A,calc=gcd)`とすれば、全体または任意範囲の最大公約数を計算できます。これを使うと、上の素因数分解で出てきた[ABC125-C](https://atcoder.jp/contests/abc125/tasks/abc125_c)を解くこともできます：

```python
# ABC125-C セグ木による解(1)
#  定義略
N = int(input())
A = inpl()
sac = SegAccumCalc(A,calc=gcd)  # 累積gcd計算機
ans = 0
for i in range(N):
    sac.update(i,None)  # i番目の値を消去
    g = sac.query()    # 全体の最大公約数
    if g > ans:
        ans = g
    sac.update(i,A[i])  # i番目の値を元に戻す
print(ans)
```

```python
# ABC125-C セグ木による解(2)
#  定義略
N = int(input())
A = inpl()
sac = SegAccumCalc(A,calc=gcd)  # 累積gcd計算機
ans = 0
for i in range(N):
    if i == 0:
        g = sac.query(1,N) # i=0: 1番目から(N-1)番目のgcd
    elif i == N-1:
        g = sac.query(0,N-1) # i=N-1: 0番目から(N-2)番目のgcd
    else: # その他のi
        g = gcd(sac.query(0,i),sac.query(i+1,N))
        # i番目を除いた左右でgcdを各々求め、さらにそれらのgcdを求める。
    if g > ans:
        ans = g
print(ans)
```

どちらでも解けます。計算量は解説の模範解答（$\mathcal{O}(N)$）よりもわずかに大きい、$\mathcal{O}(N\log N)$ 程度。  
これも鶏に牛刀の類ではあります。汎用的なアルゴリズムを特別単純な場合に使うと計算量を損するのも、よくあること。ただし、より難しい類題には応用が利きます。

## グラフ最短路探索

【対象：AtCoder 500点問題（ABC-E）以上】

ABCのEやFで最近よく出題されている印象があります。アルゴリズムを3つ（ダイクストラ法、ベルマンフォード法、ワーシャルフロイド法）紹介します。これらを適所で使い分けることが<font color="Blue">青</font>を目指すには必要です。載せているコードは僕が何度もこれらで痛い目に遭って勉強のために書いたものです。

### ダイクストラ法：dijkstra()

与えられた有向グラフの1つの頂点から、他の1つ以上の頂点への最短距離を求める代表的アルゴリズム。「探索途中で出発点から最も近い点までの距離・経路は、その後の探索でも更新されない」ことを利用して、最良優先探索かつ探索の重複を抑える効率的な方法です。普通に幅優先探索するよりも速いです。（[ABC143-E](https://atcoder.jp/contests/abc143/tasks/abc143_e)ではこれを知らず死にました。）  

実はAtCoderでも、（PyPyではだめですが）普通のPython3では**SciPy**を使うことができ、Cythonで（そこそこ）高速に実装された関数[`scipy.sparse.csgraph.dijkstra()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html)が利用可能です。ただ必ず全ての点からのパスを計算するなど、仕様は若干微妙なところもあります。何より（オープンソースとはいえ）ブラックボックス感があまりうれしくないのでやはり一度は実装してみましょう。

ここで紹介する実装は多少のひねりにも備えたもの。距離がただの数ではない場合などにも対応します。汎用性と簡便さを重視して、速度は最速ではありません。上記[ABC143-E](https://atcoder.jp/contests/abc143/tasks/abc143_e)をこれで安直に解くと`after_contest_00`のみTLEとなります（[こちら](https://atcoder.jp/contests/abc143/submissions/8564637)。コンテスト本番分はAC）。（二分木）ヒープ実装の他にセグ木を使う方法などもあります。SciPyの実装はフィボナッチヒープだそうです。

```python
# ダイクストラ法：ヒープ実装
from heapq import heappop, heappush
def dijkstra(start, goal, N, edges,
             get_route = False,
             start_distance=0,
             plus_func=lambda D,d: D+d):
    '''
    goal: int or list or set
    edges: list or dict
        edges[x] = {(y0, distance0),...}
    '''
    q = [(start_distance, start)]
    done = set()
    if isinstance(goal, int):
        goal_set = {goal}
    else:
        goal_set = set(goal)
    ret = [None]*N
    predecessor = [None]*N
    minimum = [None]*N # 無くても動くが、遅い。

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
                p = plus_func(D, d)
                if m is None or m > p:
                    heappush(q, (p, y))
                    minimum[y] = p
                    predecessor[y] = x
    if get_route:
        return ret, predecessor
    else:
        return ret
```

### ベルマンフォード法：bellman_ford()

ベルマンフォード法。距離が正（三角不等式が成立）と保証されない場合、途中段階での最短経路が更新されない保証もないためダイクストラ法は使えません。仕方がないのでループの生じない最大深さ`N-1`まで、力尽くの幅優先探索を行います。負のループが生じて最短距離が発散していないか、最後に確認します。もちろん計算量はダイクストラ法より大きいです。

最近のAtCoderだと[ABC137-E](https://atcoder.jp/contests/abc137/tasks/abc137_e)が、ベルマンフォード法で解ける典型問題です。実際に下のコードを使って一瞬でACできます（[こちら](https://atcoder.jp/contests/abc137/submissions/8790459)。実行時間は最大1800ms近くで、ぎりぎりに近いですが）。

やはり[`scipy.sparse.csgraph.bellman_ford()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.bellman_ford.html)もあり、特別なことがない場合はそちらが使えます。ただこれも癖があり、興味がないところに負のループがあってもエラーを返すようです。競プロでは大抵そういう場合が大事だったりするので、やっぱり実装はできないといけないです。

```python
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
```

### ワーシャルフロイド法：warshall_floyd()

グラフの全ての頂点から全ての頂点への最短距離を求める有名アルゴリズム。距離が正であるほか、距離の足し算に結合則の成立が必要という制約があります。それらが満たされる場合、ダイクストラ法を繰り返すよりも高速です。  
（上記[ABC143-E](https://atcoder.jp/contests/abc143/tasks/abc143_e)は安直に解こうとすると結合則が成立しない例。ただし解説解はそれでもワーシャルフロイド法で鮮やかに解いていて必見です。）

こちらも一応、実装を示します。上と同じく距離が数でない（タプルなど）場合に対応します（足し算`plus_func()`は結合則を満たすほか、`D+d`と`D1+D2`の両方に対応する必要がある）。

特別なことがない場合は、SciPyの[`scipy.sparse.csgraph.floyd_warshall()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.floyd_warshall.html)を使う方が（PyPy不可でも）2倍くらい速いようです。実戦でSciPyを使える可能性が今回挙げた3つのアルゴリズムの中ではおそらく一番高いです。

```python
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
```

## 木探索一般

【対象：AtCoder 400点問題（ABC-D）以上】

深さ優先探索（DFS）や幅優先探索（BFS）など基本的な木探索は400点程度から出てきますが、変則的なものも多くライブラリ化の意味がほぼありません。このあたりはその都度一から書いてもコード量は大したことがなく、何より勉強・鍛錬になります。

300点（ABC-C）から登場する動的計画法（DP）も同様です。さらに言えば200点（ABC-B）からのループ処理も同じです。要はループに近いレベルのもの（に感じられることを目指したいもの）と思いましょう。そこまでいけば<font color="Green">緑</font>は余裕です。複雑な場合も自由に使えれば<font color="Cyan">水色</font>や<font color="Blue">青</font>になれます。

```python
# 幅優先探索（BFS）：深さの浅いノードから順に全てを出力
# dequeに代えてただのリストだと深さ優先探索になる。
pool = collections.deque([0]) # ルートノードは0
while pool:
    node = pool.popleft()
    print(node)
   	for c in children[node]:
        pool.append(c)
```

## 組み合わせ列挙 combinations()

$_n C_k$の中身を順に列挙します。

```python
for c in combinations(n,k):
    print(c)
```

のように使用可能。`c` は`k`成分のタプルとなります。各成分の値は`0`から`n-1`です。

実装は以下の通り。元々は「競プロで`yield`を使ってみる」目的で作ったもの。

```python
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
```

## おしまい

おしまい。

## 更新履歴
- ver 1.0: 2019.06

- ver 1.1: `Counter`を`CountUp`に改称。
  `collections.Counter`と名前が重複するので、`from collections import Counter`と万一同時使用すると事故になる。

- ver 1.2: `gcd_list()`について注記。

- ver 2.0 (2019.12):
  - 前文を微修正、注意その2を追加
  - 各節に対象となる問題レベルの目安（AtCoderの配点）を追記
  - `input()`再定義を追加
  - 割り算丸め関連の節を追加
  - `get_sequence()`の実装を微修正（`arr`のコピーが発生しないようにした）
  - `inv_mod()`の実装2、3、3'を追加。`comb_mod()`のバグ修正（`k<0`, `k>N`に対応）
  - `bisect_{left,right}_withkey()`のバグ修正
  - `divisors()`を追加
  - `UnionFind`, `ParityUnionFind`, `SegmentTree`, `SegAccumCalc`の実装やメソッド名を微修正。特にセグ木の集約を非再帰実装に変更した。
  - グラフ探索と木探索の節を追加
  - ver1.2以後のコンテストから例題を若干追加
  - ライブラリ自体は半年間でほぼ増えていない。無くていいかは別としても、**ABCにはこれくらいで十分**、と考えてよいと思う。
  
- ver 2.1 (2020.02):

  `combinations()`の追加、グラフ探索関数のコードへの追加。
