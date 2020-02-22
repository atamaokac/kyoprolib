//#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <map>
//#include <boost/function.hpp>
#include <functional>
#include <cmath>
typedef long long int llint;
typedef unsigned long long int ullint;
std::string YesNo[] = {"No","Yes"};
#define loop(i,N) for(int i=0; i<(N); i++)
#define sort_all(v) std::sort(v.begin(),v.end())
#define sum_all(v) std::accumulate(v.begin(),v.end(),0) // For int. For float, 0 -> 0.0
template<typename T>
auto input(int N) {
    std::vector<T> v(N);
    loop(i,N) std::cin >> v[i];
    return std::move(v);
}

template<typename T>
auto gcd(T a, T b) {
    while (b > 0) {
        T tmp = a % b; a = b; b = tmp;
    }
    return a;
}


template<typename K, typename V>
class defaultmap: public std::map<K,V>
{
private:
    std::function<V(K)> defaultfunc;
public:
    defaultmap(std::function<V(K)> df) : defaultfunc(df) {}
    auto& operator[](const K& key) {
        auto itr = this->find(key);
        if (itr != this->end())
            return itr->second;
        else
            return this->std::map<K,V>::operator[](key)=defaultfunc(key);
    }
};
class CountUp
{
private:
    int index;
public:
    CountUp(int start=0) : index(start) {}
    auto operator()(...) { return index++; }
    auto next() { return index; }
};
/* Usage of CountUp:
    auto ui = defaultmap<K, int>(CountUp());
*/

int bitlen(unsigned long long int n)
{
    int total = sizeof(n) * 8;
    int l = total, r = -1, mid;
    while (l > r+1) {
        mid = (l+r)/2;
        if (n >> mid == 0) {
            l = mid;
        } else {
            r = mid;
        }
    }
    return l;
}
template<typename T>
class SegmentTree
{
private:
    unsigned int N;
    std::vector<int> node;
    std::vector<T> data;
    std::function<bool(int,int)> comp;
    const std::function<bool(int,int)> defaultcomp = [this](int i, int j) {
        if (j < 0) {
            return true;
        } else if (i < 0) {
            return false;
        } else {
            return data[i] < data[j];
        } 
    };

public:
    SegmentTree(const std::vector<T>& data,
                unsigned int N_=0,
                bool reverse = false,
                std::function<bool(int,int)> comp = nullptr
                ) : data(data)
    {
        if (comp == nullptr) {
            comp = defaultcomp;
        }
        if (reverse) {
            this->comp = [comp] (int i, int j) { return comp(j,i); };
        } else {
            this->comp = [comp] (int i, int j) { return comp(i,j); };
        }
        
        unsigned int M = std::max((unsigned int)(data.size()), N_);
        N = 1 << std::max(bitlen(M)-1, 0);
        if (N < M) N *= 2;
        node.resize(2*N-1);
        for (auto i=0; i<data.size(); i++) {
            node[N-1+i] = i;
        }
        for (auto i=data.size(); i<N; i++) {
            node[N-1+i] = -1;
        }

        for (int i=N-2; i>=0; i--) {
            auto left_i = node[2*i+1], right_i = node[2*i+2];
            node[i] = this->comp(left_i, right_i) ? left_i : right_i;
        }
    }

    const T& at(int i) { return data[i]; }

    void update(int n, const T& v) {
        data[n] = v;
        int i = N + n - 1;
        while (i > 0) {
            i = (i-1)/2;
            int left_i, right_i = node[2*i+1], node[2*i+2];
            auto new_i = comp(left_i, right_i) ? left_i : right_i;
            if (new_i == node[i] and new_i != n) {
                break;
            } else {
                node[i] = new_i;
            }
        }
    }

    const T& query(int l=0, int r=-1) { return data[query_index(l, r)]; }

    int query_index(int l=0, int r=-1) {
        if (r < 0) r = N;
        int L = l + N, R = r + N;
        int s = -1;
        while (L < R) {
            if (R & 1) {
                R -= 1;
                if (comp(node[R-1], s)) s = node[R-1];
            }
            if (L & 1) {
                if (comp(node[L-1], s)) s = node[L-1];
                L += 1;
            }
            L >>= 1; R >>= 1;
        }
        return s;
    }
};

template<typename Int>
auto inv_mod_norec(Int a, Int p=1000000007)
{
    p = std::abs(p);
    a %= p;
    std::vector<Int> stack;
    auto p0 = p;
    while (a > 1) {
        stack.push_back(p/a);
        const auto new_a = p % a;
        p = a;
        a = new_a;
    }
    Int x = 1, y = 0;
    while (stack.size()) {
        const auto new_x = y - x*stack.back();
        stack.pop_back();
        y = x;
        x = new_x;
    }
    x %= p0;
    if (x < 0) x += p0;
    return x;
}

template<typename T>
auto inv_mod(T a, T p) {
    struct {
        auto operator()(T a, T p) {
            if(a == 1) {
                return std::pair<T,T>(1, 0);
            } else {
                auto d = p / a;
                auto r = p % a;
                auto xy = operator()(r, a);
                auto x = xy.first;
                auto y = xy.second;
                return std::pair<T,T>(y-d*x, x);
            }
        }
    } inv_mod_sub;

    if(p < 0) p = -p;
    a %= p;
    auto ans = inv_mod_sub(a, p).first % p;
    if(ans < 0) ans += p;
    return ans;
}

template<typename Int>
auto pow_mod(Int x, Int n, Int p) {
    if(p == 1 || p == -1) {
        return 0;
    }
    Int ans = 1;
    while(n > 0) {
        if(n % 2) ans = ans * x % p;
        n /= 2;
        x = x * x % p;
    }
    return ans;
}


class UnionFind
{
private:
    typedef std::vector<int> P;
    P parent;
    // P el;
    int gN;
public:
    UnionFind(int N) : parent(N,-1), gN(N) {}

    auto root(int n) {
        if(parent[n] < 0) {
            return n;
        } else {
            auto m = root(parent[n]);
            parent[n] = m;
            return m;
        }
    }

    auto merge(int m, int n) {
        auto rm = root(m);
        auto rn = root(n);
        if(rm != rn) {
            if(parent[rm] > parent[rn]) {
                auto tmp = rm; rm = rn; rn = tmp;
            }
            parent[rm] += parent[rn];
            parent[rn] = rm;
            gN--;
        }
        return rm;
    }

    auto size(int n) { return -parent[root(n)]; }
    auto connected(int m, int n) { return root(m) == root(n); }
    auto groups_num() { return gN; }
    auto elements_num() { return parent.size(); }
    // const auto& elements() { return el; }
};

template<typename T>
class Factorizer
{
private:
    T primes_ceil;
    std::vector<T> primes;
public:
    Factorizer() : primes_ceil(5), primes({2,3}) {}

    const auto& find_primes(T pc) {
        if(pc <= primes_ceil)
            return primes;

        T prev_primes_ceil = primes_ceil;
        primes_ceil = pc;
        T r = prev_primes_ceil / 6;
        T m = prev_primes_ceil % 6;
        T i;
        bool mod_flag;
        if(m <= 1) {
            i = r*6 + 1;
            mod_flag = true;
        } else {
            i = r*6 + 5;
            mod_flag = false;
        }

        while(i < primes_ceil) {
            bool sieve = true;
            for(auto p : primes)
                if(p*p > i)
                    break;
                else if(i % p == 0) {
                    sieve = false;
                    break;
                }
            if(sieve) primes.emplace_back(i);
            if(mod_flag) {
                i += 4;
                mod_flag = false;
            } else {
                i += 2;
                mod_flag = true;
            }
        }
        return primes;
    }

    auto operator()(T n) {
        if((primes_ceil-1)*(primes_ceil-1) < n)
            find_primes(int(std::sqrt(n))+1);
        std::vector<T> factors, powers;
        for(auto p : primes)
            if(p * p > n)
                break;
            else if(n % p == 0) {
                n /= p;
                T k = 1;
                while(n % p == 0) {
                    n /= p;
                    k ++;
                }
                factors.emplace_back(p);
                powers.emplace_back(k);
            }
        if(n > 1) {
            factors.emplace_back(n);
            powers.emplace_back(1);
        }
        return std::move(std::pair<std::vector<T>,std::vector<T>>(
                    std::move(factors), std::move(powers)
                ));
    }
};


template<typename T>
class heapq : public std::vector<T>
{
private:
    void down_heapify(std::size_t i) {
        auto m = i;
        auto first_leaf = this->size()/2;
        while(i < first_leaf) {
            auto left = i*2 + 1;
            auto right = i*2 + 2;
            for(auto k : {left, right}) {
                if(k < this->size() && (*this)[k] < (*this)[m]) {
                    m = k;
                }
            }
            if(m == i) {
                return;
            } else {
                std::swap((*this)[i], (*this)[m]);
                i = m;
            }
        }
    }

    void up_heapify(std::size_t i) {
        while(i > 0) {
            auto parent = (i-1)/2;
            if((*this)[i] < (*this)[parent]) {
                std::swap((*this)[i], (*this)[parent]);
                i = parent;
            } else {
                return;
            }
        }
    }

public:
    template<typename ...Args>
    heapq(Args ...args) : std::vector<T>(args...) {
        heapify();
    }

    void heapify() {
        auto first_leaf = this->size()/2;
        for(auto i=first_leaf; i>0; i--) {
            down_heapify(i-1);
        }
    }

    void heappush(const T& x) {
        this->push_back(x);
        up_heapify(this->size()-1);
    }

    T heappop() {
        auto ret = std::move(this->front());
        this->front() = std::move(this->back());
        this->pop_back();
        down_heapify(0);
        return ret;
    }

    T heappoppush(const T& x) {
        auto ret = std::move(this->front());
        this->front() = x;
        down_heapify(0);
        return ret;        
    }

    T heappushpop(const T& x) {
        if(x <= this->front()) {
            return x;
        } else {
            return heappoppush(x);
        }
    }
};


template <typename T>
auto get_sequence(const std::vector<T>& array) {
    if (array.size() == 0) {
        return std::make_pair(std::vector<T>(0), std::vector<int>(0));
    }
    std::vector<T> elements;
    std::vector<int> nums;
    int n = 1;
    auto prev = array.begin();
    for (auto itr = prev+1; itr != array.end(); ++itr) {
        if (*itr == *prev) {
            n ++;
        } else {
            elements.push_back(*prev);
            nums.push_back(n);
            prev = itr;
            n = 1;
        }
    }
    elements.push_back(*prev);
    nums.push_back(n);
    return std::make_pair(std::move(elements), std::move(nums));
}

template <typename T>
auto divisors(T n, bool sort=true, bool reverse=false)
{
    auto sqn = T(std::sqrt(n));
    std::vector<T> div_list;
    T loopto;
    if (sqn*sqn == n) {
        div_list.push_back(sqn);
        loopto = sqn;
    } else {
        loopto = sqn + 1;
    }
    for (T i=1; i < loopto; i++) {
        if (n % i == 0) {
            div_list.push_back(i);
            div_list.push_back(n/i);
        }
    }
    if (sort) {
        if (reverse) std::sort(div_list.begin(),div_list.end(),std::greater<T>());
        else std::sort(div_list.begin(),div_list.end());
    }
    return div_list;
}


// bisect functions with key
template<typename T>
auto bisect_left_withkey(T a, std::vector<T> x, int lo=0, int hi=-1,
                         std::function<T(T)> key=[](T x){ return x; },
                         bool keyvalue_x=false)
{
    auto left = lo;
    auto right = hi < 0 ? a.size() : hi;
    auto kx = keyvalue_x ? x : key(x);
    while (left + 1 < right) {
        auto mid = (left + right)/2;
        auto kn = key(a[mid]);
        if (kx <= kn)
            right = mid;
        else
            left = mid;
    }
    if (key(a[left]) == kx)
        return left;
    else
        return right;
}

template<typename T>
auto bisect_right_withkey(T a, std::vector<T> x, int lo=0, int hi=-1,
                          std::function<T(T)> key=[](T x){ return x; },
                          bool keyvalue_x=false)
{
    auto left = lo;
    auto right = hi < 0 ? a.size() : hi;
    auto kx = keyvalue_x ? x : key(x);
    while (left + 1 < right) {
        auto mid = (left + right)/2;
        auto kn = key(a[mid]);
        if (kx < kn)
            right = mid;
        else
            left = mid;
    }
    return right;
}


class combinations
{
    std::vector<int> c;
    std::vector<std::pair<int,int>> pool;
    int n, k;
public:
    combinations(int n, int k) : c(0), pool(0), n(n), k(k) {}
    bool clear() {
        if (n >= 0 && 0 <= k && k <= n) {
            c.resize(k);
            pool.clear();
            pool.push_back(std::make_pair(-1,-1));
            return next();
        } else {
            return false;
        }
    }
    const std::vector<int>& value() const { return c; }
    bool next() {
        while (pool.size() > 0) {
            auto i = pool.back().first;
            auto j = pool.back().second;
            pool.pop_back();
            if (i >= 0) {
                c[i] = j;
            }
            if (i >= k-1) {
                return true;
            } else {
                for (auto m = i+n-k+1; m > j; m--) {
                    pool.push_back(std::make_pair(i+1,m));
                }
            }
        }
        return false;
    }

    class iterator
    {
        combinations* body;
    public:
        iterator(combinations* body) : body(body) {}
        const std::vector<int>& operator*() const { return body->value(); }
        iterator& operator++() { if (!body->next()) { body = nullptr; } return *this; }
        bool operator!=(const iterator& v) const { return body != v.body; }
    };

    iterator begin() { if (clear()) return iterator(this); else return iterator(nullptr); }
    iterator end() const { return iterator(nullptr); }
};

// cf.) https://marycore.jp/prog/cpp/custom-iterator-and-range-based-for/
//template<typename Int>
class range {
    typedef int Int;
    Int start, stop, step;

    void initialize() {
        auto sign = (step >= 0) ? 1 : -1;
        stop = start + ((stop-start-sign)/step + 1) * step;
        if ((stop-start) * step <= 0) stop = start;
    }
public:
    range(Int start, Int stop, Int step=1) : start(start), stop(stop), step(step) {
        initialize();
    }
    range(Int stop) : start(0), stop(stop), step(1) { initialize(); }

    struct iterator {
        Int i, step;
        Int& operator*() { return i; }
        iterator& operator++() { i += step; return *this; }
        bool operator!=(const iterator& v) { return i != v.i; }
    };
    iterator begin() { return {start, step}; }
    iterator end()   { return {stop, step}; }
};
/* 
// Example:
for (auto i : range(10)) {
    ...
}
*/

using namespace std;
typedef llint dtype;

int main(void)
{
    cout << "defaultmap & countup" << endl;
    auto dm = defaultmap<long long int,int>(CountUp());
    std::cout << dm[5] << std::endl;
    std::cout << dm[3] << std::endl;
    std::cout << dm[10] << std::endl;
    std::cout << dm[3] << std::endl;

    cout << "inv_mod" << endl;
    std::cout << inv_mod<int>(2,2019) << std::endl;
    std::cout << inv_mod_norec<int>(2,2019) << std::endl;

    cout << "unionfind" << endl;
    UnionFind uf(4);
    uf.merge(0,1);
    uf.merge(2,3);
    uf.merge(0,3);
    std::cout << uf.groups_num() << std::endl;

    cout << "heapq" << endl;
    heapq<int> hq;
    hq.heappush(20);
    hq.heappush(10);
    hq.heappush(5);
    hq.heappush(15);
    cout << hq[0] << endl;
    cout << hq.heappop() << endl;

    cout << "segtree" << endl;
    SegmentTree<double> sg(std::vector<double>({10,0,-3,30}));
    cout << sg.query_index() << endl;

    cout << "getsequence" << endl;
    auto p = get_sequence(vector<int>({1,1,2,2,2,3,3,3,3,3}));
    cout << p.first[2] << endl;
    cout << p.second[2] << endl;

    cout << "divisors" << endl;
    for (auto j: divisors(36)) {
        cout << j << endl;
    }

    cout << "combinations" << endl;
    int n = 5;
    int k = 2;
    for (auto& c : combinations(n,k)) {
        loop(i, k) {
            //cout << "#";
            cout << c[i];
        }
        cout << endl;
    }

    cout << "range" << endl;
    for (auto i : range(5)) {
        cout << i << endl;
    }

    return 0;
}