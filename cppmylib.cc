#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <map>
#include <boost/function.hpp>
#include <cmath>
using namespace std;
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
    boost::function<V(K)> defaultfunc;
public:
    defaultmap(boost::function<V(K)> df) { defaultfunc = df; };
    auto& operator[](K key) {
        auto itr = this->find(key);
        if( itr != this->end() )
            return itr->second;
        else
            return this->std::map<K,V>::operator[](key)=defaultfunc(key);
    };
};
class CountUp
{
private:
    int index;
public:
    CountUp() { index = 0; };
    auto operator()(...) { return index++; };
    auto next() { return index; };
};
/* Usage of CountUp:
    CountUp cu;
    auto dm = defaultmap<K, int>(cu);
*/


template<typename T>
auto inv_mod_sub(T a, T p) {
    if(a == 1) {
        return std::pair<T,T>(1, 0);
    } else {
        auto d = p / a;
        auto r = p % a;
        auto xy = inv_mod_sub(r, a);
        auto x = xy.first;
        auto y = xy.second;
        return std::pair<T,T>(y-d*x, x);
    }
}
template<typename T>
auto inv_mod(T a, T p) {
    if(p < 0) p = -p;
    a %= p;
    auto ans = inv_mod_sub<T>(a, p).first % p;
    if(ans < 0) ans += p;
    return ans;
}

template<typename T>
auto pow_mod(T x, T n, T p) {
    if(p == 1 || p == -1) {
        return 0;
    }
    T ans = 1;
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
    P el;
    int gN;
public:
    UnionFind(int N) {
        parent = std::move(P(N,-1));
        el = std::move(P(N));
        loop(i,N) el[i]=i;
        gN = N;
    };

    auto root(int n) {
        if(parent[n] < 0) {
            return n;
        } else {
            auto m = root(parent[n]);
            parent[n] = m;
            return m;
        }
    };

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
    };

    auto size(int n) { return -parent[root(n)]; };
    auto connected(int m, int n) { return root(m) == root(n); };
    auto groups_num() { return gN; };
    auto elements_num() { return parent.size(); };
    const auto& elements() { return el; };
};

template<typename T>
class Factorizer
{
private:
    T primes_ceil;
    std::vector<T> primes;
public:
    Factorizer() {
        primes_ceil = 5;
        primes = std::move(std::vector<T>({2,3}));
    };

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
    };

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
    };
};


typedef llint dtype;

int main(void)
{
    CountUp cu;
    auto dm = defaultmap<long long int,int>(cu);
    std::cout << dm[5] << std::endl;
    std::cout << dm[3] << std::endl;
    std::cout << dm[10] << std::endl;
    std::cout << dm[3] << std::endl;

    std::cout << inv_mod<int>(2,2019) << std::endl;

    UnionFind uf(4);
    uf.merge(0,1);
    uf.merge(2,3);
    uf.merge(0,3);
    std::cout << uf.groups_num() << std::endl;

    return 0;
}