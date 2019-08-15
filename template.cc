#include <bits/stdc++.h>
#include <boost/function.hpp>
using namespace std;
typedef long long int llint;
typedef unsigned long long int ullint;
std::string YesNo[] = {"No","Yes"};
#define loop(i,N) for(int i=0; i<(N); i++)
template<typename T>
auto sort_all(T v) { return std::sort(v.begin(),v.end()); }
template<typename T>
auto sum_all(T v) { auto z = v[0]; z = 0;
    return std::accumulate(v.begin(),v.end(),z); }
template<typename T>
auto input(int N) {
    std::vector<T> v(N);
    loop(i,N) std::cin >> v[i];
    return std::move(v);
}

typedef llint dtype;

int main(void)
{

    return 0;
}