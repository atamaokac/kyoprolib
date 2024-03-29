#include <bits/stdc++.h>
using namespace std;
typedef long long int llint;
typedef unsigned long long int ullint;
std::string YesNo[] = {"No","Yes"};
#define loop(i,N) for(int i=0; i<(N); i++)
template<typename T>
auto sort_all(T& v) { return std::sort(v.begin(),v.end()); }
template<typename T>
auto sum_all(const T& v) { auto z = v[0]; z = 0;
    return std::accumulate(v.begin(),v.end(),z); }
template<typename T>
auto input( std::size_t N,
            const std::vector<T>& head={},
            const std::vector<T>& tail={}) {
    auto s = head.size();
    auto t = tail.size();
    std::vector<T> v;
    v.reserve(s+N+t);
    v.insert(v.end(), head.begin(), head.end());
    v.resize(s+N);
    loop(i,N) std::cin >> v[s+i];
    v.insert(v.end(), tail.begin(), tail.end());
    return std::move(v);
}
template<typename T>
class range {
    T start, stop, step;
    void initialize() {
        auto sign = (step >= 0) ? 1 : -1;
        stop = start + ((stop-start-sign)/step + 1) * step;
        if ((stop-start) * step <= 0) stop = start;
    }
public:
    range(T start, T stop, T step=1) : start(start), stop(stop), step(step) {
        initialize();
    }
    range(T stop) : start(0), stop(stop), step(1) { initialize(); }
    struct iterator {
        T i, step;
        T& operator*() { return i; }
        iterator& operator++() { i += step; return *this; }
        bool operator!=(const iterator& v) { return i != v.i; }
    };
    iterator begin() { return {start, step}; }
    iterator end()   { return {stop, step}; }
};
template<typename T1, typename T2>
std::pair<T1,T2> operator-(const std::pair<T1,T2>& x)
{
    return std::make_pair(-x.first, -x.second);
}
template<typename T1, typename T2>
std::pair<T1,T2> operator+(
    const std::pair<T1,T2>& x,
    const std::pair<T1,T2>& y)
{
    return std::make_pair(x.first+y.first, x.second+y.second);
}
template<typename T1, typename T2>
std::pair<T1,T2> operator-(const std::pair<T1,T2>& x, const std::pair<T1,T2>& y)
{
    return x + (-y);
}
template<typename T1, typename T2>
std::pair<T1,T2>& operator+=(std::pair<T1,T2>& x, const std::pair<T1,T2>& y)
{
    x.first += y.first;
    x.second += y.second;
    return x;
}
template<typename T1, typename T2>
std::pair<T1,T2>& operator-=(std::pair<T1,T2>& x, const std::pair<T1,T2>& y)
{
    return x += (-y);
}
template<typename T1, typename T2>
std::ostream& operator<<(std::ostream& cout, const std::pair<T1,T2>& data)
{
    cout << "(" << data.first << ", " << data.second << ")";
    return cout;
}
template<typename T>
std::ostream& operator<<(std::ostream& cout, const std::vector<T>& data)
{
    cout << "[";
    if (data.size() == 0) {}
    else if (data.size() == 1) { cout << data[0]; } 
    else {
        cout << data[0];
        for (auto itr=data.begin()+1; itr!=data.end(); ++itr) {
            cout << ", " << *itr;
        }
    }
    cout << "]";
    return cout;
}

typedef llint Int;

int main(void)
{

    return 0;
}