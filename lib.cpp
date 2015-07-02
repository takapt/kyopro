#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
#include <cfloat>
#include <ctime>
#include <cassert>
#include <map>
#include <utility>
#include <set>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <sstream>
#include <complex>
#include <stack>
#include <queue>
#include <numeric>
#include <list>
#include <iomanip>
#include <fstream>
#include <bitset>
   
using namespace std;
   
 
#define rep(i, n) for (int i = 0; i < (int)(n); ++i)
#define foreach(it, c) for (__typeof__((c).begin()) it=(c).begin(); it != (c).end(); ++it)
#define rforeach(it, c) for (__typeof__((c).rbegin()) it=(c).rbegin(); it != (c).rend(); ++it)
#define all(c) (c).begin(), (c).end()
#define rall(c) (c).rbegin(), (c).rend()
#define CL(arr, val) memset(arr, val, sizeof(arr))
#define COPY(dest, src) memcpy(dest, src, sizeof(dest))
#define ten(n) ((long long)(1e##n))
#define bin(n) (1LL << (n))
#define erep(i, n) for (int i = 0; i <= (int)(n); ++i)
#define revrep(i, n) for (int i = (n); i >= 0; --i)
#define pb push_back

template <class T> void chmax(T& a, const T& b) { a = max(a, b); }
template <class T> void chmin(T& a, const T& b) { a = min(a, b); }

template <class T> void uniq(T& c) { sort(c.begin(), c.end()); c.erase(unique(c.begin(), c.end()), c.end()); }
 
template <class T> string to_s(const T& a) { ostringstream os; os << a; return os.str(); }
template <class T> T to_T(const string& s) { istringstream is(s); T res; is >> res; return res; }

template <typename T> void print_container(ostream& os, const T& c) { const char* _s = " "; if (!c.empty()) { __typeof__(c.begin()) last = --c.end(); foreach (it, c) { os << *it; if (it != last) cout << _s; } } }
template <typename T> ostream& operator<<(ostream& os, const vector<T>& c) { print_container(os, c); return os; }
template <typename T> ostream& operator<<(ostream& os, const set<T>& c) { print_container(os, c); return os; }
template <typename T> ostream& operator<<(ostream& os, const multiset<T>& c) { print_container(os, c); return os; }
template <typename T> ostream& operator<<(ostream& os, const deque<T>& c) { print_container(os, c); return os; }
template <typename T, typename U> ostream& operator<<(ostream& os, const map<T, U>& c) { print_container(os, c); return os; }
template <typename T, typename U> ostream& operator<<(ostream& os, const pair<T, U>& p) { os << "( " << p.first << ", " << p.second << " )"; return os; }

template <class T> void print(T a, int n, const string& deli = " ", int br = 1) { for (int i = 0; i < n; ++i) { cout << a[i]; if (i + 1 != n) cout << deli; } while (br--) cout << endl; }
template <class T> void print2d(T a, int w, int h, int width = -1, int br = 1) { for (int i = 0; i < h; ++i) { for (int j = 0; j < w; ++j) {    if (width != -1) cout.width(width); cout << a[i][j] << ' '; } cout << endl; } while (br--) cout << endl; }

template <class T> void input(T& a, int n) { for (int i = 0; i < n; ++i) cin >> a[i]; }
template <class T> void input(T* a, int n) { for (int i = 0; i < n; ++i) cin >> a[i]; }

void fix_pre(int n) { cout.setf(ios::fixed, ios::floatfield); cout.precision(10); }
void fast_io() { cin.tie(0); ios::sync_with_stdio(false); }
#define trace(x) (cout << #x << ": " << (x) << endl)
 
bool in_rect(int x, int y, int w, int h) { return 0 <= x && x < w && 0 <= y && y < h; }

typedef long long ll;
typedef pair<int, int> pint;

// y(v): v>^<  y(^): ^>v<
const int dx[] = { 0, 1, 0, -1 };
const int dy[] = { 1, 0, -1, 0 };

const double PI = acos(-1.0);
#define mp make_pair



// hash
#if __GNUC__
#include <tr1/unordered_map>
#include <tr1/unordered_set>
using namespace tr1;
#else
#include <unordered_map>
#include <unordered_set>
#endif
  
// popcount
#ifdef __GNUC__
template <class T> int popcount(T n);
template <> int popcount(unsigned int n) { return __builtin_popcount(n); }
template <> int popcount(int n) { return __builtin_popcount(n); }
template <> int popcount(unsigned long long n) { return __builtin_popcountll(n); }
template <> int popcount(long long n) { return __builtin_popcountll(n); }
#else
#define __typeof__ decltype
template <class T> int popcount(T n) { return n ? 1 + popcount(n & (n - 1)) : 0; }
#endif



//////////////////////////////////////////
// ローマ数字
int roman_to_n(const string& s)
{
    static const int num[] = { 1, 5, 10, 50, 100, 500, 1000 };
    static const char* roman = "IVXLCDM";

    vector<int> digits(s.size());
    for (int i = 0; i < s.size(); ++i)
        digits[i] = num[strchr(roman, s[i]) - roman];

    int res = accumulate(digits.begin(), digits.end(), 0);
    for (int i = 0; i < digits.size() - 1; ++i)
        if (digits[i] < digits[i + 1])
            res -= 2 * digits[i];
    return res;
}
string to_roman(int n)
{
    static const char* roman[][10] = {
        { "", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX" }, 
        { "", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC" },
        { "", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM" },
        { "", "M", "MM", "MMM", "MMMM", "MMMMM" },
    };

    stringstream res;
    for (int i = 3, j = 1000; i >= 0; --i, j /= 10)
        res << roman[i][(n / j) % 10];
    return res.str();
}


///////////////////////////////////////////////
// UnionFind
class UnionFind
{
private:
    vector<int> data;
    int _groups;
public:
    int n;
    UnionFind(int n) : data(n, -1), _groups(n), n(n) { }

    void unite(int x, int y)
    {
        x = root(x), y = root(y);
        if (x != y)
        {
            --_groups;
            if (data[x] > data[y])
                swap(x, y);
            data[x] += data[y];
            data[y] = x;
        }
    }
    bool same(int x, int y) { return root(x) == root(y); }
    int root(int x) { return data[x] < 0 ? x : data[x] = root(data[x]); }
    int size(int x) { return -data[root(x)]; }
    int groups() const { return _groups; }
};

class WeightUnionFind
{
private:
    vector<pint> e; // (parent, diff)

public:
    
    WeightUnionFind(int n)
        : e(n, pint(-1, 0)) {}

    // w(a) + w == b
    void unite(int a, int b, int w)
    {
        if (!same(a, b))
        {
            pint ra = find(a);
            pint rb = find(b);
            e[ra.first] = pint(rb.first, rb.second - ra.second - w);
        }
    }

    // (root, diff)
    pint find(int a)
    {
        if (e[a].first == -1)
            return pint(a, 0);
        else
        {
            pint r = find(e[a].first);
            return e[a] = pint(r.first, r.second + e[a].second);
        }
    }

    bool same(int a, int b)
    {
        return find(a).first == find(b).first;
    }
};



////////////////////////////////////////
// 2進数に変換
// digits桁の2進数に変換
template <class T>
string to_bin(T n, int digits)
{
    string res(digits, '0');
    for (int i = digits - 1; i >= 0; --i, n >>= 1)
        if (n & 1)
            res[i] = '1';
    return res;
}
template <class T>
string to_bin(T n)
{
    int digits = 0;
    for (T i = n; i > 0; i >>= 1)
        ++digits;
    return to_bin(n, max(1, digits));
}
// 2進数を整数に変換
ll bin_to_n(const string& b)
{
    ll res = 0;
    for (int i = 0; i < (int)b.size(); ++i)
        res = (res << 1) + (b[i] == '1' ? 1 : 0);
    return res;
}

/////////////////////////////////////////////////
// 基数変換
// base進数に変換
template <class T>
string convert_base(T n, int base)
{
    if (n == 0) return "0";

    static const char* digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    stringstream ss;
    for (T i = n; i > 0; i /= base)
        ss << digits[i % base];

    string t = ss.str();
    reverse(t.begin(), t.end());
    return t;
}
// base進数の文字列を整数
ll inverse_base(const string& s, int base)
{
    static const char* digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    ll res = 0;
    for (int i = 0; i < (int)s.size(); ++i)
    {
        char c = isdigit(s[i]) ? s[i] : toupper(s[i]);
        int d = strchr(digits, c) - digits;
        res = (res * base) + d;
    }
    return res;
}


vector<int> to_negabase(ll n, int base)
{
    vector<int> res;
    if (n == 0)
    {
        res.push_back(0);
        return res;
    }

    while (n != 0)
    {
        int rem = n % base;
        n /= base;
        if (rem < 0)
        {
            rem += -base;
            ++n;
        }
        res.push_back(rem);
    }
    return res;
}
string to_negabase_s(ll n, int base)
{
    vector<int> digits = to_negabase(n, base);
    string res;
    for (int i = digits.size() - 1; i >= 0; --i)
        res += to_s(digits[i]);
    return res;
}
int main()
{
    int n;
    cin >> n;
    cout << to_negabase_s(n, -2) << endl;
}


//////////////////////////////////////////////////////////
// ビット集合
// 32bitsでn = 32だと死ぬ
template <class T>
T next_combination(T combi)
{
    T leastBit = combi & -combi;
    T mostBits = combi + leastBit;
    T leastSequentialBits = combi & ~mostBits;
    return mostBits | ((leastSequentialBits / leastBit) >> 1);
}
template <class T>
vector<T> combination_bits(int n, int k)
{
    T one = 1;
    vector<T> res;
    for (T i = (one << k) - 1; i < (one << n); i = next_combination(i))
    {
        cout << to_bin(i, n) << endl;
        res.push_back(i);
    }
    return res;
}


	
// Suffix Array
struct SAComp
{
    int k;
    vector<int>* rank;
    bool operator()(int i, int j) const
    {
        if (rank->at(i) != rank->at(j))
            return rank->at(i) < rank->at(j);
        else
        {
            int ri = i + k < rank->size() ? rank->at(i + k) : -1;
            int rj = j + k < rank->size() ? rank->at(j + k) : -1;
            return ri < rj;
        }
    }
};
// O(n * (log n)^2)
template <typename T>
vector<int> create_sa(const T& s)
{
    const int n = s.size();
    vector<int> sa(n + 1);

    vector<int> rank(n + 1);
    vector<int> temp(n + 1);

    for (int i = 0; i <= n; ++i)
    {
        sa[i] = i;
        rank[i] = i < n ? s[i] : -1;
    }

    SAComp comp;
    comp.rank = &rank;
    for (int k = 1; k <= n; k *= 2)
    {
        comp.k = k;
        sort(sa.begin(), sa.end(), comp);

        temp[sa[0]] = 0;
        for (int i = 1; i <= n; ++i)
            temp[sa[i]] = temp[sa[i - 1]] + (comp(sa[i - 1], sa[i]) ? 1 : 0);
        for (int i = 0; i <= n; ++i)
            rank[i] = temp[i];
    }

    return sa;
}
template <typename T>
vector<int> create_lcp(const T& s, const vector<int>& sa)
{
    const int n = s.size();

    vector<int> rank(n + 1);
    for (int i = 0; i <= n; ++i)
        rank[sa[i]] = i;

    vector<int> lcp(n);
    int h = 0;
    lcp[0] = 0;
    for (int i = 0; i < n; ++i)
    {
        int j = sa[rank[i] - 1];

        if (h > 0)
            --h;
        for (; j + h < n && i + h < n; ++h)
        {
            if (s[j + h] != s[i + h])
                break;
        }
        lcp[rank[i] - 1] = h;
    }
    return lcp;
}


	
/////////////////////////////////////////////////////////////////////////////////
//// 数論
///////////////////////////////////////////////////

///////////////////////////////////////////////////////////
// 素数
bool is_prime(ll n)
{
    if (n <= 1)
        return false;
    for (ll i = 2; i * i <= n; ++i)
        if (n % i == 0)
            return false;
    return true;
}

vector<bool> prime_table(int upto)
{
    vector<bool> p(upto + 1, true);
    p[0] = p[1] = false;
    for (int i = 2; i * i <= upto; ++i)
        if (p[i])
            for (int j = i + i; j <= upto; j += i)
                p[j] = false;
    return p;
}
vector<int> primes(int upto, const vector<bool>& p = prime_table(0))
{
    if ((int)p.size() <= upto)
        return primes(upto, prime_table(upto));

    vector<int> primes;
    for (int i = 0; i <= upto; ++i)
        if (p[i])
            primes.push_back(i);
    return primes;
}

class PrimeFactor
{
public:
    int n;
    vector<int> factor;
    PrimeFactor(int n)
        : n(n), factor(n + 1, -1)
    {
        for (int i = 2; i <= n; ++i)
        {
            if (factor[i] == -1)
            {
                for (int j = i; j <= n; j += i)
                    factor[j] = i;
            }
        }
    }

    map<int, int> prime_factor_with_num(int x)
    {
        assert(x <= n);

        map<int, int> pf;
        for (int i = x; i > 1; )
        {
            int f = factor[i];
            ++pf[f];
            i /= f;
        }
        return pf;
    }

    vector<int> prime_factor(int x)
    {
        vector<int> p;
        for (int i = x; i > 1; )
        {
            int f = factor[i];
            p.push_back(f);
            i /= f;
        }
        sort(p.begin(), p.end());
        p.erase(unique(p.begin(), p.end()), p.end());
        return p;
    }
};

// prime_factorsを使うときに区間が10^6以上あるとTLE, MLEに注意
typedef pair<ll, int> pf_type;
class SegmentPrimeTable
{
private:
    vector<bool> table;
    vector<vector<pf_type> > _prime_factors; // (prime_factor, num of pf)

    int index(ll n) const
    {
        assert(lower <= n && n <= upper);
        return n - lower;
    }
public:
    ll lower, upper;

    SegmentPrimeTable() {}
    SegmentPrimeTable(ll lower, ll upper)
    {
        sieve(lower, upper);
    }

    void sieve(ll lower, ll upper)
    {
        assert(lower >= 0);
        this->lower = lower, this->upper = upper;

        ll len = upper - lower + 1;
        table.resize(len);
        fill(table.begin(), table.end(), true);

        vector<bool> p((int)sqrt(upper) + 128, true);
        if (lower == 0) table[index(0)] = false;
        if (lower <= 1 && 1 <= upper) table[index(1)] = false;
        for (ll i = 2; i * i <= upper; ++i)
        {
            if (p[i])
            {
                for (ll j = i + i; j * j <= upper; j += i)
                    p[j] = false;
                for (ll j = max(i + i, (lower + i - 1) / i * i); j <= upper; j += i)
                    table[index(j)] = false;
            }
        }
    }

    void sieve_prime_factors(ll lower, ll upper)
    {
        assert(lower >= 0);
        this->lower = lower, this->upper = upper;

        ll len = upper - lower + 1;

        _prime_factors.resize(len);
        for (ll i = lower; i <= upper; ++i)
            _prime_factors[index(i)].clear();

        vector<bool> p((int)sqrt(upper) + 128, true);
        for (ll i = 2; i * i <= upper; ++i)
        {
            if (p[i])
            {
                for (ll j = i + i; j * j <= upper; j += i)
                    p[j] = false;

                if (lower <= i && i <= upper)
                    _prime_factors[index(i)].push_back(make_pair(i, 0));

                for (ll j = max(i + i, (lower + i - 1) / i * i); j <= upper; j += i)
                    _prime_factors[index(j)].push_back(make_pair(i, 0));
            }
        }

        for (ll i = lower; i <= upper; ++i)
        {
            vector<pf_type>& pf = _prime_factors[index(i)];
            ll rem = i;
            for (int j = 0; j < (int)pf.size(); ++j)
            {
                pf[j].second = 0;
                while (rem % pf[j].first == 0)
                {
                    rem /= pf[j].first;
                    ++pf[j].second;
                }
            }
            if (rem > 1)
                pf.push_back(make_pair(rem, 1));
        }
    }

    bool is_prime(ll n) const { return table[index(n)]; }
    vector<ll> primes(ll low, ll up) const
    {
        vector<ll> p;
        for (ll i = low; i <= up; ++i)
            if (is_prime(i))
                p.push_back(i);
        return p;
    }

    const vector<pf_type>& prime_factors(ll n) const
    {
        return _prime_factors[index(n)];
    }
};


/////////////////////////////////////////////////////////
// 最大公約数，最小公倍数

// ユークリッドの互除法
template <class T> T gcd(T a, T b) { return b ? gcd(b, a % b) : a; }
template <class T> T lcm(T a, T b) { return a / gcd(a, b) * b; }

// 拡張ユークリッドの互除法
// ax + by = gcd(a, b), (x, y)を求める
// unsignedはまずいですよ！
template <class T>
T ext_gcd(T a, T b, T &x, T &y)
{
    if (b)
    {
        T g = ext_gcd(b, a % b, y, x);
        y -= (a / b) * x;
        return g;
    }
    else
    {
        x = 1, y = 0;
        return a;
    }
}


// 素因数分解
template <class T>
map<T, int> prime_factors_num(T n)
{
    map<T, int> res;
    for (T i = 2; i * i <= n; ++i)
    {
        if (n % i == 0)
        {
            int c = 0;
            for ( ; n % i == 0; n /= i)
                ++c;
            res[i] = c;
        }
    }
    if (n > 1)
        res[n] = 1;
    return res;
}
// 昇順で並んでるよ
template <class T>
vector<T> prime_factors(T n)
{
    vector<T> res;
    for (T i = 2; i * i <= n; ++i)
    {
        while (n % i == 0)
        {
            res.push_back(i);
            n /= i;
        }
    }
    if (n > 1)
        res.push_back(n);
    return res;
}


// 約数列挙
template <class T>
vector<T> divisors(T n)
{
    vector<T> res;
    for (T i = 1; i * i <= n; ++i)
    {
        if (n % i == 0)
        {
            res.push_back(i);
            if (i != n / i)
                res.push_back(n / i);
        }
    }
    sort(res.begin(), res.end());
    return res;
}

// the number of p factors of n! (factorial(n))
ll fact_factors(ll n, ll p)
{
    ll num = n / p;
    return num == 0 ? 0 : num + fact_factors(num, p);
}


// 1234 -> { 4, 3, 2, 1 }
vector<int> digits(ll n)
{
    vector<int> d;
    while (n > 0)
        d.push_back(n % 10), n /= 10;
    return d;
}


// オイラーのφ関数
// φ(n): 1~nでnと互いに素な数
// 性質: x^φ(n) = 1 (mod n)
int euler_phi(int n)
{
    int res = n;
    for (int i = 2; i * i <= n; ++i)
    {
        if (n % i == 0)
        {
            res = res / i * (i - 1);
            while (n % i == 0)
                n /= i;
        }
    }
    if (n > 1)
        res = res / n * (n - 1);
    return res;
}

const int MAX_EULER = 1000;
int euler[MAX_EULER + 1];
// O(MAX_EULER)
void make_euler_phi()
{
    for (int i = 0; i <= MAX_EULER; ++i)
        euler[i] = i;
    for (int i = 2; i <= MAX_EULER; ++i)
        if (euler[i] == i)
            for (int j = i; j <= MAX_EULER; j += i)
                euler[j] = euler[j] / i * (i - 1);
}


// 順列, 組み合わせ(コンビネーション)
const int MAX_P_FACT = 100000;
int fact[MAX_P_FACT];
int p_for_fact;
void calc_fact(int p)
{
    p_for_fact = p;
    fact[0] = 1;
    for (int i = 1; i < p; ++i)
        fact[i] = (long long)i * fact[i - 1] % p;
}
// 使う前にcalc_fact
// n! = a * p^eのaを求める
// O(log_p n)
int _mod_fact(int n, int p, int& e)
{
    e = 0;
    if (n == 0)
        return 1;

    long long res = _mod_fact(n / p, p, e);
    e += n / p;

    if (n / p % 2 != 0)
        return res * (p - fact[n % p]) % p;
    else
        return res * fact[n % p] % p;
}
int mod_fact(int n, int p, int& e)
{
    if (p_for_fact != p)
        calc_fact(p);
    return _mod_fact(n, p, e);
}
// 使う前にcalc_fact
// nCk mod p
// O(log_p n)
int mod_combi(int n, int k, int p)
{
    if (n < 0 || k < 0 || n < k)
        return 0;

    int a1, a2, a3, e1, e2, e3;
    a1 = mod_fact(n, p, e1);
    a2 = mod_fact(k, p, e2);
    a3 = mod_fact(n - k, p, e3);

    if (e1 > e2 + e3)
        return 0;
    else
        return a1 * mod_inverse((int)((long long)a2 * a3 % p), p) % p;
}


////////////////////////////////////////////////////
// combination, pascal triangle
template <class T>
vector<vector<T> > combi_table(int n, int k, T mod)
{
    vector<vector<T> > c(n + 1, vector<T>(k + 1));
    for (int i = 0; i <= n; ++i)
    {
        c[i][0] = 1;
        for (int j = 1; j <= min(i, k); ++j)
        {
            c[i][j] = c[i - 1][j - 1] + c[i - 1][j];
            if (c[i][j] >= mod)
                c[i][j] -= mod;
        }
    }
    return c;
}
template <class T>
vector<vector<T> > combi_table(int n, int k)
{
    vector<vector<T> > c(n + 1, vector<T>(k + 1));
    for (int i = 0; i <= n; ++i)
    {
        c[i][0] = 1;
        for (int j = 1; j <= min(i, k); ++j)
            c[i][j] = c[i - 1][j - 1] + c[i - 1][j];
    }
    return c;
}


ll combi(int n, int k)
{
    ll res = 1;
    k = min(k, n - k);
    for (int i = 1; i <= k; ++i)
        res = res * (n - k + i) / i;
    return res;
}


// 累乗
ll mod_pow(ll x, ll n, ll mod)
{
    ll res = 1;
    while (n > 0)
    {
        if (n & 1)
            (res *= x) %= mod;
        x = x * x % mod;
        n >>= 1;
    }
    return res;
}

ll mod_mul(ll x, ll n, ll mod)
{
    if (n == 0)
        return 0;
    if (x < 0)
        x = (x + mod) % mod;
    if (n < 0)
        n = (n + mod) % mod;

    ll res = 0;
    while (n > 0)
    {
        if (n & 1)
            res = (res + x) % mod;
        x = x * 2 % mod;
        n /= 2;
    }

    return res;
}
ll ext_gcd(ll a, ll b, ll &x, ll &y)
{
    if (b)
    {
        ll g = ext_gcd(b, a % b, y, x);
        y -= (a / b) * x;
        return g;
    }
    else
    {
        x = 1, y = 0;
        return a;
    }
}
// 逆元
// a * x == 1 (mod m)を満たすxを求める
// 解なしの場合0を返す
// a*x + m*y == 1をやる
ll mod_inverse(ll a, ll mod)
{
    ll x, y;
    if (ext_gcd((a + mod) % mod, mod, x, y) == 1)
        return (mod + x % mod) % mod;
    else
        return 0;
}

ll mod_fact(ll n, ll mod)
{
    ll res = 1;
    for (ll i = n; i > 0; --i)
        (res *= i) %= mod;
    return res;
}
ll mod_P(ll n, ll k, ll mod)
{
    assert(0 <= k && k <= n);
    k = min(k, n - k);
    ll res = 1;
    for (ll i = n; i > n - k; --i)
        (res *= i % mod) %= mod;
    return res;
}
ll mod_C(ll n, ll k, ll mod)
{
    assert(0 <= k && k <= n);
    k = min(k, n - k);
    return (mod_P(n, k, mod) * mod_inverse(mod_fact(k, mod), mod)) % mod;
}


// about: inv(i)
// i * (m / i) == m - (m % i)を変形すると出てくる
// dont use inv[0] !!!
class ModUtil
{
private:
    ll max, mod;
    vector<ll> _inv, _fact, _inv_fact;

    void init()
    {
        _inv[1] = 1;
        for (int i = 2; i <= max; ++i)
            _inv[i] = _inv[mod % i] * (mod - mod / i) % mod;

        _fact[0] = 1;
        for (int i = 1; i <= max; ++i)
            _fact[i] = i * _fact[i - 1] % mod;

        _inv_fact[0] = 1;
        for (int i = 1; i <= max; ++i)
            _inv_fact[i] = _inv[i] * _inv_fact[i - 1] % mod;
    }

public:
    ModUtil(ll max, ll mod)
        : max(max), mod(mod), _inv(max + 1), _fact(max + 1), _inv_fact(max + 1)
    {
        init();
    }


    ll inv(ll n) const { return _inv[n]; }
    ll fact(ll n) const { return _fact[n]; }
    ll inv_fact(ll n) const { return _inv_fact[n]; }
    ll P(ll n, ll k) const { return _fact[n] * _inv_fact[n - k] % mod; }
    ll C(ll n, ll k) const { return P(n, k) * _inv_fact[k] % mod; } 
};


// i * (m / i) == m - (m % i)を変形すると出てくる
// dont use inv[0] !!!
vector<ll> list_mod_inverse(ll until, ll mod)
{
    vector<ll> inv(until + 1);
    inv[1] = 1;
    for (int i = 2; i <= until; ++i)
        inv[i] = inv[mod % i] * (mod - mod / i) % mod;
    return inv;
}


// a[i] * x == b[i] (mod m[i])
// return (b, m) | (0, -1); // unsolvable
pair<ll, ll> linear_congruence(const vector<ll>& A, const vector<ll>& B, const vector<ll>& M)
{
    ll x = 0, m = 1;
    for (int i = 0; i < (int)A.size(); ++i)
    {
        ll a = A[i] * m,  b = B[i] - A[i] * x, d = __gcd(M[i], a);
        if (b % d != 0)
            return make_pair(0, -1); // unsolvable
        ll t = b / d * mod_inverse(a / d, M[i] / d) % (M[i] / d);
        x = x + m * t;
        m *= M[i] / d;
        x = (x + m) % m;
    }
    return make_pair(x % m, m);
}

/////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
// 文字列処理
////////////////////////////////////////////////////////


// バグでなさそうだから3,4倍遅くていいときはこっち
vector<string> split(string str, const string& deli_chars)
{
    for (int i = 0; i < (int)str.size(); ++i)
        if (deli_chars.find(str[i]) != string::npos)
            str[i] = ' ';
    stringstream ss(str);
    vector<string> res;
    for (string s; ss >> s; )
        res.push_back(s);
    return res;
}
string cat(const vector<string>& s, const string& t)
{
    string res = s[0];
    for (int i = 1; i < (int)s.size(); ++i)
        res += t + s[i];
    return res;
}
// 置換
string replace(const string& str, const string& from, const string& to)
{
    string res = str;
    string::size_type p;
    while ((p = res.find(from)) != string::npos)
        res.replace(p, from.size(), to);
    return res;
}

//
bool start_with(const string& s, const string& t)
{
    if (s.size() < t.size())
        return false;
    return s.compare(0, t.size(), t) == 0;
}
bool end_with(const string& s, const string& t)
{
    if (s.size() < t.size())
        return false;
    return s.compare(s.size() - t.size(), t.size(), t) == 0;
}


// deli_charsのいずれかの文字で区切る, deli_cars=",": "aa,aa," => { "aa", "aa", "" }
vector<string> split(const string& str, const string& deli_chars)
{
    vector<string> res;
    string::size_type i = 0, pos;
    while ((pos = str.find_first_of(deli_chars, i)) != string::npos)
    {
        res.push_back(str.substr(i, pos - i));
        i = pos + 1;
    }
    res.push_back(str.substr(i, str.size() - i));
    return res;
}


// 編集距離, レーベンシュタイン距離
template <class T>
int edit_distance(const T& a, const T& b)
{
    const int n = a.size(), m = b.size();
    vector<vector<int> > dp(n + 1, vector<int>(m + 1));

    for (int i = 0; i <= n; ++i)
        dp[i][0] = i;
    for (int i = 0; i <= m; ++i)
        dp[0][i] = i;

    const int replace_cost = 1;	// 2だと置換禁止と同じ
    for (int i = 1; i <= n; ++i)
    {
        for (int j = 1; j <= m; ++j)
        {
            int c = replace_cost;
            if (a[i - 1] == b[j - 1])
                c = 0;

            dp[i][j] = dp[i - 1][j - 1] + c;	// replace
            dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1);	// insert
            dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1);	// delete
        }
    }

    return dp[n][m];
}



// 最長共通部分文字列 (連続してなくてもおｋ)
template <class T>
T lcs(const T& a, const T& b)
{
    const int n = a.size(), m = b.size();
    vector<vector<int> > dp(n + 1, vector<int>(m + 1));
    vector<vector<int> > prev(n + 1, vector<int>(m + 1));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if (a[i] == b[j])
            {
                dp[i + 1][j + 1] = dp[i][j] + 1;
                prev[i + 1][j + 1] = 0;
            }
            else if (dp[i][j + 1] > dp[i + 1][j])
            {
                dp[i + 1][j + 1] = dp[i][j + 1];
                prev[i + 1][j + 1] = 1;
            }
            else
            {
                dp[i + 1][j + 1] = dp[i + 1][j];
                prev[i + 1][j + 1] = 2;
            }
        }
    }
    T res;
    for (int i = n, j = m; i > 0 && j > 0; )
    {
        if (prev[i][j] == 0)
        {
            res.push_back(a[i - 1]);
            --i, --j;
        }
        else if (prev[i][j] == 1)
            --i;
        else
            --j;
    }
    reverse(res.begin(), res.end());
    return res;
}


// 文字列ここまで//////////////
////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////
// グラフ


// トポロジカルソート
// O(V + E)
bool topological_sort(const vector<vector<int> >& g, vector<int>& order)
{
    const int n = g.size();
    order.clear();

    vector<int> indegree(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < (int)g[i].size(); ++j)
            ++indegree[g[i][j]];
    
    queue<int> q;
    for (int i = 0; i < n; ++i)
        if (indegree[i] == 0)
            q.push(i);

    while (!q.empty())
    {
        int v = q.front();
        q.pop();

        order.push_back(v);

        for (int i = 0; i < (int)g[v].size(); ++i)
            if (--indegree[g[v][i]] == 0)
                q.push(g[v][i]);
    }
    return (int)order.size() == n;
}


// by spaghetti source
// トポロジカルソート
bool topo_dfs(int v, const vector<vector<int> >& g, vector<int>& color, vector<int>& order)
{
    color[v] = 1;
    for (int i = 0; i < (int)g[v].size(); ++i)
    {
        int to = g[v][i];
        if (color[to] == 2)
            continue;
        else if (color[to] == 1)
            return false;
        else if (!topo_dfs(to, g, color, order))
            return false;
    }
    color[v] = 2;

    order.push_back(v);
    return true;
}
// O(V + E)
// order: res[0] -> res[1] -> res[2] -> ...
// impossible -> return empty vector
vector<int> topological_sort(const vector<vector<int> >& g)
{
    const int n = g.size();

    vector<int> color(n);
    vector<int> order;

    for (int i = 0; i < n; ++i)
        if (color[i] == 0 && !topo_dfs(i, g, color, order))
            return vector<int>();
    reverse(all(order));
    return order;
}


// ダイクストラ
struct Edge
{
    int to, cost;
    Edge(int to, int cost)
        : to(to), cost(cost) {}
};
typedef vector<vector<Edge> > Graph;
void add_edge(Graph& g, int a, int b, int cost)
{
    g[a].push_back(Edge(b, cost));
    g[b].push_back(Edge(a, cost));
}
void add_dir_edge(Graph& g, int a, int b, int cost)
{
    g[a].push_back(Edge(b, cost));
}
vector<int> dijkstra(const vector<vector<pint> >& g, const vector<int>& s)
{
    const int _INF = 2 * ten(9);
    typedef pair<int, int> pint;

    vector<int> dis(g.size(), _INF);
    priority_queue<pint, vector<pint>, greater<pint> > q;

    for (int i = 0; i < (int)s.size(); ++i)
    {
        dis[s[i]] = 0;
        q.push(pint(0, s[i]));
    }
    while (!q.empty())
    {
        pint t = q.top(); q.pop();
        int cur = t.second, d = t.first;

        if (d > dis[cur])
            continue;

        for (int i = 0; i < (int)g[cur].size(); ++i)
        {
            const pint& e = g[cur][i];
            int to = e.first;
            int next_d = d + e.second;
            if (next_d < dis[to])
            {
                dis[to] = next_d;
                q.push(pint(next_d, to));
            }
        }
    }

    return dis;
}

/////////////////////////////////////////////////
// ネットワークフロー
// 2部グラフ
// O(V * (V + E))
class BipatiteGraph
{
    vector<bool> used;

    bool _dfs(int v)
    {
        used[v] = true;
        for (int i = 0; i < (int)edges[v].size(); ++i)
        {
            int u = edges[v][i];
            int k = match[u];
            if (k < 0 || (!used[k] && _dfs(k)))
            {
                match[v] = u;
                match[u] = v;
                return true;
            }
        }
        return false;
    }

public:
    int n, m;
    int vertices;
    vector<vector<int> > edges;
    vector<int> match;
    int matching;

    BipatiteGraph(int n, int m)
        : used(n + m), n(n), m(m), vertices(n + m), edges(n + m), match(n + m), matching(0)
    {
        clear();
    }

    void add_edge(int a, int b)
    {
        edges[a].push_back(b + n);
        edges[b + n].push_back(a);
    }

    void clear()
    {
        fill(match.begin(), match.end(), -1);
        matching = 0;
    }

    bool dfs(int v)
    {
        fill(used.begin(), used.end(), false);
        return _dfs(v);
    }

    int update_matching()
    {
        for (int i = 0; i < n; ++i)
            if (match[i] < 0 && dfs(i))
                ++matching;
        return matching;
    }

    int max_matching()
    {
        fill(match.begin(), match.end(), -1);
        int res = 0;
        for (int i = 0; i < n; ++i)
        {
            if (match[i] < 0)
            {
                if (dfs(i))
                    ++res;
            }
        }
        return res;
    }

    vector<pair<int, int> > matching_edges()
    {
        max_matching();
        vector<pair<int, int> > edges;
        for (int i = 0; i < n; ++i)
            if (match[i] >= 0)
                edges.push_back(make_pair(i, match[i] - n));
        return edges;
    }
};


/////////////////////////
// 最大流

// Ford Fulkerson フォード　フルカーソン
// O(F * E)
struct Edge
{
    int to, cap, rev;
    Edge(int to, int cap, int rev)
        : to(to), cap(cap), rev(rev) {}
    Edge() { Edge(0, 0, 0); }
};
class FordFulkerson
{
    vector<bool> used;
    int _dfs(int v, int t, int f)
    {
        if (v == t)
            return f;

        used[v] = true;
        for (int i = 0; i < edges[v].size(); ++i)
        {
            Edge& e = edges[v][i];
            if (!used[e.to] && e.cap > 0)
            {
                int d = _dfs(e.to, t, min(f, e.cap));
                if (d > 0)
                {
                    e.cap -= d;
                    edges[e.to][e.rev].cap += d;
                    return d;
                }
            }
        }
        return 0;
    }

public:
    vector<vector<Edge> > edges;
    int V;
    FordFulkerson(int V)
        : V(V), edges(V), used(V) {}

    void add_edge(int from, int to, int cap)
    {
        edges[from].push_back(Edge(to, cap, edges[to].size()));
        edges[to].push_back(Edge(from, 0, edges[from].size() - 1));
    }

    int dfs(int v, int t, int f)
    {
        fill(used.begin(), used.end(), false);
        return _dfs(v, t, f);
    }

    int max_flow(int s, int t)
    {
        int flow = 0;
        for (;;)
        {
            const int INF = 1000000000;
            int f = dfs(s, t, INF);
            if (f == 0)
                break;
            flow += f;
        }
        return flow;
    }
};


// template
// Dinic
// O(E * V^2), これよりも速く動作することがほとんど
template <typename T>
class Dinic
{
    struct DinicEdge
    {
        int to;
        T cap;
        int rev;
        DinicEdge(int to, T cap, int rev)
            : to(to), cap(cap), rev(rev) {}
        DinicEdge() {}
    };


    void bfs(int s)
    {
        fill(level.begin(), level.end(), -1);
        queue<int> q;
        q.push(s);
        level[s] = 0;
        while (!q.empty())
        {
            int v = q.front(); q.pop();
            for (int i = 0; i < (int)edges[v].size(); ++i)
            {
                DinicEdge& e = edges[v][i];
                if (e.cap > 0 && level[e.to] == -1)
                {
                    level[e.to] = level[v] + 1;
                    q.push(e.to);
                }
            }
        }
    }

public:
    int V;
    vector<vector<DinicEdge> > edges;
    vector<int> level;
    vector<int> iter;

    Dinic(int V)
        : V(V), edges(V), level(V), iter(V) { }

    void add_edge(int from, int to, T cap)
    {
        assert(0 <= from && from < V);
        assert(0 <= to && to < V);
        edges[from].push_back(DinicEdge(to, cap, (int)edges[to].size()));
        edges[to].push_back(DinicEdge(from, 0, (int)(edges[from].size() - 1)));
    }
    void add_undirected(int from, int to, T cap)
    {
        assert(0 <= from && from < V);
        assert(0 <= to && to < V);
        edges[from].push_back(DinicEdge(to, cap, (int)edges[to].size()));
        edges[to].push_back(DinicEdge(from, cap, (int)(edges[from].size() - 1)));
    }

    T dfs(int v, int t, T f)
    {
        if (v == t)
            return f;

        for (int& i = iter[v]; i < (int)edges[v].size(); ++i)
        {
            DinicEdge& e = edges[v][i];
            if (e.cap > 0 && level[v] < level[e.to])
            {
                T d = dfs(e.to, t, min(f, e.cap));
                if (d > 0)
                {
                    e.cap -= d;
                    edges[e.to][e.rev].cap += d;
                    return d;
                }
            }
        }
        return 0;
    }

    T max_flow(int s, int t)
    {
        assert(0 <= s && s < V);
        assert(0 <= t && t < V);

        T flow = 0;
        for (;;)
        {
            bfs(s);
            if (level[t] == -1)
                break;

            fill(iter.begin(), iter.end(), 0);
            const T INF = (T)(1LL << 60) | (1 << 29);
            for (T f; (f = dfs(s, t, INF)) > 0; )
                flow += f;
        }
        return flow;
    }
};

///// minimize cost to partition into 2 groups
// precondition
// each cost >= 0

// basic_cost = { (cost when i in a, cost when i in b) }

// check commentout!!!!!!!
// diff_cost = { ((i, j), cost) } | cost when i in a and j in b
// or
// diff_cost = { ((i, j), cost) } | cost when i and j are in different
template <typename T>
T min_cost_partition(const vector<pair<T, T> >& basic_cost, const vector<pair<pair<int, int>, T> >& diff_cost)
{
    const int n = basic_cost.size();
    Dinic<T> dinic(n + 2);
    const int src = n;
    const int sink = src + 1;

    for (int i = 0; i < n; ++i)
    {
        const T a_cost = basic_cost[i].first;
        const T b_cost = basic_cost[i].second;
        dinic.add_edge(i, sink, a_cost); // i in a
        dinic.add_edge(src, i, b_cost); // i in b
    }

    for (int di = 0; di < (int)diff_cost.size(); ++di)
    {
        const int i = diff_cost[di].first.first;
        const int j = diff_cost[di].first.second;
        const T cost = diff_cost[di].second;
        assert(0 <= i && i < n);
        assert(0 <= j && j < n);

        dinic.add_edge(i, j, cost); // i in a -> j in b

        // remove commentout if diff_cost means i and j are in different group
//         dinic.add_edge(a_offset + j, b_offset + i, cost); // j in a -> i in b
    }

    return dinic.max_flow(src, sink);
}



// 最小費用流, Primal Dual
// O(F * E * logV) or O(F * V^2)
template <typename FLOW, typename COST>
class PrimalDual
{
private:
    struct PrimalDualEdge
    {
        int to;
        FLOW cap;
        COST cost;
        int rev;
        PrimalDualEdge(int to, FLOW cap, COST cost, int rev)
            : to(to), cap(cap), cost(cost), rev(rev) { }
    };
public:
    int V;
    vector<vector<PrimalDualEdge> > g;

    PrimalDual(int V) : V(V), g(V) { }

    void add_edge(int from, int to, FLOW cap, COST cost)
    {
        g[from].push_back(PrimalDualEdge(to, cap, cost, g[to].size()));
        g[to].push_back(PrimalDualEdge(from, 0, -cost, g[from].size() - 1));
    }
    void add_undirected(int a, int b, FLOW cap, COST cost)
    {
        add_edge(a, b, cap, cost);
        add_edge(b, a, cap, cost);
    }

    COST min_cost_flow(int s, int t, FLOW f)
    {
        vector<COST> h(V);
        COST res = 0;
        int _f = f;
        while (_f > 0)
        {
            typedef pair<COST, int> _p;
            const COST _INF = (COST)((1LL << 60) | (1 << 29));
            priority_queue<_p, vector<_p>, greater<_p> > q;
            vector<COST> dis(V, _INF);
            vector<int> prevv(V), preve(V);
            dis[s] = 0;
            q.push(_p(0, s));
            while (!q.empty())
            {
                _p p = q.top(); q.pop();
                int v = p.second;
                COST cost = p.first;
                if (cost > dis[v])
                    continue;

                for (int i = 0; i < g[v].size(); ++i)
                {
                    PrimalDualEdge& e = g[v][i];
                    const COST _eps = 1e-10;
                    COST c = cost + e.cost + h[v] - h[e.to];
                    if (e.cap > 0 && c + _eps < dis[e.to])
                    {
                        dis[e.to] = c;
                        prevv[e.to] = v;
                        preve[e.to] = i;
                        q.push(_p(c, e.to));
                    }
                }
            }

            if (dis[t] == _INF)
            {
                // cant flow _f
                return -_INF;
            }

            for (int i = 0; i < V; ++i)
                h[i] += dis[i];

            FLOW d = _f;
            for (int i = t; i != s; i = prevv[i])
                d = min(d, g[prevv[i]][preve[i]].cap);
            _f -= d;
            res += d * h[t];
            for (int i = t; i != s; i = prevv[i])
            {
                PrimalDualEdge& e = g[prevv[i]][preve[i]];
                e.cap -= d;
                g[e.to][e.rev].cap += d;
            }
        }

        return res;
    }

    COST min_cost_flow_spfa(int s, int t, FLOW f)
    {
        COST res = 0;
        int _f = f;
        while (_f > 0)
        {
            const COST _INF = (COST)((1LL << 60) | (1 << 29));
            vector<COST> dis(V, _INF);
            vector<int> prevv(V), preve(V);

            vector<bool> in_q(V);
            queue<int> q;
            dis[s] = 0;
            q.push(s);
            in_q[s] = true;

            while (!q.empty())
            {
                int v = q.front(); q.pop();
                COST cost = dis[v];
                in_q[v] = false;

                for (int i = 0; i < g[v].size(); ++i)
                {
                    PrimalDualEdge& e = g[v][i];
                    const COST _eps = 1e-10;
                    COST c = cost + e.cost;
                    if (e.cap > 0 && c + _eps < dis[e.to])
                    {
                        dis[e.to] = c;
                        prevv[e.to] = v;
                        preve[e.to] = i;
                        if (!in_q[e.to])
                        {
                            q.push(e.to);
                            in_q[e.to] = true;
                        }
                    }
                }
            }

            if (dis[t] == _INF)
            {
                // cant flow _f
                return -_INF;
            }

            FLOW d = _f;
            for (int i = t; i != s; i = prevv[i])
                d = min(d, g[prevv[i]][preve[i]].cap);
            _f -= d;
            res += d * dis[t];
            for (int i = t; i != s; i = prevv[i])
            {
                PrimalDualEdge& e = g[prevv[i]][preve[i]];
                e.cap -= d;
                g[e.to][e.rev].cap += d;
            }
        }

        return res;
    }
};


///////////////////////////////////////////////////////////////
// 安定マッチング、安定結婚問題
pair<vector<int>, vector<int> > stable_matching(const vector< vector<int> >& a_order, const vector< vector<int> >& b_order)
{
    assert(a_order.size() == b_order.size());

    const int n = a_order.size();
    vector<vector<int> > b_prefer(n, vector<int>(n + 1, n));
    vector<int> b_match(n, n), a_proposed(n);
    for (int b = 0; b < n; ++b)
        for (int i = 0; i < n; ++i)
            b_prefer[b][b_order[b][i]] = i;

    for (int a_ = 0; a_ < n; ++a_)
    {
        for (int a = a_; a < n; )
        {
            int b = a_order[a][a_proposed[a]++];
            if (b_prefer[b][a] < b_prefer[b][b_match[b]])
                swap(a, b_match[b]);
        }
    }

    vector<int> a_match(n);
    for (int i = 0; i < n; ++i)
        a_match[b_match[i]] = i;
    return make_pair(a_match, b_match);
}


///////////////////////////////////////////////////////////////
// SCC(Strongly Connected Component), 強連結成分分解
class SCC
{
private:
    vector<int> vs;	// 帰りの順番, 後ろのほう要素は始点側
    vector<bool> used;

    void dfs(int v)
    {
        used[v] = true;
        for (int i = 0; i < (int)g[v].size(); ++i)
            if (!used[g[v][i]])
                dfs(g[v][i]);
        vs.push_back(v);
    }

    void rdfs(int v, int k)
    {
        used[v] = true;
        component[v] = k;
        for (int i = 0; i < (int)rg[v].size(); ++i)
            if (!used[rg[v][i]])
                rdfs(rg[v][i], k);
    }

public:
    int V;
    vector<vector<int> > g;
    vector<vector<int> > rg;
    vector<int> component;
    int num_components;

    SCC(int V)
        : vs(V), used(V), V(V), g(V), rg(V), component(V), num_components(-1) { }

    void add_edge(int from, int to)
    {
        g[from].push_back(to);
        rg[to].push_back(from);
    }

    // 強連結成分の数を返す
    // O(V + E)
    // グラフの始点側の値(component[v])は小さい
    // 末端は大きい値, 最末端(component[v] == scc() - 1)
    int scc()
    {
        vs.clear();
        fill(used.begin(), used.end(), false);
        for (int i = 0; i < V; ++i)
            if (!used[i])
                dfs(i);

        fill(used.begin(), used.end(), false);
        int k = 0;
        for (int i = vs.size() - 1; i >= 0; --i)
            if (!used[vs[i]])
                rdfs(vs[i], k++);
        return num_components = k;
    }

    // scc_g: scc graph
    // original_nodes: the numbers of original graph in scc_g[i]
    void build_graph(vector<vector<int> >& scc_g, vector<vector<int> >& original_nodes)
    {
        if (num_components == -1)
            scc();

        scc_g.clear();
        scc_g.resize(num_components);
        original_nodes.clear();
        original_nodes.resize(num_components);


        for (int i = 0; i < V; ++i)
            original_nodes[component[i]].push_back(i);

        for (int i = 0; i < V; ++i)
            for (int j = 0; j < (int)g[i].size(); ++j)
                if (component[i] != component[g[i][j]]) // remove self loop
                    scc_g[component[i]].push_back(component[g[i][j]]);

        // remove duplicated edges (O(ElogE)... decrease complexity???)
        for (int i = 0; i < num_components; ++i)
        {
            sort(scc_g[i].begin(), scc_g[i].end());
            scc_g[i].erase(unique(scc_g[i].begin(), scc_g[i].end()), scc_g[i].end());
        }
    }
};

// memo
// x = x or x = ~x -> x
// a and b = (a or a) and (b or b)
class TwoSAT
{
private:
    SCC scc;
public:
    int n;
    
    TwoSAT(int n)
        :  scc(2 * n), n(n) {}

    int NOT(int i) { return i < n ? i + n : i - n; }

    // raw method by adding edge directly
    // a -> b (~a or b)
    void add_edge(int a, int b)
    {
        scc.add_edge(a, b);
    }

    // a or b (~a -> b and ~b -> a)
    void add_or(int a, int b)
    {
        add_edge(NOT(a), b);
        add_edge(NOT(b), a);
    }

    // a and b
    void add_and(int a, int b)
    {
        assign(a, true);
        assign(b, true);
    }

    // do a = val
    void assign(int a, bool val)
    {
        if (!val)
            a = NOT(a); // a = false to ~a = true
        add_edge(NOT(a), a); // a, a or a, ~a -> a
    }

    bool satisfy()
    {
        scc.scc();
        for (int i = 0; i < n; ++i)
            if (scc.component[i] == scc.component[NOT(i)])
                return false;
        return true;
    }

    bool val(int i)
    {
        return scc.component[i] > scc.component[NOT(i)];
    }
};

// LCA(Lowest Common Ancestor)
class LCA
{
    friend class LCABuilder;
    friend LCA build_lca(vector<vector<int> >& g, int root);

private:
    int max_log;

    LCA(vector<int>& depth, vector<vector<int> >& parent, int max_log)
        : max_log(max_log), depth(depth), parent(parent) { }

public:
    vector<int> depth;
    vector<vector<int> > parent;

    int lca(int u, int v)
    {
        if (depth[u] > depth[v])
            swap(u, v);

        for (int k = 0; k < max_log; ++k)
            if ((depth[v] - depth[u]) >> k & 1)
                v = parent[k][v];

        if (u == v)
            return u;

        for (int k = max_log - 1; k >= 0; --k)
        {
            if (parent[k][u] != parent[k][v])
            {
                u = parent[k][u];
                v = parent[k][v];
            }
        }

        return parent[0][u];
    }
};
class LCABuilder
{
    friend LCA build_lca(vector<vector<int> >& g, int root);

private:
    int root, n;
    vector<vector<int> >& g;

    int max_log;
    vector<int> depth;
    vector<vector<int> > parent;

    LCABuilder(vector<vector<int> >& graph, int root)
        : root(root), n(graph.size()), g(graph), depth(n)
    {
        max_log = 0;
        for (int i = 1; i <= n; i *= 2)
            ++max_log;
        parent = vector<vector<int> >(max_log, vector<int>(n));

        init();
    }

    void calc_depth(int v, int p, int d)
    {
        parent[0][v] = p;
        depth[v] = d;

        for (int i = 0; i < (int)g[v].size(); ++i)
            if (g[v][i] != p)
                calc_depth(g[v][i], v, d + 1);
    }

    void init()
    {
        calc_depth(root, -1, 0);

        for (int k = 0; k + 1 < max_log; ++k)
        {
            for (int v = 0; v < n; ++v)
            {
                if (parent[k][v] < 0)
                    parent[k + 1][v] = -1;
                else
                    parent[k + 1][v] = parent[k][parent[k][v]];
            }
        }
    }
};
LCA build_lca(vector<vector<int> >& g, int root)
{
    LCABuilder builder(g, root);
    return LCA(builder.depth, builder.parent, builder.max_log);
}



int subtree_lower[ten(5)], subtree_upper[ten(5)];
void _euler_tour(int pos, int& k)
{
    subtree_lower[pos] = k++;
    for (int to : g[pos])
        if (to != parent[pos])
            _euler_tour(to, k);
    subtree_upper[pos] = k;
}
void euler_tour()
{
    int k = 0;
    _euler_tour(0, k);
}
////////////////////////////////////////////////////////////////////////////////
// 行列

typedef vector<ll> Vec;
typedef vector<Vec> Matrix;

Matrix mul(const Matrix& a, const Matrix& b, ll mod)
{
    assert(a[0].size() == b.size());

    Matrix c(a.size(), Vec(b[0].size()));
    for (int i = 0; i < (int)a.size(); ++i)
        for (int k = 0; k < (int)b.size(); ++k)
            for (int j = 0; j < (int)b[0].size(); ++j)
                c[i][j] = (c[i][j] + a[i][k] * b[k][j]) % mod;
    return c;
}
Matrix pow(Matrix a, ll n, ll mod)
{
    assert(a.size() == a[0].size());

    Matrix b(a.size(), Vec(a.size()));
    for (int i = 0; i < (int)b.size(); ++i)
        b[i][i] = 1;
    while (n > 0)
    {
        if (n & 1)
            b = mul(b, a, mod);
        a = mul(a, a, mod);
        n >>= 1;
    }
    return b;
}

Matrix sum_pow(const Matrix& a, ll k, ll mod)
{
    ll n = a.size();
    // | a 0 |
    // | I I |
    Matrix b(n * 2, Vec(n * 2));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
            b[i][j] = a[i][j];
        b[n + i][i] = b[n + i][n + i] = 1;
    }

    b = pow(b, k + 1, mod);

    Matrix res(n, Vec(n));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            res[i][j] = b[n + i][j];
            if (i == j)
                res[i][j] = (res[i][j] - 1 + mod) % mod;
        }
    }
    return res;
}


void print(const Matrix& a, int width = 5)
{
    for (int i = 0; i < (int)a.size(); ++i)
    {
        for (int j = 0; j < (int)a[0].size(); ++j)
        {
            cout.width(width);
            cout << a[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}
Matrix transpose(const Matrix& a)
{
    Matrix b(a[0].size(), Vec(a.size()));
    for (int i = 0; i < (int)b.size(); ++i)
        for (int j = 0; j < (int)b[0].size(); ++j)
            b[i][j] = a[j][i];
    return b;
}

Matrix add(const Matrix& a, const Matrix& b, ll mod)
{
    assert(a.size() == b.size() && a[0].size() == b[0].size());

    Matrix c(a.size(), Vec(a[0].size()));
    for (int i = 0; i < (int)c.size(); ++i)
        for (int j = 0; j < (int)c[0].size(); ++j)
            c[i][j] = (a[i][j] + b[i][j]) % mod;
    return c;
}
Matrix sub(const Matrix& a, const Matrix& b, ll mod)
{
    assert(a.size() == b.size() && a[0].size() == b[0].size());

    Matrix c(a.size(), Vec(a[0].size()));
    for (int i = 0; i < (int)c.size(); ++i)
        for (int j = 0; j < (int)c[0].size(); ++j)
            c[i][j] = (a[i][j] - b[i][j] + mod) % mod;
    return c;
}

Matrix unit_matrix(int n)
{
    Matrix a(n, Vec(n));
    for (int i = 0; i < n; ++i)
        a[i][i] = 1;
    return a;
}
// Vec
Vec mul(const Matrix& a, const Vec& v, ll mod)
{
    assert(a[0].size() == v.size());

    Vec x(a.size());
    for (int i = 0; i < x.size(); ++i)
        for (int j = 0; j < v.size(); ++j)
            x[i] = (x[i] + a[i][j] * v[j]) % mod;
    return x;
}

ll det(Matrix a, const ll mod)
{
    const int n = a.size();
    assert(n == a[0].size());

    for (int i = 0; i < n; ++i)
    {
        int pivot = -1;
        for (int j = i; j < n; ++j)
        {
            if (a[j][i] > 0)
            {
                pivot = j;
                break;
            }
        }
        if (pivot == -1)
            return 0;
        swap(a[i], a[pivot]);

        const ll inv = mod_inverse(a[i][i], mod);
        for (int j = i + 1; j < n; ++j)
        {
            const ll coef = a[j][i] * inv % mod;
            for (int k = i; k < n; ++k)
            {
                a[j][k] -= coef * a[i][k] % mod;
                if (a[j][k] < 0)
                    a[j][k] += mod;
            }
        }
    }

    ll res = 1;
    for (int i = 0; i < n; ++i)
        (res *= a[i][i]) %= mod;
    return res;
}


// 全域木数え上げとか (ARC18 D)
vector<map<int, vector<int>>> extract_connected_graphs(const map<int, vector<int>>& g)
{
    vector<map<int, vector<int>>> connected_gs;
    set<int> visit;
    for (auto& it : g)
    {
        int v = it.first;
        if (visit.count(v))
            continue;

        map<int, vector<int>> connected_g;
        queue<int> q;
        q.push(v);
        visit.insert(v);
        connected_g[v] = vector<int>();
        while (!q.empty())
        {
            int cur = q.front();
            q.pop();

            for (int to : g.find(cur)->second) // g[cur]だとコンパイラさんが怒る
            {
                connected_g[cur].push_back(to);
                if (!visit.count(to))
                {
                    q.push(to);
                    visit.insert(to);
                }
            }
        }
        connected_gs.push_back(connected_g);
    }
    return connected_gs;
}
vector<vector<int>> reindex(const map<int, vector<int>>& g)
{
    vector<int> vs;
    for (auto& it : g)
        vs.push_back(it.first);

    vector<vector<int>> res_g(vs.size());
    for (auto& it : g)
    {
        int v = it.first;
        int a = lower_bound(all(vs), v) - vs.begin();
        for (int to : it.second)
        {
            int b = lower_bound(all(vs), to) - vs.begin();
            res_g[a].push_back(b);
        }
    }
    return res_g;
}
ll count_spanning_tree(const vector<vector<int>>& g, const ll mod)
{
    int n = g.size();
    Matrix a(n - 1, Vec(n - 1));
    rep(i, n - 1)
    {
        a[i][i] = g[i].size();
        for (int j : g[i])
        {
            if (j < n - 1)
                --a[i][j];
        }
    }

    rep(i, n - 1) rep(j, n - 1)
        (a[i][j] += mod) %= mod;
    return det(a, mod);
}





////////////////////////////////
// fibonacchi by matrix
ll fib(ll n, ll mod)
{
    Matrix a(2, Vec(2));
    a[0][0] = a[0][1] = a[1][0] = 1;
    Matrix b = pow(a, n, mod);
    return b[1][0];
}



//////////////////////
// データ構造
// BIT, FenwickTree
// 0-indexed
template <typename T>
class FenwickTree
{
public:
    int n;
    vector<T> a;
    FenwickTree(int n) : n(n), a(n + 1) {}
    FenwickTree() { }

    void add(int i, T x)
    {
        ++i;
        assert(i > 0);

        while (i <= n)
        {
            a[i] += x;
            i += i & -i;
        }
    }

    // [0, i)
    T sum(int i) const
    {
        T res = 0;
        while (i > 0)
        {
            res += a[i];
            i -= i & -i;
        }
        return res;
    }

    // [low, high)
    T range_sum(int low, int high) const { return sum(high) - sum(low); }
    T at(int i) const { return sum(i + 1) - sum(i); }
    void assign(int i, T x) { add(i, x - at(i)); }
};
template <typename T>
int lower_bound(const FenwickTree<T>& ft, T sum)
{
    int low = -1, high = ft.n;
    while (low + 1 < high)
    {
        int mid = (low + high) / 2;
        if (ft.sum(mid + 1) >= sum)
            high = mid;
        else
            low = mid;
    }
    return high;
}
// precondition: all elements are 0 or 1
template <typename T>
int find_first_zero(const FenwickTree<T>& ft)
{
    int low = -1, high = ft.n;
    while (high - low > 1)
    {
        int mid = (low + high) / 2;
        if (ft.sum(mid) == mid)
            low = mid;
        else
            high = mid;
    }
    return high;
}

// Segment Tree
// RMQ
template <typename T>
class RMQ
{
private:
    vector<T> data;
    int n;
    const T UNDEF;
 
    T _query(int a, int b, int k, int l, int r)
    {
        if (r <= a || b <= l)
            return UNDEF;
 
        if (a <= l && r <= b)
            return data[k];
        else
        {
            T left = _query(a, b, 2 * k + 1, l, (l + r) / 2);
            T right = _query(a, b, 2 * k + 2, (l + r) / 2, r);
            return min(left, right);
        }
    }
public:
 
    RMQ(int n, T UNDEF)
        : UNDEF(UNDEF)
    {
        this->n = 1;
        while (this->n < n)
            this->n *= 2;
        data = vector<T>(this->n * 2, UNDEF);
    }
 
    void update(int k, T a)
    {
        k += n - 1;
        data[k] = a;
        while (k > 0)
        {
            k = (k - 1) / 2;
            data[k] = min(data[2 * k + 1], data[2 * k + 2]);
        }
    }

    // [a, b)
    T query(int a, int b) { return _query(a, b, 0, 0, n); }
};

class XorSegTree
{
private:
    int size;
    vector<int> one;
    vector<bool> flipped;

    void flip(int a, int b, int k, int l, int r)
    {
        if (r <= a || b <= l)
            return;

        if (a <= l && r <= b)
        {
            flipped[k] = !flipped[k];
            one[k] = (r - l) - one[k];
        }
        else
        {
            int mid = (l + r) / 2;
            flip(a, b, 2 * k + 1, l, mid);
            flip(a, b, 2 * k + 2, mid, r);

            int res = one[2 * k + 1] + one[2 * k + 2];
            if (flipped[k])
                res = (r - l) - res;
            one[k] = res;
        }
    }

    int count(int a, int b, int k, int l, int r)
    {
        if (r <= a || b <= l)
            return 0;

        if (a <= l && r <= b)
            return one[k];
        else
        {
            int mid = (l + r) / 2;
            int res = count(a, b, 2 * k + 1, l, mid) + count(a, b, 2 * k + 2, mid, r);
            if (flipped[k])
                res = (min(b, r) - max(a, l)) - res;
            return res;
        }
    }

public:
    void init(int n)
    {
        size = 1;
        while (size < n)
            size *= 2;
        flipped.resize(2 * size);
        one.resize(2 * size);
        for (int i = 0; i < 2 * size; ++i)
        {
            flipped[i] = false;
            one[i] = 0;
        }
    }

    void flip(int a, int b)
    {
        flip(a, b, 0, 0, size);
    }

    int count(int a, int b)
    {
        return count(a, b, 0, 0, size);
    }
};

template <typename T>
class CountSegTree
{
public:
    CountSegTree(){}
    CountSegTree(vector<T> a, T UNDEF = 1e9 + 100)
    {
        int n = 1;
        while (n < a.size())
            n *= 2;
        while (a.size() < n)
            a.push_back(UNDEF);

        data.resize(2 * n);
        init(a, 0);
    }

    int count_less(int l, int r, const T& num)
    {
        return count_less(l, r, num, 0, 0, data.size() / 2);
    }

private:
    vector<vector<T>> data;

    void init(vector<T>& a, int k)
    {
        const int n = a.size();
        if (n <= 1)
        {
            data[k] = a;
            return;
        }

        vector<T> b(a.begin(), a.begin() + n / 2);
        init(b, 2 * k + 1);
        vector<T> c(a.begin() + n / 2, a.end());
        init(c, 2 * k + 2);

        for (int ai = 0, bi = 0, ci = 0; ai < n; )
        {
            if (bi < b.size() && (ci == c.size() || b[bi] <= c[ci]))
                a[ai++] = b[bi++];
            else
                a[ai++] = c[ci++];
        }

        data[k] = a;
    }

    int count_less(int l, int r, const T& num, int k, int a, int b)
    {
        if (r <= a || b <= l)
            return 0;
        else if (l <= a && b <= r)
        {
            const vector<T>& d = data[k];
            return lower_bound(d.begin(), d.end(), num) - d.begin();
        }
        else
        {
            return count_less(l, r, num, 2 * k + 1, a, (a + b) / 2)
                 + count_less(l, r, num, 2 * k + 2, (a + b) / 2, b);
        }
    }
};

class HotelSegTree
{
private:
    struct Seg
    {
        bool empty;
        int max_space;
        int left_space;
        int right_space;
        int lazy;
        Seg()
            : empty(false), max_space(0),
              left_space(0), right_space(0),
              lazy(-1)
        {
        }
    };

    vector<Seg> seg;
    int SIZE;

    static const int IN = 1, OUT = 0;
    void update(int a, int b, int ope, int k, int l, int r)
    {
        const int space = r - l;
        const int mid = (l + r) / 2;
        const int lk = 2 * k + 1, rk = 2 * k + 2;

        if (seg[k].lazy != -1)
        {
            int lazy_ope = seg[k].lazy;
            seg[k].lazy = -1;
            update(l, r, lazy_ope, k, l, r);
        }

        if (b <= l || r <= a || space == 0)
            return;

        if (a <= l && r <= b)
        {
            seg[k].max_space = seg[k].left_space = seg[k].right_space
                = ope == OUT ? space : 0;
            seg[k].empty = ope == OUT;

            if (k < SIZE)
                seg[lk].lazy = seg[rk].lazy = ope;
        }
        else
        {
            update(a, b, ope, lk, l, mid);
            update(a, b, ope, rk, mid, r);

            const int cat_space = seg[lk].right_space + seg[rk].left_space;
            seg[k].max_space = max(cat_space, max(seg[lk].max_space, seg[rk].max_space));
            seg[k].empty = seg[k].max_space == space;

            seg[k].left_space = seg[lk].left_space;
            if (seg[lk].empty)
                seg[k].left_space += seg[rk].left_space;

            seg[k].right_space = seg[rk].right_space;
            if (seg[rk].empty)
                seg[k].right_space += seg[lk].right_space;
        }
    }

    int find(int need_space, int k, int l, int r)
    {
        if (seg[k].left_space >= need_space)
            return l;

        int mid = (l + r) / 2;
        int lk = 2 * k + 1, rk = 2 * k + 2;
        if (seg[lk].max_space >= need_space)
            return find(need_space, lk, l, mid);
        else if (seg[lk].right_space + seg[rk].left_space >= need_space)
            return mid - seg[lk].right_space;
        else
            return find(need_space, rk, mid, r);
    }

public:
    HotelSegTree(int n)
    {
        SIZE = 1;
        while (SIZE < n)
            SIZE *= 2;
        seg.resize(2 * SIZE);
    }

    void out(int l, int r)
    {
        update(l, r, OUT, 0, 0, SIZE);
    }

    void in(int l, int r)
    {
        update(l, r, IN, 0, 0, SIZE);
    }

    // 連続でneed_space空いている区間の左端のindex(複数ある場合最小のindex)
    int find(int need_space)
    {
        if (seg[0].max_space < need_space)
            return -1;
        else
            return find(need_space, 0, 0, SIZE);
    }
};



////////////
// 二次元累積和
#define CALC_SQ_SUM(sum, data, x, y) (sum[y][x + 1] + (sum[y + 1][x] - sum[y][x]) + data[y][x])
#define GET_SQ_SUM(sum, x, y, w, h) (sum[y + h][x + w] - (sum[y][x + w] + sum[y + h][x]) + sum[y][x])

template <class T, class U>
void calc_sq_sum(T& sum, const U& data, int w, int h)
{
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            sum[i + 1][j + 1] = sum[i][j + 1] + sum[i + 1][j] - sum[i][j] + data[i][j];
}
template <class T, int H>
T get_sq_sum(T (*sum)[H], int x1, int y1, int x2, int y2)
{
    if (x1 > x2 || y1 > y2)
        return 0;
    return sum[y2 + 1][x2 + 1] + sum[y1][x1] - sum[y1][x2 + 1] - sum[y2 + 1][x1];
}


///////////////////////////
// 浮動小数点誤差
double calc_eps(double n)
{
    n += DBL_MIN;
    double res = n;
    double low = n, high = 2 * n;
    double a = n, b = 2 * n;
    if (low > DBL_MAX)
        return -1;
    queue<double> q;
    for (int i = 0; i < 3; ++i)
        q.push(high);
    while (q.back() != high - low)
    {
        q.pop();
        q.push(high - low);
        high = (low + high) / 2;
        //cout << "(" << low << ", " << high << ")" << " " << high - low << endl;
    }
    return q.front();
}
#include <iomanip>
void p(double n, double eps)
{
    cout.setf(ios::scientific);
    cout << setprecision(1) << n << ": " << setprecision(6) << eps << endl;
}
void print_eps(int low_e, int high_e)
{
    p(0, calc_eps(0));
    for (int i = low_e; i <= high_e; ++i)
    {
        double n = pow(10.0, (double)i);
        p(n, calc_eps(n));
    }
}

// print vector
template <class T>
void print_v(const vector<T>& v)
{
    cout << "{ " << v[0];
    for (int i = 1; i < v.size(); ++i)
        cout << ", " << v[i];
    cout << " }" << endl;
}

// 乱数
class Random
{
private:
    unsigned int  x, y, z, w;
public:
    Random(unsigned int x
             , unsigned int y
             , unsigned int z
             , unsigned int w)
        : x(x), y(y), z(z), w(w) { }
    Random() 
        : x(123456789), y(362436069), z(521288629), w(88675123) { }
    Random(unsigned int seed)
        : x(123456789), y(362436069), z(521288629), w(seed) { }

    unsigned int next()
    {
        unsigned int t = x ^ (x << 11);
        x = y;
        y = z;
        z = w;
        return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    }

    int next_int() { return next(); }

    // [0, upper)
    int next_int(int upper) { return next() % upper; }

    // [low, high]
    int next_int(int low, int high) { return next_int(high - low + 1) + low; }

    double next_double(double upper) { return upper * next() / UINT_MAX; }
    double next_double(double low, double high) { return next_double(high - low) + low; }

    template <typename T>
    int select(const vector<T>& ratio)
    {
        T sum = accumulate(ratio.begin(), ratio.end(), (T)0);
        T v = next_double(sum) + (T)1e-6;
        for (int i = 0; i < (int)ratio.size(); ++i)
        {
            v -= ratio[i];
            if (v <= 0)
                return i;
        }
        return 0;
    }
};

ull rdtsc()
{
#ifdef __amd64
    ull a, d;
    __asm__ volatile ("rdtsc" : "=a" (a), "=d" (d)); 
    return (d<<32) | a;
#else
    ull x;
    __asm__ volatile ("rdtsc" : "=A" (x)); 
    return x;
#endif
}
#ifdef LOCAL
const double CYCLES_PER_SEC = 3.30198e9;
#else
const double CYCLES_PER_SEC = 2.5e9;
#endif
double get_absolute_sec()
{
    return (double)rdtsc() / CYCLES_PER_SEC;
}
#ifdef _MSC_VER
#include <Windows.h>
    double get_ms() { return (double)GetTickCount64() / 1000; }
#else
#include <sys/time.h>
    double get_ms() { struct timeval t; gettimeofday(&t, NULL); return (double)t.tv_sec * 1000 + (double)t.tv_usec / 1000; }
#endif

#define USE_RDTSC
class Timer
{
private:
    double start_time;
    double elapsed;

#ifdef USE_RDTSC
    double get_sec() { return get_absolute_sec(); }
#else
    double get_sec() { return get_ms() / 1000; }
#endif

public:
    Timer() {}

    void start() { start_time = get_sec(); }
    double get_elapsed() { return elapsed = get_sec() - start_time; }
};

void calc_cycles_per_sec()
{
    Timer timer;
    timer.start();

    ull start_rdtsc = rdtsc();
    double start = get_ms();
    int loops = 0;
    const double LOOP_SEC = 100;
    while (get_ms() - start < LOOP_SEC * 1000)
        ++loops;
    dump(loops);
    ull end_rdtsc = rdtsc();
    dump(start_rdtsc);
    dump(end_rdtsc);
    cerr << "CYCLES_PER_SEC = " << (end_rdtsc - start_rdtsc) / LOOP_SEC << endl;

    dump(timer.get_elapsed());
}
void count_timeof_call()
{
    double start = get_ms();
    int loops = 0;
    const double LOOP_SEC = 10;
    while (get_ms() - start < LOOP_SEC * 1000)
        ++loops;
    dump(loops);
    cerr << "timeof calls / sec = " << loops / LOOP_SEC << endl;
}
void count_rdtsc_call()
{
    cerr << "rdtsc" << endl;
    double start = get_absolute_sec();
    int loops = 0;
    const double LOOP_SEC = 10;
    while (get_absolute_sec() - start < LOOP_SEC)
        ++loops;
    dump(loops);
    cerr << "rdtsc calls / sec = " << loops / LOOP_SEC << endl;
}


// サイコロ
enum Face { TOP, BOTTOM, FRONT, BACK, LEFT, RIGHT };
template <class T>
class TDice
{
private:
    int id[6];
    T var[6];

    void init()
    {
        id[TOP] = 0; id[BOTTOM] = 5;
        id[FRONT] = 1; id[BACK] = 4;
        id[LEFT] = 2; id[RIGHT] = 3;
    }

    void roll(Face a, Face b, Face c, Face d)
    {
        int t = id[a];
        id[a] = id[b]; id[b] = id[c];
        id[c] = id[d]; id[d] = t;
    }
public:
    TDice()
    {
        init();
        for (int i = 0; i < 6; ++i)
            var[id[i]] = id[i] + 1;
    }

    T& operator[](Face f) { return var[id[f]]; }
    const T& operator[](Face f) const { return var[id[f]]; }
    bool operator==(const TDice<T>& b) const
    {
        const TDice<T>& a = *this;
        for (int i = 0; i < 6; ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }

    // FRONT
    void roll_x() { roll(TOP, BACK, BOTTOM, FRONT); }
    // RIGHT
    void roll_y() { roll(TOP, LEFT, BOTTOM, RIGHT); }
    // left rotate
    void roll_z() { roll(FRONT, RIGHT, BACK, LEFT); }
    // BACK
    void rev_roll_x() { for (int i = 0; i < 3; ++i) roll_x(); }
    // LEFT
    void rev_roll_y() { for (int i = 0; i < 3; ++i) roll_y(); }
    // right rotate
    void rev_roll_z() { for (int i = 0; i < 3; ++i) roll_z(); }

    vector<TDice> all_rolls()
    {
        vector<TDice> res;
        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                res.push_back(*this);
                roll_z();
            }
            if (i & 1) roll_y();
            else roll_x();
        }
        return res;
    }
};
typedef TDice<int> Dice;


template <class bit_type, class T>
bit_type bit_encode(int width, const T& a, int n)
{
    bit_type bit = 0;
    for (int i = 0; i < n; ++i)
    {
        assert(((bit << width) & a[i]) == 0); // wrong width
        bit = (bit << width) | a[i];
    }
    return bit;
}
template <class bit_type, class T>
bit_type bit_encode(int width, const vector<T>& a)
{
    return bit_encode<bit_type>(width, a, a.size());
}
template <class T, class bit_type>
void bit_decode(bit_type enc_bit, int width, T& a, int n)
{
    bit_type mask = (((bit_type)1) << width) - 1;
    for (int i = n - 1; i >= 0; --i)
    {
        a[i] = enc_bit & mask;
        enc_bit >>= width;
    }
}
template <class T, class bit_type>
vector<T> bit_decode(bit_type enc_bit, int width, int n)
{
    vector<T> a(n);
    bit_decode(enc_bit, width, a, n);
    return a;
}

int balanced_tree_height(ll nodes)
{
    int h = 0;
    for (ll i = 1; i <= nodes; i *= 2)
        ++h;
    return h;
}


bool overflow(ll a, ll b)
{
    if (a > b)
        swap(a, b);
    if (a == 0)
        return false;
    return b > LLONG_MAX / a;
}


// FFT
typedef double fft_t;
typedef complex<fft_t> fft_C;
void _fft(vector<fft_C>& a, int sign)
{
    const fft_C I(0, 1);

    const int n = a.size();
    double theta = sign * 2 * PI / n;

    for (int m = n; m >= 2; m >>= 1)
    {
        int mh = m >> 1;
        for (int i = 0; i < mh; ++i)
        {
            fft_C w = exp(i * theta * I);
            for (int j = i; j < n; j += m)
            {
                int k = j + mh;
                fft_C x = a[j] - a[k];
                a[j] += a[k];
                a[k] = w * x;
            }
        }
        theta *= 2;
    }
    int i = 0;
    for (int j = 1; j < n - 1; ++j)
    {
        for (int k = n >> 1; k > (i ^= k); k >>= 1)
            ;
        if (j < i)
            swap(a[i], a[j]);
    }
}
void fft(vector<fft_C>& a)
{
    _fft(a, 1);
}
void ifft(vector<fft_C>& a)
{
    _fft(a, -1);
    const int n = a.size();
    for (int i = 0; i < n; ++i)
        a[i] /= n;
}
template <typename T>
void _poly_mul(const vector<T>& a, const vector<T>& b, vector<fft_t>& res)
{
    const int n = a.size();

    int m = 1;
    while (m < 2 * n)
        m <<= 1;

    vector<fft_C> fa(m, fft_C(0, 0)), fb(m, fft_C(0, 0));
    for (int i = 0; i < n; ++i)
        fa[i] = a[i];
    for (int i = 0; i < n; ++i)
        fb[i] = b[i];

    fft(fa);
    fft(fb);

    for (int i = 0; i < m; ++i)
        fa[i] *= fb[i];
    ifft(fa);

    res.resize(m);
    for (int i = 0; i < m; ++i)
        res[i] = fa[i].real();
}
template <typename T>
vector<fft_t> poly_mul(const vector<T>& a, const vector<T>& b)
{
    vector<fft_t> c;
    _poly_mul(a, b, c);
    return vector<fft_t>(c.begin(), c.end());
}
template <typename T>
vector<T> poly_mul_int(const vector<T>& a, const vector<T>& b)
{
    vector<fft_t> c;
    _poly_mul(a, b, c);

    vector<T> res(c.size());
    for (int i = 0; i < (int)c.size(); ++i)
        res[i] = (T)(c[i] + 0.5);
    return res;
}


// gauss jordan, for only bool (これ大丈夫だっけ？)
bool gauss(const vector<vector<int> >& _a, const vector<int>& b)
{
    const int n = _a.size();
    vector<vector<int> > a(n, vector<int>(n + 1));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            a[i][j] = _a[i][j];
    for (int i = 0; i < n; ++i)
        a[i][n] = b[i];

    for (int i = 0; i < n; ++i)
    {
        int pivot = -1;
        for (int j = i; j < n; ++j)
        {
            if (a[j][i] != 0)
            {
                pivot = j;
                break;
            }
        }
        if (pivot == -1)
            continue;

        swap(a[i], a[pivot]);

        for (int j = 0; j < n; ++j)
        {
            if (i != j && a[j][i] != 0)
            {
                for (int k = i; k <= n; ++k)
                    a[j][k] = (a[j][k] + a[i][k]) % 2;
            }
        }
    }

    for (int i = 0; i < n; ++i)
    {
        if (a[i][i] == 0 && a[i][n] == 1)
            return false;
    }
    return true;
}

typedef vector<double> Vec;
typedef vector<Vec> Matrix;
// solve Ax = b
// return empty if no solutions or not unique solution
Vec gauss_jordan(const Matrix& A, const Vec& b)
{
    int n = A.size();
    Matrix B(n, Vec(n + 1));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            B[i][j] = A[i][j];
    for (int i = 0; i < n; ++i)
        B[i][n] = b[i];

    for (int i = 0; i < n; ++i)
    {
        int pivot = i;
        for (int j = i; j < n; ++j)
            if (abs(B[j][i]) > abs(B[pivot][i]))
                pivot = j;
        swap(B[i], B[pivot]);

        // no or not uniq
        const double eps = 1e-8;
        if (abs(B[i][i]) < eps)
            return Vec();

        for (int j = i + 1; j <= n; ++j)
            B[i][j] /= B[i][i];
        for (int j = 0; j < n; ++j)
        {
            if (i != j)
            {
                for (int k = i + 1; k <= n; ++k)
                    B[j][k] -= B[j][i] * B[i][k];
            }
        }
    }
    Vec x(n);
    for (int i = 0; i < n; ++i)
        x[i] = B[i][n];
    return x;
}



//// Date
// __days[month - 1]
static const int __days[] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

// __sum_days[month] -> sum_days(1, 2, .., month - 1)
static const int __sum_days[] = { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 };

const int days_of_year = 365;

class Date
{
private:
    int _year, _month, _day;    // _month, day: 1-based

public:
//////// class method

    // definition
    // 0 year is leap year
    // n(n < 0) year is NOT leap year

    static bool is_leap(int year)
    {
        return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    }

    // [0, until]
    static int leap_years(int until)
    {
        if (until < 0)
            return 0;
        int res = (until / 4) - (until / 100) + (until / 400);
        ++res; // by definition: 0 year is leap
        return res;
    }

    static int days(int year)
    {
        return is_leap(year) ? days_of_year + 1 : days_of_year;
    }
    static int days(int year, int month)
    {
        if (is_leap(year) && month == 2)
            return __days[month - 1] + 1;
        else
            return __days[month - 1];
    }


//////// instance method
    Date(int year, int month, int day)
        : _year(year), _month(month), _day(day)
    {
        assert(1 <= month && month <= 12);
        assert(1 <= day && day <= days(year, month));
    }

    int year() const { return _year; }
    int month() const { return _month; }
    int day() const { return _day; }

    // 0000/01/01 -> 1
    ll to_n() const
    {
        ll res = 0;
        int leap = leap_years(year() - 1);
        int not_leap = year() - leap;
        res += not_leap * (ll)days_of_year + leap * (ll)(days_of_year + 1);
        res += __sum_days[month() - 1] + (is_leap(year()) && month() > 2 ? 1 : 0);
        res += day();
        return res;
    }

    Date next(ll next_days) const
    {
        if (next_days < 0)
            return prev(-next_days);

        ll correct_n = this->to_n() + next_days;

        int y = year() + (next_days / (days_of_year + 1));
        while (Date(y, 12, days(y, 12)).to_n() < correct_n)
            ++y;

        int m = 1;
        while (Date(y, m, days(y, m)).to_n() < correct_n)
            ++m;

        int d = 1 + (correct_n - Date(y, m, 1).to_n());

        return Date(y, m, d);
    }

    Date prev(ll prev_days) const
    {
        if (prev_days < 0)
            return next(-prev_days);

        ll correct_n = this->to_n() - prev_days;

        int y = year() - (prev_days / (days_of_year + 1));
        while (Date(y, 1, 1).to_n() > correct_n)
            --y;

        int m = 12;
        while (Date(y, m, 1).to_n() > correct_n)
            --m;

        int d = 1 + (correct_n - Date(y, m, 1).to_n());

        return Date(y, m, d);
    }

    string to_s() const
    {
        char buf[128];
        sprintf(buf, "%d/%d/%d", year(), month(), day());
        return buf;
    }
};


// maya
// 1 day = 1 kin
// uninal, tun, katun, baktun
int maya_weight[] = { 20, 18, 20, 20 };
ll maya_to_num(int baktun, int katun, int tun, int unial, int kin)
{
    int arg[] = { unial, tun, katun, baktun };
    ll res = kin, days = 1;
    for (int i = 0; i < 4; ++i)
    {
        days *= maya_weight[i];
        res += days * arg[i];
    }
    return res;
}
// return: { kin, uninal, tun, katun, baktun }
vector<int> num_to_maya(ll num)
{
    vector<int> res(5);
    for (int i = 0; i < 4; ++i)
    {
        res[i] = num % maya_weight[i];
        num /= maya_weight[i];
    }
    res[4] = num;
    return res;
}
// end maya


//// parser
ll mod;
bool error;
int pos;
string s;
void init(const string& s, int p)
{
    ::s = s;
    pos = 0;
    error = false;
    mod = p;
}
ll expr();
ll factor();
ll term();
ll add(ll a, ll b)
{
    return (a + b) % mod;
}
ll sub(ll a, ll b)
{
    return (a - b + mod) % mod;
}
ll mul(ll a, ll b)
{
    return a * b % mod;
}
ll divi(ll a, ll b)
{
    if (b == 0)
    {
        error = true;
        return 0;
    }
    return a * mod_inverse(b, mod) % mod;
}
ll expr()
{
    ll res = factor();
    while (pos < s.size() && s[pos] == '+' || s[pos] == '-')
    {
        if (s[pos] == '+')
        {
            ++pos;
            res = add(res, factor());
        }
        else
        {
            ++pos;
            res = sub(res, factor());
        }
    }
    return res;
}
ll factor()
{
    ll res = term();
    while (s[pos] == '*' || s[pos] == '/')
    {
        if (s[pos] == '*')
        {
            ++pos;
            res = mul(res, term());
        }
        else
        {
            ++pos;
            res = divi(res, term());
        }
    }
    return res;
}
ll term()
{
    if (s[pos] == '(')
    {
        ++pos;
        ll res = expr();
        assert(s[pos] == ')');
        ++pos;
        return res;
    }
    else
    {
        assert(isdigit(s[pos]));
        ll res = 0;
        while (isdigit(s[pos]))
        {
            res = res * 10 + (s[pos] - '0');
            ++pos;
        }
        return res;
    }
}
// end parser


// RollingHash
typedef unsigned long long ull;
map<ull, vector<ull> > __base_pow;
class RollingHash
{
private:
    int n;
    string _s;
    ull base;
    vector<ull> _hash; // hash[i] -> [0, i)
    vector<ull>* base_pow;
    
    void expand_base_pow()
    {
        if (base_pow->empty())
            base_pow->push_back(1);
        while ((int)base_pow->size() <= n)
            base_pow->push_back(base * base_pow->back());
    }
public:
    RollingHash(const string& s, ull base)
        : n(0), base(base), _hash(1),
          base_pow(&__base_pow[base])
    {
        add(s);
    }
    RollingHash() {}

    void operator+=(const string& s) { add(s); }
    void operator+=(char c) { add(c); }

    int size() const { return n; }
    const string& str() const { return _s; }

    void add(const string& s)
    {
        int prev_n = n;
        n += s.size();
        _s += s;
        expand_base_pow();

        _hash.resize(n + 1);
        for (int i = prev_n; i < n; ++i)
            _hash[i + 1] = base * _hash[i] + _s[i];
    }
    void add(char c)
    {
        add(string(1, c));
    }

    // delete len chars from back
    void del(int len)
    {
        assert(len <= n);
        n -= len;
        _s.erase(_s.end() - len, _s.end());
        _hash.resize(n + 1);
    }
   
    // [i, j]
    ull hash(int i, int j)
    {
        assert(0 <= i && i <= j && j < n);
        return _hash[j + 1] - base_pow->at(j - i + 1) * _hash[i];
    }
    ull hash_by_len(int i, int len)
    {
        return hash(i, i + len - 1);
    }
};

string roli_rand_s(int n)
{
    string s;
    rep(i, n)
        s += rand() % 4 + 'a';
    return s;
}
void test_rollinghash()
{
    const ull b = ten(9) + 9;
    string s = roli_rand_s(3);
    RollingHash h("", 1);
    h = RollingHash("", b);
    h += s;
    for (;;)
    {
        trace(s);
        assert(s == h.str());
        rep(j, s.size()) erep(i, j) rep(q, s.size()) erep(p, q)
        {
            bool x = s.substr(i, j - i + 1) == s.substr(p, q - p + 1);
            bool y = h.hash(i, j) == h.hash(p, q);
            assert(x == y);
        }

        string t = roli_rand_s(10);
        s += t;
        h += t;

        int del = rand() % (s.size() - 1);
        s.erase(s.end() - del, s.end());
        h.del(del);
    }
}



// precondition: [a, b]: a <= b
// { [0, 1], [2, 5], [3, 8], [8, 10], [13, 13] } -> { [0, 1], [2, 10] }
// remove point (ex. [13, 13])
template <typename T>
vector<pair<T, T> > cat_segments(vector<pair<T, T> > seg)
{
    sort(seg.begin(), seg.end());

    vector<pair<T, T> > res;
    pair<T, T> cur = seg[0];
    for (int i = 1; i < seg.size(); ++i)
    {
        if (seg[i].first <= cur.second)
            cur.second = max(cur.second, seg[i].second);
        else
        {
            if (cur.first < cur.second)
                res.push_back(cur);

            cur = seg[i];
        }
    }
    if (cur.first < cur.second)
        res.push_back(cur);

    return res;
}



#include <iterator>
template <typename T>
vector<T> to_v(const string& in)
{
    stringstream ss(in);
    istream_iterator<T> first(ss), last;
    return vector<T>(first, last);
}
template <typename T>
vector<T> to_v(const vector<string>& in)
{
    stringstream ss(accumulate(in.begin(), in.end(), string()));
    istream_iterator<T> first(ss), last;
    return vector<T>(first, last);
}



// 最近点対
bool cmp_y(const pint& a, const pint& b)
{
    return a.second < b.second;
}
int sq(int n) { return n * n; }
int closet(vector<pint>& p, int l, int r)
{
    if (r - l <= 1)
        return ten(9);

    int mid = (l + r) / 2;
    int cx = p[mid].first;
    int d = min(closet(p, l, mid), closet(p, mid, r));
    inplace_merge(p.begin() + l, p.begin() + mid, p.begin() + r, cmp_y);
    
    vector<pint> cand;
    for (int i = l; i < r; ++i)
    {
        if (sq(p[i].first - cx) >= d)
            continue;

        for (int j = (int)cand.size() - 1; j >= 0; --j)
        {
            if (sq(p[i].second - cand[j].second) >= d)
                break;
            chmin(d, sq(p[i].first - cand[j].first) + sq(p[i].second - cand[j].second));
        }
        cand.pb(p[i]);
    }
    return d;
}

// グラフ直径
pint dfs(int v, int par, const vector<vector<int>>& g)
{
    pint res(0, v);
    for (auto to : g[v])
    {
        if (to != par)
        {
            auto t = dfs(to, v, g);
            ++t.first;
            upmax(res, t);
        }
    }
    return res;
}
int diameter(int s, const vector<vector<int>>& g)
{
    pint r = dfs(s, -1, g);
    pint t = dfs(r.second, -1, g);
    return t.first;
}

// weighted
typedef ll Weight;
typedef pair<Weight, int> P;
P visit(int p, int v, const vector<vector<P>>& g)
{
    P r(0, v);
    for (auto& e : g[v])
    {
        if (e.second != p)
        {
            P t = visit(v, e.second, g);
            t.first += e.first;
            if (r.first < t.first)
                r = t;
        }
    }
    return r;
}
Weight diameter(const vector<vector<P>>& g, int s = 0)
{
    P r = visit(-1, s, g);
    P t = visit(-1, r.second, g);
    return t.first; // (r.second, t.second) is farthest pair
}


template <typename T>
class Sum2d 
{
public:
    int width, height;
    vector<vector<T>> cum;
    Sum2d(const vector<vector<T>>& a)
        : width(a[0].size()), height(a.size()), cum(height + 1, vector<T>(width + 1))
    {
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                cum[y + 1][x + 1] = cum[y][x + 1] + cum[y + 1][x] - cum[y][x] + a[y][x];
    }
    Sum2d() {}

    T get_sum(int x1, int x2, int y1, int y2) const
    {
        assert(0 <= x1 && x1 <= x2 && x2 <= width);
        assert(0 <= y1 && y1 <= y2 && y2 <= height);
        return cum[y2][x2] - cum[y1][x2] - cum[y2][x1] + cum[y1][x1];
    }

    T get_sum_by_len(int x, int y, int w, int h) const
    {
        assert(0 <= x && x + w <= width);
        assert(0 <= y && y + h <= height);
        return cum[y + h][x + w] - cum[y][x + w] - cum[y + h][x] + cum[y][x];
    }
};

int gcd(int a, int b) { return __gcd(a, b); }
class GcdSegTree
{
public:
    GcdSegTree(int n)
    {
        m = 1;
        while (m < n)
            m *= 2;
        a.resize(2 * m);
    }
 
    void update(int i, int v)
    {
        int k = i + m - 1;
        a[k] = abs(v);
        while (k > 0)
        {
            k = (k - 1) / 2;
            a[k] = gcd(a[2 * k + 1], a[2 * k + 2]);
        }
    }
 
    int query(int l, int r)
    {
        return query(l, r, 0, 0, m);
    }
 
private:
    int query(int l, int r, int k, int p, int q)
    {
        if (l <= p && q <= r)
            return a[k];
        else if (r <= p || q <= l)
            return 0;
        else
        {
            int mid = (p + q) / 2;
            return gcd(query(l, r, 2 * k + 1, p, mid), query(l, r, 2 * k + 2, mid, q));
        }
    }
 
    int m;
    vector<int> a;
};



namespace RBST
{
// #define RBST_LAZY
// #define RBST_REV
#define RBST_SUM_VAL

typedef int Value;
typedef ll SumValue;

struct Node
{
    Node(Value val)
        :
            size(1), left(nullptr), right(nullptr),
            val(val)
#ifdef RBST_REV
            , rev(false)
#endif
#ifdef RBST_SUM_VAL
            , sum_val(val)
#endif
    {
    }
    Node()
        :
            size(1), left(nullptr), right(nullptr)
#ifdef RBST_REV
            , rev(false)
#endif
    {
    }

    int size;

    Node* left;
    Node* right;

    Value val;

#ifdef RBST_SUM_VAL
    SumValue sum_val;
#endif

    // lazy
#ifdef RBST_REV
    bool rev;
#endif
};

int size(Node* t)
{
    return t ? t->size : 0;
}

#ifdef RBST_SUM_VAL
SumValue sum_val(Node* t)
{
    return t ? t->sum_val : 0;
}
#endif

Node* update(Node* t)
{
    t->size = size(t->left) + size(t->right) + 1;

#ifdef RBST_SUM_VAL
    t->sum_val = t->val + sum_val(t->left) + sum_val(t->right);
#endif

    return t;
}

void push(Node* t)
{
#ifdef RBST_LAZY
    if (!t)
        return;

#ifdef RBST_REV
    if (t->rev)
    {
        t->rev = false;

        swap(t->left, t->right);
        if (t->left)
            t->left->rev ^= true;
        if (t->right)
            t->right->rev ^= true;
    }
#endif

#endif
}

#ifdef RBST_REV
void reverse(Node* t)
{
    if (!t)
        return;

    t->rev ^= true;
}
#endif


Node* merge(Node* l, Node* r)
{
    if (!l)
        return r;
    else if (!r)
        return l;

    push(l);
    push(r);

    if (rand() % (size(l) + size(r)) < size(l))
    {
        l->right = merge(l->right, r);
        return update(l);
    }
    else
    {
        r->left = merge(l, r->left);
        return update(r);
    }
}

// [0, k), (k, n)
pair<Node*, Node*> split(Node* t, int k)
{
    assert(0 <= k && k <= size(t));

    if (!t)
        return make_pair(nullptr, nullptr);

    push(t);

    if (k <= size(t->left))
    {
        auto s = split(t->left, k);
        t->left = s.second;
        return make_pair(s.first, update(t));
    }
    else
    {
        auto s = split(t->right,  k - size(t->left) - 1);
        t->right = s.first;
        return make_pair(update(t), s.second);
    }
}

// [0, k) + inserted + [k, n)
Node* insert(Node* t, int k, Node* inserted)
{
    assert(0 <= k && k <= size(t));

    auto s = split(t, k);
    return merge(merge(s.first, inserted), s.second);
}

Node* erase(Node* t, int k)
{
    assert(0 <= k && k < size(t));

    auto a = split(t, k);
    auto b = split(a.second, 1);
    delete b.first;
    return merge(a.first, b.second);
}

Node* nth(Node* t, int n)
{
    assert(0 <= n && n < size(t));

    push(t);

    if (n < size(t->left))
        return nth(t->left, n);
    else if (n > size(t->left))
        return nth(t->right, n - size(t->left) - 1);
    else
        return t;
}

// precondition: pos is acending
vector<Node*> split(Node* t, const vector<int>& pos)
{
    vector<Node*> res(pos.size() + 1);
    for (int i = (int)pos.size() - 1; i >= 0; --i)
    {
        auto s = split(t, pos[i]);
        res[i + 1] = s.second;
        t = s.first;
    }
    res[0] = t;
    return res;
}

Node* merge(const vector<Node*>& ts)
{
    Node* root = nullptr;
    for (auto t : ts)
        root = merge(root, t);
    return root;
}

void _list_val(Node* t, vector<Value>& val)
{
    if (!t)
        return;

    _list_val(t->left, val);
    _list_val(t->right, val);

    val.push_back(t->val);
}
vector<int> list_val(Node* root)
{
    vector<Value> val;
    _list_val(root, val);
    return val;
}

int lower_bound_index(Node* t, Value val)
{
    if (!t)
        return 0;

    if (val <= t->val)
        return lower_bound_index(t->left, val);
    else
        return size(t->left) + 1 + lower_bound_index(t->right, val);
}
int upper_bound_index(Node* t, Value val)
{
    if (!t)
        return 0;

    if (val < t->val)
        return upper_bound_index(t->left, val);
    else
        return size(t->left) + 1 + upper_bound_index(t->right, val);
}

Node* ordered_insert(Node* t, Node* inserted)
{
    return insert(t, lower_bound_index(t, inserted->val), inserted);
}


// persistent
Node* persistent_merge(Node* l, Node* r)
{
    if (!l)
        return r;
    else if (!r)
        return l;

    if (rand() % (size(l) + size(r)) < size(l))
    {
        Node* copied_l = new Node(*l);
        copied_l->right = persistent_merge(copied_l->right, r);
        return update(copied_l);
    }
    else
    {
        Node* copied_r = new Node(*r);
        copied_r->left = persistent_merge(l, copied_r->left);
        return update(copied_r);
    }
}

// [0, k), (k, n)
pair<Node*, Node*> persistent_split(Node* t, int k)
{
    assert(0 <= k && k <= size(t));

    if (!t)
        return make_pair(nullptr, nullptr);

    Node* copied_t = new Node(*t);
    if (k <= size(t->left))
    {
        auto s = persistent_split(copied_t->left, k);
        copied_t->left = s.second;
        return make_pair(s.first, update(copied_t));
    }
    else
    {
        auto s = persistent_split(copied_t->right,  k - size(copied_t->left) - 1);
        copied_t->right = s.first;
        return make_pair(update(copied_t), s.second);
    }
}

// [0, k) + inserted + [k, n)
Node* persistent_insert(Node* t, int k, Node* inserted)
{
    assert(0 <= k && k <= size(t));

    auto s = persistent_split(t, k);
    return persistent_merge(persistent_merge(s.first, inserted), s.second);
}

Node* persistent_ordered_insert(Node* t, Node* inserted)
{
    return persistent_insert(t, lower_bound_index(t, inserted->val), inserted);
}


} // namespace RBST


using namespace RBST;
bool same(vector<int> a, Node* root)
{
    assert(size(root) == (int)a.size());

    rep(i, size(root))
    {
        if (nth(root, i)->val != a[i])
        {
//             printf("%4d: %d %d\n", i, a[i], nth(root, i)->val);
            return false;
        }
    }
    return true;
}
void test_rbst()
{
    vector<int> a;
    Node* root = nullptr;
    rep(i, ten(3))
    {
        // insert
        {
            int k = rand() % (1 + size(root));
            k = 0;
            int val = i;

            a.insert(a.begin() + k, val);
            root = insert(root, k, new Node(val));

            assert(same(a, root));
        }

#ifdef RBST_REV
        // reverse
//         rep(_, 100)
//         {
//             int l = rand() % size(root);
//             int r = l + rand() % (1 + size(root) - l);
//
//             reverse(a.begin() + l, a.begin() + r);
//
//             auto trees = split(root, {l, r});
//             reverse(trees[1]);
//             root = merge(trees);
//
// //             dump(a);
//             assert(same(a, root));
//         }
#endif
    }
}
void test_ordered_rbst()
{
    vector<int> a;
    Node* root = nullptr;

    rep(_, ten(5))
    {
        int val = (rand() % ten(5));
        val = _;

//         int low_i = lower_bound(all(a), val) - a.begin();
//         int up_i = upper_bound(all(a), val) - a.begin();

//         assert(lower_bound_index(root, val) == low_i);
//         assert(upper_bound_index(root, val) == up_i);


//         a.push_back(val);
//         sort(all(a));

        root = ordered_insert(root, new Node(val));
    }

    dump(nth(root, size(root) / 2)->val);
}

void test_persistent()
{
    vector<vector<int>> hist_a = {{}};
    vector<Node*> hist_root = {nullptr};

    rep(i, ten(2))
    {
        int k = rand() % (1 + size(hist_root.back()));
        int val = i;

        auto a = hist_a.back();
        a.insert(a.begin() + k, val);

        auto root = persistent_insert(hist_root.back(), k, new Node(val));

        assert(same(a, root));

        hist_a.push_back(a);
        hist_root.push_back(root);

        rep(j, hist_a.size())
            assert(same(hist_a[j], hist_root[j]));
    }
}


class HLDecomposition
{
public:
    int n;

    vector<int> depth, par;

    vector<int> cluster; // cluster[v] -> vの属するcluster_i
    vector<vector<int>> paths; // paths[cluster_i] -> path, path: 上から下順
    vector<int> index_in_path;

    HLDecomposition(){};

    HLDecomposition(const vector<vector<int>>& g, int root)
        : n(g.size()), depth(n), par(n), cluster(n, -1), index_in_path(n)
    {
        vector<int> bfs_order(n);

        // order
        {
            depth[root] = 0;
            par[root] = -1;
            bfs_order[0] = root;
            for (int p = 0, r = 1; p < r; ++p)
            {
                int cur = bfs_order[p];
                for (int next : g[cur])
                {
                    if (next != par[cur])
                    {
                        bfs_order[r++] = next;
                        par[next] = cur;
                        depth[next] = depth[cur] + 1;
                    }
                }
            }
        }

        // decomposition
        {
            vector<int> subtree_size(n, 1);
            for (int i = n - 1; i > 0; --i)
                subtree_size[par[bfs_order[i]]] += subtree_size[bfs_order[i]];

            int cluster_i = 0;
            for (int i = 0; i < n; ++i)
            {
                int u = bfs_order[i];
                if (cluster[u] == -1)
                    cluster[u] = cluster_i++;

                bool found = false;
                for (int v : g[u])
                {
                    if (par[u] != v && subtree_size[v] >= subtree_size[u] / 2)
                    {
                        cluster[v] = cluster[u];
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    for (int v : g[u])
                    {
                        if (par[u] != v)
                        {
                            cluster[v] = cluster[u];
                            break;
                        }
                    }
                }
            }
        }

        // path
        {
            int cluster_num = 0;
            vector<int> path_len(n);
            for (int i = 0; i < n; ++i)
            {
                ++path_len[cluster[i]];
                cluster_num = max(cluster_num, cluster[i]);
            }
            ++cluster_num;

            paths.resize(cluster_num);
            for (int i = 0; i < cluster_num; ++i)
                paths[i].resize(path_len[i]);

            for (int i = n - 1; i >= 0; --i)
            {
                int u = bfs_order[i];
                paths[cluster[u]][--path_len[cluster[u]]] = u;
            }

            for (vector<int>& path : paths)
            {
                for (int i = 0; i < (int)path.size(); ++i)
                    index_in_path[path[i]] = i;
            }
        }
    }
};
