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
bool in_seg(int n, int l, int r) { return l <= n && n < r; } // n in [l, r)?

typedef long long ll;
typedef pair<int, int> pint;

// y(v): v>^<  y(^): ^>v<
const int dx[] = { 0, 1, 0, -1 };
const int dy[] = { 1, 0, -1, 0 };

const double PI = acos(-1.0);
#define mp make_pair



/////////////////////////////////////////////
typedef double gtype;
const gtype EPS_FOR_LIB = 1e-9;
typedef complex<gtype> Point;

namespace std
{
	bool operator<(const Point& a, const Point& b)
	{
		return a.real() != b.real() ? a.real() < b.real() : a.imag() < b.imag();
	}
	std::istream& operator>>(std::istream& is, Point& p)
	{
		gtype a, b;
        is >> a >> b;
		p = Point(a, b);
		return is;
	}
	std::ostream& operator<<(std::ostream& os, const Point& p)
	{
		return os << "(" << p.real() << ", " << p.imag() << ")";
	}
};
struct Line
{
	Point first, second;
	Line() {}
    Line(const Point& first, const Point& second)
        : first(first), second(second)
    {
        // 距離で割るときにNaNになるのを防ぐ
        if (first == second)
            this->first.real(this->first.real() + 1e-12);
    }
};
typedef Line Seg;
ostream& operator<<(ostream& os, const Line& line)
{
    return os << "(" << line.first << ", " << line.second << ")";
}

struct Circle
{
    Point p;
    gtype r;
    Circle() {}
    Circle(const Point& p, gtype r)
        : p(p), r(r)
    {
    }
};
ostream& operator<<(ostream& os, const Circle& c)
{
    return os << "(" << c.p << ", " << c.r << ")";
}


gtype to_rad(gtype deg)
{
    return deg * PI / 180;
}
gtype to_deg(gtype rad)
{
    return rad * 180 / PI;
}


typedef vector<Point> G;
typedef G Convex;
Seg side_G(const G& g, int i)
{
    return Seg(g[i], g[(i + 1) % g.size()]);
}


gtype dot(const Point& a, const Point& b)
{
	return a.real() * b.real() + a.imag() * b.imag();
}
gtype cross(const Point& a, const Point& b)
{
	return a.real() * b.imag() - a.imag() * b.real();
}

enum res_ccw
{
	counter_clockwise = +1,
	clockwise = -1,
	on = 0,
};
res_ccw ccw(const Point& a, const Point& b, const Point& c)
{
    const gtype feps = 1e-10;
	Point p = b - a, q = c - a;
	if (cross(p, q) > feps) return counter_clockwise;
	else if (cross(p, q) < -feps) return clockwise;
	return on;
}


//// 交差判定

// 直交判定
// AOJ0058
bool is_orthogonal(const Line& line1, const Line& line2)
{
	return abs(dot(line1.second - line1.first, line2.second - line2.first)) < 1e-9;
}
// AOJ0021, AOJ0187
bool is_parallel(const Line& line1, const Line& line2)
{
	return abs(cross(line1.second - line1.first, line2.second - line2.first)) < 1e-9;
}

bool intersect_LP(const Line& line, const Point& p)
{
	return abs(cross(line.second - line.first, p - line.first)) < 1e-20;
}
bool intersect_SP(const Line& seg, const Point& p)
{
	return abs(seg.first - p) + abs(p - seg.second) < abs(seg.first - seg.second) + EPS_FOR_LIB;
}


bool intersect_LL(const Line& line1, const Line& line2)
{
	return !is_parallel(line1, line2);
}
// AOJ1183
bool intersect_LS(const Line& line, const Line& seg)
{
	return cross(line.second - line.first, seg.first - line.first)
		* cross(line.second - line.first, seg.second - line.first) < EPS_FOR_LIB;
}
bool intersect_SL(const Line& seg, const Line& line)
{
    return intersect_LS(line, seg);
}
// AOJ0187, AOJ1183
bool intersect_SS(const Line& seg1, const Line& seg2)
{
    const gtype feps = 1e-9;
	return (cross(seg1.second - seg1.first, seg2.first - seg1.first) * cross(seg1.second - seg1.first, seg2.second - seg1.first) < -feps
            && cross(seg2.second - seg2.first, seg1.first - seg2.first) * cross(seg2.second - seg2.first, seg1.second - seg2.first) < -feps)
		|| intersect_SP(seg1, seg2.first)
		|| intersect_SP(seg1, seg2.second)
		|| intersect_SP(seg2, seg1.first)
		|| intersect_SP(seg2, seg1.second);
}


//// 距離
gtype dist_LP(const Line& line, const Point& p)
{
	return abs(cross(line.first - line.second, p - line.second) / abs(line.first - line.second));
}
gtype dist_PL(const Point& p, const Line& line)
{
    return dist_LP(line, p);
}
gtype dist_LL(const Line& line1, const Line& line2)
{
	return is_parallel(line1, line2) ? dist_LP(line1, line2.first) : 0;
}
gtype dist_LS(const Line& line, const Line& seg)
{
	if (intersect_LS(line, seg))
		return 0;
	else
		return min(dist_LP(line, seg.first), dist_LP(line, seg.second));
}
gtype dist_SL(const Line& seg, const Line& line)
{
    return dist_LS(line, seg); // 写すとき注意！！！
}
gtype dist_SP(const Line& seg, const Point& p)
{
	if (dot(seg.second - seg.first, p - seg.first) < 0)
		return abs(seg.first - p);
	else if (dot(seg.first - seg.second, p - seg.second) < 0)
		return abs(seg.second - p);
	else
		return dist_LP(seg, p);
}
gtype dist_PS(const Point& p, const Line& seg)
{
    return dist_SP(seg, p);
}
// AOJ1157
gtype dist_SS(const Line& seg1, const Line& seg2)
{
	if (intersect_SS(seg1, seg2))
		return 0;
	else
		return min(min(dist_SP(seg1, seg2.first), dist_SP(seg1, seg2.second))
			, min(dist_SP(seg2, seg1.first), dist_SP(seg2, seg1.second)));
}

Point ip_SS(const Line& seg1, const Line& seg2)
{
	if (!intersect_SS(seg1, seg2))
	{
		cerr << "ip_SS: 前提条件満たしてない" << endl;
		exit(1);
	}

	gtype a = abs(cross(seg1.second - seg1.first, seg2.first - seg1.first));
	gtype b = abs(cross(seg1.second - seg1.first, seg2.second - seg1.first));
    if (a < 1e-9 && b < 1e-9)
    {
        cerr << "same line" << endl;
        exit(1);
    }

	gtype t = a / (a + b);
	return seg2.first + t * (seg2.second - seg2.first);
}
Point ip_LL(const Line& line1, const Line& line2)
{
	if (is_parallel(line1, line2))
	{
		cerr << "ip_LL: 前提条件満たしてない" << endl;
		exit(1);
	}

	Point a = line1.second - line1.first, b = line2.second - line2.first;
	gtype p = cross(b, line2.first - line1.first);
	gtype q = cross(b, a);
	return line1.first + p / q * a;
}


// 回転
Point rotate(const Point& p, gtype angle)
{
	gtype c = cos(angle), s = sin(angle);
	return Point(p.real() * c - p.imag() * s, p.real() * s + p.imag() * c);
}
Point rotate(const Point& p, gtype angle, const Point& base)
{
	Point t = p - base;
	return rotate(t, angle) + base;
}

// 点から直線に垂線を下ろした点
// AOJ0081(by reflection), AOJ1183(by ip_CL)
Point projection(const Line& line, const Point& p)
{
	Point a = line.first - line.second;
	gtype t = dot(p - line.first, a) / norm(a);
	return line.first + t * a;
}

// 線対称な点
// AOJ0081
Point reflection(const Line& line, const Point& p)
{
	return p + ((gtype)2) * (projection(line, p) - p);
}


// 凸包
// AOJ0068
bool allow_line(res_ccw r) { return r < 0; }
bool strict(res_ccw r) { return r <= 0; }
Convex convex_hull(vector<Point> ps, bool f(res_ccw) = strict)
{
    sort(ps.begin(), ps.end());
    ps.erase(unique(ps.begin(), ps.end()), ps.end());

    int n = ps.size(), k = 0;
    G res;
    res.resize(2 * n);
    for (int i = 0; i < n; ++i)
    {
        while (k >= 2 && f(ccw(res[k - 2], res[k - 1], ps[i])))
            --k;
        res[k++] = ps[i];
    }
    for (int i = n - 2, t = k + 1; i >= 0; --i)
    {
        while (k >= t && f(ccw(res[k - 2], res[k - 1], ps[i])))
            --k;
        res[k++] = ps[i];
    }
    res.resize(k - 1);
    return res;
}

// 凸包判定, 反時計回り
// AOJ0035
bool is_convex(const G& g)
{
    for (int i = 0; i < (int)g.size(); ++i)
    {
        if (ccw(g[(i - 1 + g.size()) % g.size()], g[i], g[(i + 1) % g.size()]) < 0)
            return false;
    }
    return true;
}

// AOJ0079, AOJ0187
gtype calc_area(const G& g)
{
	gtype s = 0;
	for (int i = 0; i < (int)g.size(); ++i)
		s += cross(g[i], g[(i + 1) % g.size()]);
	return abs(s / 2);
}

// AOJ0012, AOJ0143
enum res_contain { OUT, ON, IN };
res_contain contain_GP(const G& g, const Point& p)
{
	bool in = false;
	for (int i = 0; i < (int)g.size(); ++i)
	{
		Point a = g[i] - p, b = g[(i + 1) % g.size()] - p;
		if (a.imag() > b.imag())
			swap(a, b);
		if ((a.imag() <= 0 && 0 < b.imag()) && cross(a, b) < 0)
			in = !in;
		if (intersect_SP(Line(g[i], g[(i + 1) % g.size()]), p))
			return ON;
	}
	return in ? IN : OUT;
}

// a contains b?
// AOJ0214
bool contain_GG(const G& a, const G& b)
{
    rep(i, b.size())
        if (contain_GP(a, b[i]) == OUT)
            return false;
    return true;
}
// AOJ0214
bool intersect_GG(const G& a, const G& b)
{
    rep(i, a.size()) rep(j, b.size())
        if (intersect_SS(Line(a[i], a[(i + 1) % a.size()]), Line(b[j], b[(j + 1) % b.size()])))
            return true;
    return contain_GG(a, b) || contain_GG(b, a);
}

// AOJ1157(gは長方形)
bool intersect_GS(const G& g, const Seg& s)
{
    if (contain_GP(g, s.first) || contain_GP(g, s.second))
        return true;
    rep(i, g.size())
        if (intersect_SS(Seg(g[i], g[(i + 1) % g.size()]), s))
            return true;
    return false;
}
bool intersect_SG(const Seg& s, const G& g)
{
    return intersect_GS(g, s);
}

// 円

// AOJ0023(feps = 0)
enum res_pos_CC
{
    not_intersect,
    intersect,

    tangent,

    a_in_b,
    b_in_a,
};
res_pos_CC pos_CC(const Circle& a, const Circle& b)
{
    const gtype feps = 1e-9;
    gtype d = abs(a.p - b.p);
    if (d + feps > a.r + b.r)
        return not_intersect;
    else
    {
        if (d + feps < a.r - b.r)
            return b_in_a;
        else if (d + feps < b.r - a.r)
            return a_in_b;
        else
            return intersect;
    }
}

bool intersect_GC(const G& g, const Circle& c)
{
    for (int i = 0; i < (int)g.size(); ++i)
        if (dist_SP(Line(g[i], g[(i + 1) % g.size()]), c.p) < c.r + 1e-9)
            return true;
    return contain_GP(g, c.p) != OUT;
}

// AOJ0129, AOJ1132
res_contain contain_CP(const Circle& c, const Point& p)
{
    const gtype feps = 1e-9;
    gtype d = abs(c.p - p);
    if (d > c.r + feps)
        return OUT;
    else if (d < c.r + feps)
        return IN;
    else
        return ON;
}

// 円周と線分が交わるか
// AOJ0129
bool intersect_CS(const Circle& c, const Seg& seg)
{
    return dist_SP(seg, c.p) < c.r + 1e-9;
}
bool intersect_SC(const Seg& seg, const Circle& c)
{
    return intersect_CS(c, seg);
}

// AOJ2201
gtype dist_CL(const Circle& c, const Line& line)
{
    return max<gtype>(0, dist_LP(line, c.p) - c.r);
}
gtype dist_LC(const Line& line, const Circle& c)
{
    return dist_CL(c, line);
}

// AOJ1183(必ず交点が2点あるテストケース)
vector<Point> ip_CC(const Circle& a, const Circle& b)
{
    const gtype feps = 1e-9;

    if (pos_CC(a, b) != intersect)
        return vector<Point>();
    // if (abs(a.p - b.p) < a.r + b.r - feps)
    //     return vector<Line>();

    Point ab = b.p - a.p;
    gtype t = (norm(ab) + a.r*a.r - b.r*b.r) / (2 * abs(ab));
    Point u = ab / abs(ab);
    Point q = a.p + t * u;

    gtype h = sqrt(max<gtype>(0, a.r*a.r - t*t));
    Point v = Point(0, h) * u;

    vector<Point> res;
    res.push_back(q + v);
    if (h > feps)
        res.push_back(q - v); // 2点
    return res;
}

// AOJ1183
vector<Point> ip_CL(const Circle& c, const Line& line)
{
    const gtype feps = 1e-9;

    Point p = projection(line, c.p);
    Point cp = p - c.p;
    gtype d = abs(cp);
    if (d > c.r + feps)
        return vector<Point>();

    gtype t = sqrt(max<gtype>(0, c.r*c.r - d*d));
    Point u = line.second - line.first;
    Point v = u / abs(u) * t;

    vector<Point> res;
    res.push_back(p + v);
    if (t > feps)
        res.push_back(p - v); // 2点
    return res;
}

// AOJ1183(必ず交点が2点あるテストケース)
vector<Point> ip_CS(const Circle& c, const Seg& seg)
{
    vector<Point> ip = ip_CL(c, seg);
    vector<Point> res;
    rep(i, ip.size())
        if (intersect_SP(seg, ip[i]))
            res.push_back(ip[i]);
    return res;
}

// 点pを通るcの接線
// AOJ 2201(円が重ならないケースしかない)
vector<Line> tangent_CP(const Circle& c, const Point& p)
{
    const gtype feps = 1e-8;

    Point vec = c.p - p;
    gtype d = abs(vec);
    if (d < c.r)
        return vector<Line>();

    gtype t = sqrt(max<gtype>(0, d*d - c.r*c.r));
    Point rota = Point(t / d, c.r / d);
    rota *= 1; // 線分の長さが0にならないように // 何これ？

    vector<Line> res;
    res.push_back(Line(p, p + vec * rota));
    if (d > feps)
        res.push_back(Line(p, p + vec * conj(rota)));
    return res;
}

// 共通外接線
// AOJ 2201(円が重ならないケースしかない)
vector<Line> tangent_ext_CC(const Circle& a, const Circle& b)
{
    if (abs(a.p - b.p) < abs(a.r - b.r))
        return vector<Line>(); // 内包
    
    if (abs(a.r - b.r) > 1e-8)
    {
        Point ip = (-a.p * b.r + b.p * a.r) / (a.r - b.r);
        return tangent_CP(a, ip);
    }
    else
    {
        vector<Line> res;
        Point v = b.p - a.p;
        v /= abs(v);
        v *= Point(0, a.r);
        res.push_back(Line(a.p + v, b.p + v));
        res.push_back(Line(a.p - v, b.p - v));
        return res;
    }
}
// 共通内接線
// AOJ 2201(円が重ならないケースしかない)
vector<Line> tangent_in_CC(const Circle& a, const Circle& b)
{
    const gtype feps = 1e-8;
    if (abs(a.p - b.p) < a.r + b.r - feps)
        return vector<Line>();
    else
    {
        Point ip = (a.p * b.r + b.p * a.r) / (a.r + b.r);
        return tangent_CP(a, ip);
    }
}
// 共通接線
// AOJ 2201(円が重ならないケースしかない)
vector<Line> tangent_CC(const Circle& a, const Circle& b)
{
    vector<Line> res;
    vector<Line> ext = tangent_ext_CC(a, b);
    vector<Line> in = tangent_in_CC(a, b);
    rep(i, ext.size())
        res.push_back(ext[i]);
    rep(i, in.size())
        res.push_back(in[i]);
    return res;
}

// 半径rで点a, bを通る円
// AOJ1132
vector<Circle> circle_by_PP(const Point& a, const Point& b, gtype r)
{
    Point v = 0.5 * (b - a);
    gtype t = sqrt(max<gtype>(0, r*r - norm(v))); 
    Point u = t / abs(v) * Point(0, 1) * v;

   vector<Circle> res;
   res.push_back(Circle(a + v + u, r));
   res.push_back(Circle(a + v - u, r));
   return res;
}

// gは凸多角形
// lineより左側の凸多角形を返す
// AOJ1213(正方形に2回カットなので緩い), AOJ1283(厳し目)
Convex convex_cut(const Convex& g, const Line& line)
{ 
    Convex left;
    rep(i, g.size())
    {
        const Point& a = g[i], b = g[(i + 1) % g.size()];
        if (abs(a - b) < 1e-9)
            continue;
        if (ccw(line.first, line.second, a) != clockwise)
            left.push_back(a);
        if (ccw(line.first, line.second, a) * ccw(line.first, line.second, b) < 0)
            left.push_back(ip_LL(Line(a, b), line));
    }
    return left;
}
Convex convex_cut(const Convex& g, const Point& a, const Point& b)
{
    return convex_cut(g, Line(a, b));
}

// abの垂直二等分線、Lineの向きはabを反時計に90度
// AOJ1213
Line perp_bisector(const Point& a, const Point& b)
{
    Point v = 0.5 * (b - a);
    return Line(a + v, a + v + Point(0, 1) * v);
}




///// いらないかも

// 長方形
struct Rect
{
	Point low, high;
	Rect(Point low, Point high)
		: low(low), high(high) { }
	Rect() { }

	gtype x1() const { return low.real(); }
	gtype x2() const { return high.real(); }
	gtype y1() const { return low.imag(); }
	gtype y2() const { return high.imag(); }

    Point top_left() const { return Point(x1(), y2()); }
    Point bottom_left() const { return Point(x1(), y1()); }
    Point bottom_right() const { return Point(x2(), y1()); }
    Point top_right() const { return Point(x2(), y2()); }

    G to_g() const
    {
        G res;
        res.push_back(top_left());
        res.push_back(bottom_left());
        res.push_back(bottom_right());
        res.push_back(top_right());
        return res;
    }
};
// 境界交差はfalse
bool intersect_rect_area(const Rect& a, const Rect& b)
{
	bool x = a.low.real() < b.high.real() && a.high.real() > b.low.real();
	bool y = a.low.imag() < b.high.imag() && a.high.imag() > b.low.imag();
	return x && y;
}
// allow segment
bool intersect_rect(const Rect& a, const Rect& b)
{
	bool x = !(a.low.real() > b.high.real()) && !(a.high.real() < b.low.real());
	bool y = !(a.low.imag() > b.high.imag()) && !(a.high.imag() < b.low.imag());
	return x && y;
}
vector<Point> corner(const Rect& r)
{
	gtype x[] = { r.low.real(), r.high.real() };
	gtype y[] = { r.low.imag(), r.high.imag() };
	vector<Point> res;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			res.push_back(Point(x[i], y[j]));
	return res;
}

/////////////////// 


int main()
{
    fast_io();

    int T;
    cin >> T;
    while (T--)
    {
        Seg s;
        cin >> s.first >> s.second;
        Point rota(1, -arg(s.second - s.first));
        s.first *= rota, s.second *= rota;
        
        
        vector<pair<Point, bool> > c;
        int n;
        cin >> n;
        rep(i, n)
        {
            Seg t;
            int o, l;
            cin >> t.first >> t.second >> o >> l;
            t.first *= rota, t.second *= rota;
            if (intersect_SS(s, t))
                c.pb(mp(ip_SS(s, t), (o ^ l)));
        }
        sort(all(c));

        int res = 0;
        if (!c.empty())
        {
            bool f = c[0].second;
            for (int i = 1; i < c.size(); ++i)
            {
                if (f != c[i].second)
                {
                    ++res;
                    f ^= 1;
                }
            }
        }
        cout << res << endl;
    }
}
