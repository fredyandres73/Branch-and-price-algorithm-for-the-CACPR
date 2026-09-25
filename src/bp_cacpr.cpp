// bp_cacpr.cpp - Capacitated Angular Set Covering Problem with Rotation (C-ACPR)
// Compact model and branch-and-price of
//   "A branch-and-price algorithm for the capacitated angular set covering problem
//    with rotation", Barriga-Gallegos, Montecinos, Luer-Villagra, Gutierrez-Jarpa.
//
// bp_cacpr <instance> <time_limit_s> <threads> [Q=3] [rho=15] [norot=0] [beta=0.5]
//          [model=1] [reserved=0] [tree_log_nodes=2000] [lagrangian=1] [cardinality=1]
//          [hierarchical=1] [warm_start=1] [integer_rounding=1] [primal=1]
//
// Q >= |T|*360/rho leaves the capacity immaterial; norot=1 sets V_t={0};
// model: 0 branch-and-price, 1 both, 2 compact model only; the last six are the
// ablation switches. Reads Instances/, writes Results/. See README.md.
#include <ilcplex/ilocplex.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <array>
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <queue>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace std::chrono;
ILOSTLBEGIN

typedef IloArray<IloBoolVarArray> IloBoolVar2D;
typedef IloArray<IloBoolVar2D>    IloBoolVar3D;
typedef IloArray<IloBoolVar3D>    IloBoolVar4D;
typedef IloArray<IloBoolVar4D>    IloBoolVar5D;

// ---- tolerances and parameters ----
static const double EPS_RC = 0.01;             // reduced-cost tolerance
static const double INT_TOL = 1e-4;            // integrality tolerance
static const double PRUNE_TOL = 1e-4;          // prune if lb >= ub - PRUNE_TOL
static const double ART_COST = 1e7;            // artificial-variable penalty
static const char* CODE_VERSION = "v3.6_final_260925";

static const double SP_NOBND = -1e100;         // subproblem without a usable bound
static inline bool spSinCota(double cb) { return cb <= -1e99; }
static const int    MAXIT_ROOT = 500;          // column generation iterations, root
static const int    MAXIT_NODE = 20;           // column generation iterations, node
static const double SP_TCAP = 10.0;            // time cap per exact subproblem
static const int    IMP_EVERY = 50;            // restricted-master heuristic cadence
static const double IMP_BUDGET = 30.0;         // budget of each periodic call
static const int    CERT_EVERY = 10;           // exact certification cadence
static const double CERT_NEAR = 0.98;          // certify when the RMP is within 2% of the ub
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- instance data ----
int P, U, C, S;                                // demand points, locations, configurations, types
int QCAP = 3;                                  // servers per located facility
int RHO = 15;                                  // rotation magnitude
bool NOROT = false;                            // V_t = {0}
double BRANCH_TARGET = 0.5;                    // beta
int  MODEL_MODE = 1;                           // 0 branch-and-price, 1 both, 2 compact model
bool HIER_OFF = false;                         // ablation: hierarchical branching
bool WARM_OFF = false;                         // ablation: warm start
bool INTR_OFF = false;                         // ablation: integer rounding of the bounds
bool PRIM_OFF = false;                         // ablation: primal package
static inline double intCeil(double lb) {      // integer costs, so ceil keeps a bound valid
    if (INTR_OFF || lb <= -1e99 || lb >= 1e99) return lb;
    return ceil(lb - 1e-6);
}
bool CARD_OFF = false;                         // ablation: cardinality cut
int    CARD_K = 0;                             // cut right-hand side, 0 = no cut
double CARD_LP = -1.0;
bool LAGR_OFF = false;                         // ablation: Lagrangian bounds
vector<int> conf_, pos_, covd_, rot_;          // angles, positions, distances, rotations
int facilityCost = 0;
vector<vector<int>> srvCost;
vector<array<double, 2>> pcord, ucord;
vector<vector<double>> dist_, ang_;            // from each location to each demand point
vector<vector<double>> srvDist;                // covering distance per type and configuration

int              SRV = 0;                      // servers (t,p,s,v) per location
vector<int>      tBase;
vector<char>     A2F;                          // coverage matrix alpha, flattened

static inline int srvLocal(int t, int p, int s, int v) {
    return tBase[t] + (p * S + s) * rot_[t] + v;
}
static inline size_t a2base(int j, int t, int p, int s, int v) {
    return ((size_t)j * SRV + (size_t)srvLocal(t, p, s, v)) * (size_t)P;
}
static inline char A2at(int i, int j, int t, int p, int s, int v) {
    return A2F[a2base(j, t, p, s, v) + (size_t)i];
}

static double elap(steady_clock::time_point t0) {
    return duration<double>(steady_clock::now() - t0).count();
}

// ---- clock ----
static string fmtClock(system_clock::time_point tp, bool withDate = false) {
    time_t tt = system_clock::to_time_t(tp);
    tm lt;
#ifdef _WIN32
    localtime_s(&lt, &tt);
#else
    localtime_r(&tt, &lt);
#endif
    char buf[32];
    strftime(buf, sizeof(buf), withDate ? "%Y-%m-%d %H:%M:%S" : "%H:%M:%S", &lt);
    return string(buf);
}
static string nowStr(bool withDate = false) { return fmtClock(system_clock::now(), withDate); }
static string clockPlus(double secs) {
    auto tp = system_clock::now() + duration_cast<system_clock::duration>(duration<double>(secs));
    return fmtClock(tp);
}

// ---- console output ----
static void rule(char c = '-') { cout << string(72, c) << "\n"; }
static void banner(const string& title) {
    rule('=');
    cout << "  " << title << "\n";
    rule('=');
}

bool readInstance(const string& inst) {
    stringstream ph; ph << "Instances/" << inst << ".txt";
    ifstream data(ph.str());
    if (!data.good()) { cout << "No se pudo abrir " << ph.str() << endl; return false; }
    data >> P >> U >> C >> S;
    conf_.resize(C); for (int t = 0;t < C;++t) data >> conf_[t];
    pos_.resize(C);  for (int t = 0;t < C;++t) data >> pos_[t];
    covd_.resize(S); for (int s = 0;s < S;++s) data >> covd_[s];
    data >> facilityCost;
    srvCost.assign(S, vector<int>(C));
    for (int s = 0;s < S;++s) for (int t = 0;t < C;++t) data >> srvCost[s][t];
    pcord.resize(P); for (int i = 0;i < P;++i) data >> pcord[i][0] >> pcord[i][1];
    ucord.resize(U); for (int j = 0;j < U;++j) data >> ucord[j][0] >> ucord[j][1];
    return true;
}

void processData() {
    srvDist.assign(S, vector<double>(C));
    for (int s = 0;s < S;++s) {
        double area = M_PI * pow((double)covd_[s], 2) / 4.0;
        for (int t = 0;t < C;++t) srvDist[s][t] = sqrt(area) * sqrt(pos_[t] / M_PI);
    }
    dist_.assign(P, vector<double>(U));
    ang_.assign(P, vector<double>(U));
    for (int i = 0;i < P;++i) for (int j = 0;j < U;++j) {
        double dx = pcord[i][0] - ucord[j][0], dy = pcord[i][1] - ucord[j][1];
        dist_[i][j] = hypot(dx, dy);
        double a = atan2(dy, dx); if (a < 0) a += 2 * M_PI;
        ang_[i][j] = a * 180.0 / M_PI;
    }
    rot_.resize(C); for (int t = 0;t < C;++t) rot_[t] = NOROT ? 1 : conf_[t] / RHO;

    tBase.assign(C, 0); SRV = 0;
    for (int t = 0;t < C;++t) { tBase[t] = SRV; SRV += pos_[t] * S * rot_[t]; }
    A2F.assign((size_t)U * (size_t)SRV * (size_t)P, 0);

    for (int i = 0;i < P;++i) for (int j = 0;j < U;++j) {
        double d = dist_[i][j];
        for (int t = 0;t < C;++t)
            for (int p = 0;p < pos_[t];++p)
                for (int s = 0;s < S;++s)
                    for (int v = 0; v < rot_[t]; ++v) {
                        double s0 = (double)conf_[t] * p + (double)RHO * v;
                        double m = fmod(ang_[i][j] - s0, 360.0); if (m < 0) m += 360.0;
                        A2F[a2base(j, t, p, s, v) + (size_t)i] =
                            (d > 1e-9 && d <= srvDist[s][t] && m < (double)conf_[t]) ? 1 : 0;
                    }
    }
}

// ---- compact model ----
struct ModelRes { double Z = 1e100, gap = 0.0, time = 0.0; bool feasible = false; };

ModelRes solveCompactModel(int threads, double timelimit) {
    ModelRes MR; auto t0 = steady_clock::now();
    IloEnv env;
    try {
        IloModel m(env);
        IloBoolVarArray w(env, U);
        IloBoolVar5D Y(env, U);
        for (int j = 0;j < U;++j) {
            Y[j] = IloBoolVar4D(env, C);
            for (int t = 0;t < C;++t) {
                Y[j][t] = IloBoolVar3D(env, pos_[t]);
                for (int p = 0;p < pos_[t];++p) {
                    Y[j][t][p] = IloBoolVar2D(env, S);
                    for (int s = 0;s < S;++s) Y[j][t][p][s] = IloBoolVarArray(env, rot_[t]);
                }
            }
        }
        IloExpr ob(env);
        for (int j = 0;j < U;++j) {
            ob += facilityCost * w[j];
            for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p)
                for (int s = 0;s < S;++s) for (int v = 0; v < rot_[t]; ++v)
                    ob += srvCost[s][t] * Y[j][t][p][s][v];
        }
        m.add(IloMinimize(env, ob)); ob.end();
        for (int j = 0;j < U;++j) {
            IloExpr cap(env);
            for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p) {
                IloExpr pk(env);
                for (int s = 0;s < S;++s) for (int v = 0; v < rot_[t]; ++v) {
                    m.add(Y[j][t][p][s][v] <= w[j]);
                    cap += Y[j][t][p][s][v];
                }
                for (int v = 0; v < rot_[t]; ++v) {
                    IloExpr pk1(env);
                    for (int s = 0;s < S;++s) pk1 += Y[j][t][p][s][v];
                    m.add(pk1 <= 1);
                    pk1.end();
                }
                pk.end();
            }
            m.add(cap <= QCAP * w[j]);
            cap.end();
        }
        for (int i = 0;i < P;++i) {
            IloExpr cov(env);
            for (int j = 0;j < U;++j) for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p)
                for (int s = 0;s < S;++s) for (int v = 0; v < rot_[t]; ++v)
                    if (A2at(i, j, t, p, s, v)) cov += Y[j][t][p][s][v];
            m.add(cov >= 1); cov.end();
        }
        IloCplex cx(m);
        cx.setParam(IloCplex::Param::Threads, threads);
        cx.setParam(IloCplex::Param::TimeLimit, timelimit);
        cx.setOut(env.getNullStream()); cx.setWarning(env.getNullStream());
        bool ok = cx.solve();
        if (ok || cx.getStatus() == IloAlgorithm::Feasible) {
            MR.feasible = true; MR.Z = cx.getObjValue(); MR.gap = cx.getMIPRelativeGap();
        }
    }
    catch (IloException& e) { cout << "Compacto: " << e.getMessage() << endl; }
    env.end();
    MR.time = elap(t0);
    cout << fixed << setprecision(2);
    cout << "[Modelo compacto]  Z = " << setw(12) << MR.Z
        << "   gap = " << setw(6) << MR.gap * 100 << "%"
        << "   t = " << setw(7) << MR.time << " s"
        << (MR.feasible ? "" : "   (SIN SOLUCION)") << "\n";
    return MR;
}

// ---- branch-and-price structures ----
typedef tuple<int, int, int, int, int> Srv;

struct Column { vector<Srv> srv; double cost; vector<char> delta; };
vector<vector<Column>> POOL;

struct BDec { int lvl, j, t, p, s, v, fix; };
map<int, BDec> decOf;
map<int, int>  dad;

struct Node { int id; double lb; };
struct CmpN { bool operator()(const Node& a, const Node& b)const { return a.lb > b.lb; } };

long   CG_totalIt = 0, colsHeur = 0, colsExact = 0;
int    BP_nodesSolved = 0, nextNodeId = 1;
double BP_UB = 1e100, BP_LB = -1e100;
vector<pair<int, int>> BEST_SOL;
long   DIVE_calls = 0, DIVE_hits = 0;
long   IMP_ROUND = 0;
int    OMP_THREADS = 1;
int    rootIt = 0; long rootCols = 0; double rootLB = -1e100, rootTime = 0;
double rootIMP_obj = 0, rootIMP_gap = 0, rootIMP_time = 0;
double IMP_obj = 0, IMP_gap = 0, IMP_time = 0;
bool   outOfTime = false;
bool   branchIncomplete = false;
double lbCap = 1e100;
bool   BP_proven = false;

// ---- search tree log ----
static ofstream TREE;
static int LOG_DETAIL_MAX = 2000;

static string yName(int j, int t, int p, int s, int v) {
    ostringstream os;
    os << "y(j=" << j << ",t=" << t << ",p=" << p << ",s=" << s << ",v=" << v << ")";
    return os.str();
}
static string fmtB(double x) {
    if (x >= 1e99)  return "+inf";
    if (x <= -1e99) return "-inf";
    ostringstream os; os << fixed << setprecision(2) << x;
    return os.str();
}
static string colSrvStr(const Column& c) {
    ostringstream os; os << "{";
    for (size_t k = 0;k < c.srv.size();++k) {
        const auto& e = c.srv[k];
        os << (k ? " " : "") << "(" << get<1>(e) << "," << get<2>(e)
            << "," << get<3>(e) << "," << get<4>(e) << ")";
    }
    os << "}";
    return os.str();
}

static bool colMatches(const Column& c, const BDec& d) {
    if (d.lvl == 0) return true;
    for (const auto& e : c.srv) {
        if (get<1>(e) != d.t || get<2>(e) != d.p) continue;
        if (d.lvl == 1) return true;
        if (get<3>(e) == d.s && get<4>(e) == d.v) return true;
    }
    return false;
}
static string decName(const BDec& d) {
    ostringstream os;
    if (d.lvl == 0)      os << "w(j=" << d.j << ")";
    else if (d.lvl == 1) os << "pos(j=" << d.j << ",t=" << d.t << ",p=" << d.p << ")";
    else                 os << "y(j=" << d.j << ",t=" << d.t << ",p=" << d.p << ",s=" << d.s << ",v=" << d.v << ")";
    return os.str();
}
static string decConsInc(const BDec& d) {
    ostringstream os;
    if (d.lvl == 0)
        os << "maestro += { sum f[" << d.j << "][a] : TODAS las columnas } >= 1 (ubicacion abierta; dual omega al pricing de j=" << d.j << ")";
    else if (d.lvl == 1)
        os << "maestro += { sum f[" << d.j << "][a] : a usa la posicion (t=" << d.t << ",p=" << d.p
        << ") } >= 1 ;  SP_" << d.j << ": sum_{s,v} y[" << d.t << "][" << d.p << "][s][v] >= 1  (dual omega)";
    else
        os << "maestro += { sum f[" << d.j << "][a] : a contiene " << decName(d) << " } >= 1 ;  SP_" << d.j << " fija y=1  (dual omega)";
    return os.str();
}
static string decConsExc(const BDec& d) {
    ostringstream os;
    if (d.lvl == 0)
        os << "ubicacion j=" << d.j << " CERRADA: todas sus columnas con f=0 y su subproblema apagado";
    else if (d.lvl == 1)
        os << "columnas de j=" << d.j << " que usan la posicion (t=" << d.t << ",p=" << d.p
        << ") quedan con f=0 ;  SP_" << d.j << ": y[" << d.t << "][" << d.p << "][s][v]=0 para todo s,v";
    else
        os << "columnas de j=" << d.j << " con " << decName(d) << " quedan con f=0 ;  SP_" << d.j << " fija y=0";
    return os.str();
}
static vector<BDec> pathDecs(int node) {
    vector<BDec> r;
    while (node != 0) { r.push_back(decOf[node]); node = dad[node]; }
    return r;
}
static bool addColumnIfNew(int j, vector<Srv> sel, bool fromHeur) {
    sort(sel.begin(), sel.end());
    for (const auto& c : POOL[j]) if (c.srv == sel) return false;
    Column col; col.srv = sel; col.cost = facilityCost; col.delta.assign(P, 0);
    for (const auto& e : sel) col.cost += srvCost[get<3>(e)][get<1>(e)];
    for (const auto& e : sel) {
        size_t b = a2base(j, get<1>(e), get<2>(e), get<3>(e), get<4>(e));
        for (int i = 0;i < P;++i) if (A2F[b + (size_t)i]) col.delta[i] = 1;
    }
    POOL[j].push_back(move(col));
    if (fromHeur) {
#ifdef _OPENMP
#pragma omp atomic
#endif
        ++colsHeur;
    }
    else {
#ifdef _OPENMP
#pragma omp atomic
#endif
        ++colsExact;
    }
    return true;
}

// ---- restricted master problem ----
struct RMPOut {
    bool ok = false; double obj = 1e100, artSum = 0.0;
    vector<double> th, la, om;
    double mu = 0.0;
    bool integer_ = false;
    map<pair<int, int>, double> fval;
};
RMPOut solveRMPRebuild(const vector<BDec>& decs, int threads, double tl) {
    RMPOut R; if (tl <= 1) return R;
    IloEnv env;
    try {
        IloModel m(env);
        vector<IloNumVarArray> f(U);
        for (int j = 0;j < U;++j) f[j] = IloNumVarArray(env, (IloInt)POOL[j].size(), 0.0, IloInfinity);
        for (const auto& d : decs) if (d.fix == 0)
            for (size_t a = 0;a < POOL[d.j].size();++a)
                if (colMatches(POOL[d.j][a], d)) f[d.j][(IloInt)a].setUB(0.0);
        IloNumVarArray art(env, P, 0.0, IloInfinity);
        IloNumVar artC(env, 0.0, IloInfinity);
        IloExpr ob(env);
        for (int j = 0;j < U;++j) for (size_t a = 0;a < POOL[j].size();++a) ob += POOL[j][a].cost * f[j][(IloInt)a];
        for (int i = 0;i < P;++i) ob += ART_COST * art[i];
        if (CARD_K >= 1) ob += ART_COST * artC;
        m.add(IloMinimize(env, ob)); ob.end();
        IloRange Rcard;
        if (CARD_K >= 1) {
            IloExpr e(env);
            for (int j = 0;j < U;++j) for (size_t a = 0;a < POOL[j].size();++a) e += f[j][(IloInt)a];
            e += artC;
            Rcard = IloRange(env, (double)CARD_K, e, IloInfinity); m.add(Rcard); e.end();
        }
        IloRangeArray R1(env);
        for (int i = 0;i < P;++i) {
            IloExpr e(env);
            for (int j = 0;j < U;++j) for (size_t a = 0;a < POOL[j].size();++a)
                if (POOL[j][a].delta[i]) e += f[j][(IloInt)a];
            e += art[i];
            R1.add(IloRange(env, 1.0, e, IloInfinity)); m.add(R1[i]); e.end();
        }
        IloRangeArray R2(env);
        for (int j = 0;j < U;++j) {
            IloExpr e(env);
            for (size_t a = 0;a < POOL[j].size();++a) e += f[j][(IloInt)a];
            R2.add(IloRange(env, -IloInfinity, e, 1.0)); m.add(R2[j]); e.end();
        }
        IloRangeArray R3(env); vector<int> r3dec;
        for (size_t k = 0;k < decs.size();++k) if (decs[k].fix == 1) {
            const BDec& d = decs[k];
            IloExpr e(env); bool any = false;
            for (size_t a = 0;a < POOL[d.j].size();++a)
                if (colMatches(POOL[d.j][a], d)) { e += f[d.j][(IloInt)a]; any = true; }
            if (any) { R3.add(IloRange(env, 1.0, e, IloInfinity)); m.add(R3[R3.getSize() - 1]); r3dec.push_back((int)k); e.end(); }
            else { e.end(); env.end(); R.ok = false; return R; }
        }
        IloCplex cx(m);
        cx.setParam(IloCplex::Param::Threads, threads);
        cx.setParam(IloCplex::Param::TimeLimit, tl);
        cx.setOut(env.getNullStream()); cx.setWarning(env.getNullStream());
        if (!cx.solve()) { env.end(); return R; }
        R.ok = true; R.obj = cx.getObjValue();
        R.th.resize(P); R.la.resize(U); R.om.assign(decs.size(), 0.0);
        for (int i = 0;i < P;++i) R.th[i] = cx.getDual(R1[i]);
        for (int j = 0;j < U;++j) R.la[j] = cx.getDual(R2[j]);
        for (IloInt r = 0;r < R3.getSize();++r) R.om[r3dec[r]] = cx.getDual(R3[r]);
        R.mu = (CARD_K >= 1) ? cx.getDual(Rcard) : 0.0;
        R.artSum = 0.0; for (int i = 0;i < P;++i) R.artSum += cx.getValue(art[i]);
        if (CARD_K >= 1) R.artSum += cx.getValue(artC);
        R.integer_ = (R.artSum < INT_TOL);
        for (int j = 0;j < U;++j) for (size_t a = 0;a < POOL[j].size();++a) {
            double v = cx.getValue(f[j][(IloInt)a]);
            if (v > 1e-6) {
                R.fval[{j, (int)a}] = v;
                if (fabs(v - round(v)) > INT_TOL) R.integer_ = false;
            }
        }
    }
    catch (IloException& e) { cout << "RMP: " << e.getMessage() << endl; R.ok = false; }
    env.end();
    return R;
}

// ---- restricted master problem, persistent ----
struct PersistentRMP {
    IloEnv env; IloModel m; IloCplex cx; IloObjective obj;
    IloRangeArray R1, R2; IloRange Rcard; bool hasCard = false;
    IloNumVarArray art; IloNumVar artC;
    vector<vector<IloNumVar>> fv;
    vector<IloRange> R3;
    PersistentRMP() : env(), m(env), cx(m), obj(IloMinimize(env)),
        R1(env), R2(env), art(env), fv(U) {
        m.add(obj);
        R1 = IloRangeArray(env, P, 1.0, IloInfinity);      m.add(R1);
        R2 = IloRangeArray(env, U, -IloInfinity, 1.0);     m.add(R2);
        hasCard = (CARD_K >= 1);
        if (hasCard) { Rcard = IloRange(env, (double)CARD_K, IloInfinity); m.add(Rcard); }
        art = IloNumVarArray(env, P);
        for (int i = 0;i < P;++i) {
            IloNumColumn c = obj(ART_COST) + R1[i](1.0);
            art[i] = IloNumVar(c, 0.0, IloInfinity); c.end();
        }
        if (hasCard) {
            IloNumColumn c = obj(ART_COST) + Rcard(1.0);
            artC = IloNumVar(c, 0.0, IloInfinity); c.end();
        }
        for (int j = 0;j < U;++j) for (size_t a = 0;a < POOL[j].size();++a) addCol(j, a);
        cx.setOut(env.getNullStream()); cx.setWarning(env.getNullStream());
    }
    void addCol(int j, size_t a) {
        const Column& cc = POOL[j][a];
        IloNumColumn c = obj(cc.cost);
        for (int i = 0;i < P;++i) if (cc.delta[i]) c += R1[i](1.0);
        c += R2[j](1.0);
        if (hasCard) c += Rcard(1.0);
        fv[j].push_back(IloNumVar(c, 0.0, IloInfinity)); c.end();
    }
    void sync() {
        for (int j = 0;j < U;++j)
            for (size_t a = fv[j].size(); a < POOL[j].size(); ++a) addCol(j, a);
    }
};
static PersistentRMP* PRMP = nullptr;

RMPOut solveRMPWarm(const vector<BDec>& decs, int threads, double tl) {
    RMPOut R; if (tl <= 1) return R;
    try {
        if (!PRMP) PRMP = new PersistentRMP();
        PersistentRMP& W = *PRMP;
        W.sync();
        for (int j = 0;j < U;++j) for (auto& v : W.fv[j]) v.setUB(IloInfinity);
        for (const auto& d : decs) if (d.fix == 0)
            for (size_t a = 0;a < POOL[d.j].size();++a)
                if (colMatches(POOL[d.j][a], d)) W.fv[d.j][a].setUB(0.0);
        for (auto& r : W.R3) { W.m.remove(r); r.end(); }
        W.R3.clear();
        vector<int> r3dec;
        for (size_t k = 0;k < decs.size();++k) if (decs[k].fix == 1) {
            const BDec& d = decs[k];
            IloExpr e(W.env); bool any = false;
            for (size_t a = 0;a < POOL[d.j].size();++a)
                if (colMatches(POOL[d.j][a], d)) { e += W.fv[d.j][a]; any = true; }
            if (!any) { e.end(); R.ok = false; return R; }
            IloRange r(W.env, 1.0, e, IloInfinity);
            W.m.add(r); W.R3.push_back(r); r3dec.push_back((int)k); e.end();
        }
        W.cx.setParam(IloCplex::Param::Threads, threads);
        W.cx.setParam(IloCplex::Param::TimeLimit, tl);
        if (!W.cx.solve()) return R;
        R.ok = true; R.obj = W.cx.getObjValue();
        R.th.resize(P); R.la.resize(U); R.om.assign(decs.size(), 0.0);
        for (int i = 0;i < P;++i) R.th[i] = W.cx.getDual(W.R1[i]);
        for (int j = 0;j < U;++j) R.la[j] = W.cx.getDual(W.R2[j]);
        for (size_t r = 0;r < W.R3.size();++r) R.om[r3dec[r]] = W.cx.getDual(W.R3[r]);
        R.mu = W.hasCard ? W.cx.getDual(W.Rcard) : 0.0;
        R.artSum = 0.0;
        for (int i = 0;i < P;++i) R.artSum += W.cx.getValue(W.art[i]);
        if (W.hasCard) R.artSum += W.cx.getValue(W.artC);
        R.integer_ = (R.artSum < INT_TOL);
        for (int j = 0;j < U;++j) for (size_t a = 0;a < W.fv[j].size();++a) {
            double v = W.cx.getValue(W.fv[j][a]);
            if (v > 1e-6) {
                R.fval[{j, (int)a}] = v;
                if (fabs(v - round(v)) > INT_TOL) R.integer_ = false;
            }
        }
    }
    catch (IloException& e) { cout << "RMPwarm: " << e.getMessage() << endl; R.ok = false; }
    return R;
}

RMPOut solveRMP(const vector<BDec>& decs, int threads, double tl) {
    return WARM_OFF ? solveRMPRebuild(decs, threads, tl) : solveRMPWarm(decs, threads, tl);
}

static vector<double> omegaSum(const vector<BDec>& decs, const vector<double>& om) {
    vector<double> s(U, 0.0);
    for (size_t k = 0;k < decs.size();++k) if (decs[k].fix == 1) s[decs[k].j] += om[k];
    return s;
}

// ---- greedy pricing ----
static bool greedyPricing(int j, const vector<BDec>& decs,
    const vector<double>& th, double laj, double omj) {
    set<tuple<int, int, int, int>> forb;
    set<pair<int, int>>            forbPos;
    set<pair<int, int>>            reqPos;
    set<tuple<int, int, int>>     occ;
    vector<Srv> sel;
    for (const auto& d : decs) if (d.j == j) {
        if (d.lvl == 0) { if (d.fix == 0) return false; }
        else if (d.lvl == 1) {
            if (d.fix == 0) forbPos.insert(make_pair(d.t, d.p));
            else            reqPos.insert(make_pair(d.t, d.p));
        }
        else if (d.fix == 0) forb.insert(make_tuple(d.t, d.p, d.s, d.v));
        else {
            if (occ.count(make_tuple(d.t, d.p, d.v))) return false;
            sel.emplace_back(j, d.t, d.p, d.s, d.v);
            occ.insert(make_tuple(d.t, d.p, d.v));
        }
    }
    if ((int)sel.size() > QCAP) return false;
    vector<char> covered(P, 0);
    for (const auto& e : sel) {
        size_t b = a2base(j, get<1>(e), get<2>(e), get<3>(e), get<4>(e));
        for (int i = 0;i < P;++i) if (A2F[b + (size_t)i]) covered[i] = 1;
    }
    for (const auto& e : sel) reqPos.erase(make_pair(get<1>(e), get<2>(e)));
    for (const auto& tp : reqPos) {
        if ((int)sel.size() >= QCAP) return false;
        int t = tp.first, p = tp.second;
        double best = -1e100; int bs = -1, bv = -1;
        for (int v = 0; v < rot_[t]; ++v) {
            if (occ.count(make_tuple(t, p, v))) continue;
            for (int s = 0;s < S;++s) {
                if (forb.count(make_tuple(t, p, s, v))) continue;
                size_t b = a2base(j, t, p, s, v);
                double g = -(double)srvCost[s][t];
                for (int i = 0;i < P;++i) if (!covered[i] && A2F[b + (size_t)i]) g += th[i];
                if (g > best) { best = g; bs = s; bv = v; }
            }
        }
        if (bs < 0) return false;
        sel.emplace_back(j, t, p, bs, bv); occ.insert(make_tuple(t, p, bv));
        size_t bb = a2base(j, t, p, bs, bv);
        for (int i = 0;i < P;++i) if (A2F[bb + (size_t)i]) covered[i] = 1;
    }
    while ((int)sel.size() < QCAP) {
        double best = 1e-9; int bt = -1, bp = -1, bs = -1, bv = -1;
        for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p) {
            if (forbPos.count(make_pair(t, p))) continue;
            for (int v = 0; v < rot_[t]; ++v) {
                if (occ.count(make_tuple(t, p, v))) continue;
                for (int s = 0;s < S;++s) {
                    if (forb.count(make_tuple(t, p, s, v))) continue;
                    size_t b = a2base(j, t, p, s, v);
                    double g = -(double)srvCost[s][t];
                    for (int i = 0;i < P;++i) if (!covered[i] && A2F[b + (size_t)i]) g += th[i];
                    if (g > best) { best = g; bt = t; bp = p; bs = s; bv = v; }
                }
            }
        }
        if (bt < 0) break;
        sel.emplace_back(j, bt, bp, bs, bv); occ.insert(make_tuple(bt, bp, bv));
        size_t bb = a2base(j, bt, bp, bs, bv);
        for (int i = 0;i < P;++i) if (A2F[bb + (size_t)i]) covered[i] = 1;
    }
    if (sel.empty()) return false;
    double rc = (double)facilityCost - laj - omj;
    for (const auto& e : sel) rc += srvCost[get<3>(e)][get<1>(e)];
    for (int i = 0;i < P;++i) if (covered[i]) rc -= th[i];
    if (rc < -EPS_RC) return addColumnIfNew(j, sel, true);
    return false;
}

static bool greedyPricingUnit(int j, const vector<double>& th, double laj) {
    set<tuple<int, int, int>> occ; vector<Srv> sel;
    vector<char> covered(P, 0);
    while ((int)sel.size() < QCAP) {
        double best = 1e-9; int bt = -1, bp = -1, bs = -1, bv = -1;
        for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p) for (int v = 0; v < rot_[t]; ++v) {
            if (occ.count(make_tuple(t, p, v))) continue;
            for (int s = 0;s < S;++s) {
                size_t b = a2base(j, t, p, s, v);
                double g = 0.0;
                for (int i = 0;i < P;++i) if (!covered[i] && A2F[b + (size_t)i]) g += th[i];
                if (g > best) { best = g; bt = t; bp = p; bs = s; bv = v; }
            }
        }
        if (bt < 0) break;
        sel.emplace_back(j, bt, bp, bs, bv); occ.insert(make_tuple(bt, bp, bv));
        size_t bb = a2base(j, bt, bp, bs, bv);
        for (int i = 0;i < P;++i) if (A2F[bb + (size_t)i]) covered[i] = 1;
    }
    if (sel.empty()) return false;
    double rc = 1.0 - laj;
    for (int i = 0;i < P;++i) if (covered[i]) rc -= th[i];
    if (rc < -EPS_RC) return addColumnIfNew(j, sel, true);
    return false;
}

// ---- exact pricing ----
// ---- LP first ----
static double exactSPRebuild(int j, const vector<BDec>& decs, const vector<double>& th,
    double laj, double omj, int threads, double tl,
    bool lpOnly, bool addCols, bool& added, bool& certified,
    bool unitCost) {
    added = false; certified = false;
    if (tl <= 1) return SP_NOBND;
    double cbarLB = SP_NOBND;
    IloEnv env;
    try {
        IloModel m(env);
        IloBoolVar4D Y(env, C);
        for (int t = 0;t < C;++t) {
            Y[t] = IloBoolVar3D(env, pos_[t]);
            for (int p = 0;p < pos_[t];++p) {
                Y[t][p] = IloBoolVar2D(env, S);
                for (int s = 0;s < S;++s) Y[t][p][s] = IloBoolVarArray(env, rot_[t]);
            }
        }
        IloBoolVarArray Z(env, P);
        IloExpr ob(env);
        ob += (unitCost ? 1.0 : (double)facilityCost) - laj - omj;
        if (!unitCost)
            for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p)
                for (int s = 0;s < S;++s) for (int v = 0; v < rot_[t]; ++v)
                    ob += srvCost[s][t] * Y[t][p][s][v];
        for (int i = 0;i < P;++i) ob -= th[i] * Z[i];
        m.add(IloMinimize(env, ob)); ob.end();
        for (int i = 0;i < P;++i) {
            IloExpr e(env);
            for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p)
                for (int s = 0;s < S;++s) for (int v = 0; v < rot_[t]; ++v)
                    if (A2at(i, j, t, p, s, v)) e += Y[t][p][s][v];
            m.add(e >= Z[i]); e.end();
        }
        for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p) for (int v = 0; v < rot_[t]; ++v) {
            IloExpr e(env);
            for (int s = 0;s < S;++s) e += Y[t][p][s][v];
            m.add(e <= 1); e.end();
        }
        IloExpr cap(env);
        for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p)
            for (int s = 0;s < S;++s) for (int v = 0; v < rot_[t]; ++v) cap += Y[t][p][s][v];
        m.add(cap <= QCAP); cap.end();
        for (const auto& d : decs) if (d.j == j) {
            if (d.lvl == 0) continue;
            if (d.lvl == 1) {
                if (d.fix == 0) {
                    for (int s = 0;s < S;++s) for (int v = 0; v < rot_[d.t]; ++v)
                        Y[d.t][d.p][s][v].setUB(0);
                }
                else {
                    IloExpr e(env);
                    for (int s = 0;s < S;++s) for (int v = 0; v < rot_[d.t]; ++v)
                        e += Y[d.t][d.p][s][v];
                    m.add(e >= 1); e.end();
                }
            }
            else if (d.fix == 0) Y[d.t][d.p][d.s][d.v].setUB(0);
            else                 Y[d.t][d.p][d.s][d.v].setLB(1);
        }
        IloCplex cx(m);
        cx.setParam(IloCplex::Param::Threads, threads);
        cx.setParam(IloCplex::Param::TimeLimit, min(tl, SP_TCAP));
        cx.setOut(env.getNullStream()); cx.setWarning(env.getNullStream());

        // ---- LP first ----
        IloNumVarArray allv(env);
        for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p)
            for (int s = 0;s < S;++s) for (int v = 0; v < rot_[t]; ++v) allv.add(Y[t][p][s][v]);
        for (int i = 0;i < P;++i) allv.add(Z[i]);
        IloConversion relax(env, allv, ILOFLOAT); m.add(relax);
        bool lpok = cx.solve();
        IloAlgorithm::Status lst = cx.getStatus();
        if (lst == IloAlgorithm::Infeasible) { certified = true; env.end(); return 1e100; }
        if (lpOnly) {
            if (lst == IloAlgorithm::Optimal && lpok) { cbarLB = cx.getObjValue(); certified = true; }
            else certified = false;
            env.end(); return cbarLB;
        }
        if (lst == IloAlgorithm::Optimal && lpok && cx.getObjValue() >= -EPS_RC) {
            cbarLB = cx.getObjValue(); certified = true;
            env.end(); return cbarLB;
        }
        double lpLB = (lst == IloAlgorithm::Optimal && lpok) ? cx.getObjValue() : SP_NOBND;
        // ---- the LP promises a column: solve the MILP ----
        m.remove(relax);
        bool ok = cx.solve();
        IloAlgorithm::Status st = cx.getStatus();
        if (st == IloAlgorithm::Infeasible) { certified = true; env.end(); return 1e100; }
        double milpLB = SP_NOBND;
        try { milpLB = cx.getBestObjValue(); }
        catch (IloException&) { milpLB = SP_NOBND; }
        cbarLB = max(lpLB, milpLB);
        certified = (st == IloAlgorithm::Optimal);
        if (ok || st == IloAlgorithm::Feasible) {
            if (addCols && cx.getObjValue() < -EPS_RC) {
                vector<Srv> sel;
                for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p)
                    for (int s = 0;s < S;++s) for (int v = 0; v < rot_[t]; ++v)
                        if (cx.getValue(Y[t][p][s][v]) > 0.5) sel.emplace_back(j, t, p, s, v);
                added = addColumnIfNew(j, sel, false);
            }
        }
    }
    catch (IloException& e) {
        cout << "SP[" << j << "]: " << e.getMessage() << endl;
        cbarLB = SP_NOBND; certified = false;
    }
    env.end();
    return cbarLB;
}

// ---- exact pricing, persistent ----
// ---- LP first ----
struct PersistentSP {
    IloEnv env; IloModel m; IloCplex cx; IloObjective obj;
    IloNumVarArray Y;
    IloNumVarArray Z;
    IloNumVarArray allv;
    vector<IloRange> bRows;
    PersistentSP(int j) : env(), m(env), cx(m), obj(IloMinimize(env)),
        Y(env), Z(env), allv(env) {
        m.add(obj);
        Y = IloNumVarArray(env, SRV, 0, 1, ILOBOOL);
        Z = IloNumVarArray(env, P, 0, 1, ILOBOOL);
        for (int i = 0;i < P;++i) {
            IloExpr e(env);
            for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p)
                for (int s = 0;s < S;++s) for (int v = 0; v < rot_[t]; ++v)
                    if (A2at(i, j, t, p, s, v)) e += Y[srvLocal(t, p, s, v)];
            m.add(e >= Z[i]); e.end();
        }
        for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p)
            for (int v = 0; v < rot_[t]; ++v) {
                IloExpr e(env);
                for (int s = 0;s < S;++s) e += Y[srvLocal(t, p, s, v)];
                m.add(e <= 1); e.end();
            }
        IloExpr cap(env);
        for (int k = 0;k < SRV;++k) cap += Y[k];
        m.add(cap <= QCAP); cap.end();
        for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p)
            for (int s = 0;s < S;++s) for (int v = 0; v < rot_[t]; ++v)
                obj.setLinearCoef(Y[srvLocal(t, p, s, v)], (double)srvCost[s][t]);
        for (int k = 0;k < SRV;++k) allv.add(Y[k]);
        for (int i = 0;i < P;++i) allv.add(Z[i]);
        cx.setOut(env.getNullStream()); cx.setWarning(env.getNullStream());
    }
};
static vector<PersistentSP*> PSP;

static double exactSPWarm(int j, const vector<BDec>& decs, const vector<double>& th,
    double laj, double omj, int threads, double tl,
    bool lpOnly, bool addCols, bool& added, bool& certified) {
    added = false; certified = false;
    if (tl <= 1) return SP_NOBND;
    const double cst = (double)facilityCost - laj - omj;
    try {
        if ((int)PSP.size() < U) PSP.assign(U, nullptr);
        if (!PSP[j]) PSP[j] = new PersistentSP(j);
        PersistentSP& W = *PSP[j];
        for (auto& r : W.bRows) { W.m.remove(r); r.end(); }
        W.bRows.clear();
        for (int k = 0;k < SRV;++k) W.Y[k].setBounds(0, 1);
        for (int i = 0;i < P;++i) W.obj.setLinearCoef(W.Z[i], -th[i]);
        for (const auto& d : decs) if (d.j == j) {
            if (d.lvl == 0) continue;
            if (d.lvl == 1) {
                if (d.fix == 0) {
                    for (int s = 0;s < S;++s) for (int v = 0; v < rot_[d.t]; ++v)
                        W.Y[srvLocal(d.t, d.p, s, v)].setUB(0);
                }
                else {
                    IloExpr e(W.env);
                    for (int s = 0;s < S;++s) for (int v = 0; v < rot_[d.t]; ++v)
                        e += W.Y[srvLocal(d.t, d.p, s, v)];
                    IloRange r(W.env, 1.0, e, IloInfinity);
                    W.m.add(r); W.bRows.push_back(r); e.end();
                }
            }
            else if (d.fix == 0) W.Y[srvLocal(d.t, d.p, d.s, d.v)].setUB(0);
            else                 W.Y[srvLocal(d.t, d.p, d.s, d.v)].setLB(1);
        }
        W.cx.setParam(IloCplex::Param::Threads, threads);
        W.cx.setParam(IloCplex::Param::TimeLimit, min(tl, SP_TCAP));
        // ---- LP first ----
        IloConversion relax(W.env, W.allv, ILOFLOAT);
        W.m.add(relax);
        bool lpok = W.cx.solve();
        IloAlgorithm::Status lst = W.cx.getStatus();
        if (lst == IloAlgorithm::Infeasible) {
            W.m.remove(relax); relax.end();
            certified = true; return 1e100;
        }
        if (lpOnly) {
            double r = SP_NOBND;
            if (lst == IloAlgorithm::Optimal && lpok) { r = W.cx.getObjValue() + cst; certified = true; }
            W.m.remove(relax); relax.end();
            return r;
        }
        if (lst == IloAlgorithm::Optimal && lpok && W.cx.getObjValue() + cst >= -EPS_RC) {
            double r = W.cx.getObjValue() + cst; certified = true;
            W.m.remove(relax); relax.end();
            return r;
        }
        double lpLB = (lst == IloAlgorithm::Optimal && lpok) ? W.cx.getObjValue() + cst : SP_NOBND;
        W.m.remove(relax); relax.end();
        // ---- the LP promises a column: solve the MILP ----
        bool ok = W.cx.solve();
        IloAlgorithm::Status st = W.cx.getStatus();
        if (st == IloAlgorithm::Infeasible) { certified = true; return 1e100; }
        double milpLB = SP_NOBND;
        try { milpLB = W.cx.getBestObjValue() + cst; }
        catch (IloException&) { milpLB = SP_NOBND; }
        double cbarLB = max(lpLB, milpLB);
        certified = (st == IloAlgorithm::Optimal);
        if (ok || st == IloAlgorithm::Feasible) {
            if (addCols && W.cx.getObjValue() + cst < -EPS_RC) {
                vector<Srv> sel;
                for (int t = 0;t < C;++t) for (int p = 0;p < pos_[t];++p)
                    for (int s = 0;s < S;++s) for (int v = 0; v < rot_[t]; ++v)
                        if (W.cx.getValue(W.Y[srvLocal(t, p, s, v)]) > 0.5) sel.emplace_back(j, t, p, s, v);
                added = addColumnIfNew(j, sel, false);
            }
        }
        return cbarLB;
    }
    catch (IloException& e) {
        cout << "SPwarm[" << j << "]: " << e.getMessage() << endl;
        certified = false;
    }
    return SP_NOBND;
}

// ---- unit-cost greedy pricing ----
static double exactSP(int j, const vector<BDec>& decs, const vector<double>& th,
    double laj, double omj, int threads, double tl,
    bool lpOnly, bool addCols, bool& added, bool& certified,
    bool unitCost = false) {
    added = false; certified = false;
    for (const auto& d : decs)
        if (d.j == j && d.lvl == 0 && d.fix == 0) { certified = true; return 0.0; }
    if (WARM_OFF || unitCost || U > 256 || P > 512)
        return exactSPRebuild(j, decs, th, laj, omj, threads, tl, lpOnly, addCols, added, certified, unitCost);
    return exactSPWarm(j, decs, th, laj, omj, threads, tl, lpOnly, addCols, added, certified);
}

// ---- minimum-cardinality LP ----
static double cardLPBound(int threads, double budget, steady_clock::time_point t0, double timelimit) {
    auto s0 = steady_clock::now();
    auto remB = [&]() { return min(budget - elap(s0), timelimit - elap(t0)); };
    vector<double> th(P), la(U);
    for (int it = 0; it < 300; ++it) {
        if (remB() <= 1) return -1;
        // ---- unit-cost RMP ----
        double z = 1e100; bool ok = false;
        IloEnv env;
        try {
            IloModel m(env);
            vector<IloNumVarArray> f(U);
            for (int j = 0;j < U;++j) f[j] = IloNumVarArray(env, (IloInt)POOL[j].size(), 0.0, IloInfinity);
            IloNumVarArray art(env, P, 0.0, IloInfinity);
            IloExpr ob(env);
            for (int j = 0;j < U;++j) for (size_t a = 0;a < POOL[j].size();++a) ob += f[j][(IloInt)a];
            for (int i = 0;i < P;++i) ob += ART_COST * art[i];
            m.add(IloMinimize(env, ob)); ob.end();
            IloRangeArray R1(env);
            for (int i = 0;i < P;++i) {
                IloExpr e(env);
                for (int j = 0;j < U;++j) for (size_t a = 0;a < POOL[j].size();++a)
                    if (POOL[j][a].delta[i]) e += f[j][(IloInt)a];
                e += art[i];
                R1.add(IloRange(env, 1.0, e, IloInfinity)); m.add(R1[i]); e.end();
            }
            IloRangeArray R2(env);
            for (int j = 0;j < U;++j) {
                IloExpr e(env);
                for (size_t a = 0;a < POOL[j].size();++a) e += f[j][(IloInt)a];
                R2.add(IloRange(env, -IloInfinity, e, 1.0)); m.add(R2[j]); e.end();
            }
            IloCplex cx(m);
            cx.setParam(IloCplex::Param::Threads, threads);
            cx.setParam(IloCplex::Param::TimeLimit, remB());
            cx.setOut(env.getNullStream()); cx.setWarning(env.getNullStream());
            if (cx.solve()) {
                z = cx.getObjValue();
                for (int i = 0;i < P;++i) th[i] = cx.getDual(R1[i]);
                for (int j = 0;j < U;++j) la[j] = cx.getDual(R2[j]);
                ok = true;
            }
        }
        catch (IloException& e) { cout << "cardRMP: " << e.getMessage() << endl; }
        env.end();
        if (!ok) return -1;
        // ---- unit-cost greedy pricing ----
        bool any = false;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(||:any)
#endif
        for (int j = 0;j < U;++j)
            if (greedyPricingUnit(j, th, la[j])) any = true;
        if (any) continue;
        // ---- exact round ----
        static const vector<BDec> noDecs;
        double lagr = 0.0; bool added = false;
        for (int j = 0;j < U;++j) {
            bool ad = false, ct = false;
            double cb = exactSP(j, noDecs, th, la[j], 0.0, threads, remB(),
                false /*lpOnly*/, true /*addCols*/, ad, ct, true /*unitCost*/);
            if (remB() <= 0) return -1;
            if (spSinCota(cb)) return -1;
            added = added || ad;
            lagr += min(0.0, cb);
        }
        if (added) continue;
        return z + lagr;
    }
    return -1;
}

// ---- Lagrangian bound from the subproblem LPs ----
static double lagrangianLPBound(const vector<BDec>& decs, const RMPOut& R,
    const vector<double>& omS, int threads, double timelimit,
    steady_clock::time_point t0, bool& valid, bool& converged) {
    valid = true; converged = true;
    double lagr = 0.0;
    for (int j = 0; j < U; ++j) {
        bool ad = false, ct = false;
        double cb = exactSP(j, decs, R.th, R.la[j], omS[j], threads,
            timelimit - elap(t0), true /*lpOnly*/, false /*addCols*/, ad, ct);
        if (elap(t0) >= timelimit) { valid = false; return 0.0; }
        if (!ct) { valid = false; return 0.0; }
        if (cb < -EPS_RC) converged = false;
        lagr += min(0.0, cb);
    }
    return R.obj + lagr;
}

// ---- column generation at a node ----
struct CGOut {
    double lbValid;
    double rmpObj = 1e100;
    bool   integerSol = false, infeasible = false, converged = false;
    map<pair<int, int>, double> fval;
    int    its = 0;
};
CGOut runCG(int nodeId, double parentLb, int threads, double timelimit,
    steady_clock::time_point t0, bool isRoot) {
    CGOut O; O.lbValid = parentLb;
    vector<BDec> decs = pathDecs(nodeId);
    const int maxIt = isRoot ? MAXIT_ROOT : MAXIT_NODE;
    RMPOut R;
    for (;;) {
        double rem = timelimit - elap(t0);
        if (rem <= 1) { outOfTime = true; return O; }
        R = solveRMP(decs, threads, rem);
        ++O.its; ++CG_totalIt;
        if (!R.ok) { O.infeasible = true; O.lbValid = 1e100; return O; }
        vector<double> omS = omegaSum(decs, R.om);
        if (CARD_K >= 1)
            for (int j = 0;j < U;++j) omS[j] += R.mu;
        bool any = false;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(||:any)
#endif
        for (int j = 0;j < U;++j)
            if (greedyPricing(j, decs, R.th, R.la[j], omS[j])) any = true;
        if (!any) {
            bool doCert = isRoot || (BP_nodesSolved % CERT_EVERY == 0)
                || (BP_UB < 1e99 && R.obj >= CERT_NEAR * BP_UB)
                || R.integer_;
            if (!doCert) {
                if (LAGR_OFF) { O.lbValid = parentLb; break; }
                bool valid = false, conv = false;
                double lb = lagrangianLPBound(decs, R, omS, threads, timelimit, t0, valid, conv);
                if (!valid) { O.lbValid = parentLb; break; }
                O.lbValid = max(parentLb, intCeil(lb));
                if (conv) O.converged = true;
                break;
            }
            bool all = true, anyEx = false, allCert = true, lagrOK = true; double lagr = 0.0;
            for (int j = 0;j < U;++j) {
                bool ad = false, ct = false;
                double cb = exactSP(j, decs, R.th, R.la[j], omS[j], threads,
                    timelimit - elap(t0), false, true, ad, ct);
                if (elap(t0) >= timelimit) { all = false; break; }
                anyEx = anyEx || ad;
                allCert = allCert && ct;
                if (spSinCota(cb)) lagrOK = false;
                else lagr += min(0.0, cb);
            }
            if (!all) { outOfTime = true; return O; }
            if (!anyEx) {                              // converged only if every subproblem was certified
                O.converged = allCert;
                O.lbValid = lagrOK ? max(parentLb, intCeil(R.obj + lagr)) : parentLb;
                break;
            }
            any = true;
        }
        if (O.its >= maxIt) {
            bool doCert = isRoot || (BP_nodesSolved % CERT_EVERY == 0)
                || (BP_UB < 1e99 && R.obj >= CERT_NEAR * BP_UB)
                || R.integer_;
            if (!doCert) {
                if (LAGR_OFF) { O.lbValid = parentLb; break; }
                bool valid = false, conv = false;
                double lb = lagrangianLPBound(decs, R, omS, threads, timelimit, t0, valid, conv);
                O.lbValid = valid ? max(parentLb, intCeil(lb)) : parentLb;
                if (valid && conv) O.converged = true;
                break;
            }
            if (LAGR_OFF) { O.lbValid = parentLb; break; }
            double lagr = 0.0; bool all = true;
            for (int j = 0;j < U;++j) {
                bool ad = false, ct = false;
                double cb = exactSP(j, decs, R.th, R.la[j], omS[j], threads,
                    timelimit - elap(t0), false, false, ad, ct);
                if (elap(t0) >= timelimit) { all = false; break; }
                if (spSinCota(cb)) { all = false; break; }
                lagr += min(0.0, cb);
            }
            O.lbValid = all ? max(parentLb, intCeil(R.obj + lagr)) : parentLb;
            break;
        }
    }
    O.rmpObj = R.obj; O.fval = R.fval;
    O.infeasible = (O.converged && R.artSum > INT_TOL);
    O.integerSol = R.integer_ && !O.infeasible;
    return O;
}

// ---- restricted-master heuristic ----
void solveIMP(int threads, double budget, steady_clock::time_point t0, double timelimit,
    int lbK = 0,
    const vector<pair<int, int>>* lbCenter = nullptr) {
    auto s0 = steady_clock::now();
    IloEnv env;
    try {
        IloModel m(env);
        vector<IloNumVarArray> f(U);
        for (int j = 0;j < U;++j) f[j] = IloNumVarArray(env, (IloInt)POOL[j].size(), 0, 1, ILOBOOL);
        const vector<pair<int, int>>& CEN = lbCenter ? *lbCenter : BEST_SOL;
        if (lbK > 0 && !CEN.empty()) {
            IloExpr e(env); int cnt = 0;
            for (const auto& ja : CEN)
                if (ja.second < (int)POOL[ja.first].size()) { e += f[ja.first][(IloInt)ja.second]; ++cnt; }
            if (cnt > lbK) m.add(e >= (double)(cnt - lbK));
            e.end();
        }
        IloNumVarArray art(env, P, 0.0, IloInfinity);
        IloExpr ob(env);
        for (int j = 0;j < U;++j) for (size_t a = 0;a < POOL[j].size();++a) ob += POOL[j][a].cost * f[j][(IloInt)a];
        for (int i = 0;i < P;++i) ob += ART_COST * art[i];
        m.add(IloMinimize(env, ob)); ob.end();
        for (int i = 0;i < P;++i) {
            IloExpr e(env);
            for (int j = 0;j < U;++j) for (size_t a = 0;a < POOL[j].size();++a)
                if (POOL[j][a].delta[i]) e += f[j][(IloInt)a];
            e += art[i]; m.add(e >= 1); e.end();
        }
        for (int j = 0;j < U;++j) {
            IloExpr e(env);
            for (size_t a = 0;a < POOL[j].size();++a) e += f[j][(IloInt)a];
            m.add(e <= 1); e.end();
        }
        IloCplex cx(m);
        cx.setParam(IloCplex::Param::Threads, threads);
        cx.setParam(IloCplex::Param::TimeLimit, min(budget, timelimit - elap(t0)));
        cx.setOut(env.getNullStream()); cx.setWarning(env.getNullStream());
        if (!PRIM_OFF) {
            if ((size_t)U * (size_t)P >= 30000)
                cx.setParam(IloCplex::Param::Emphasis::MIP, 1);
            if (!BEST_SOL.empty()) {
                IloNumVarArray sv(env); IloNumArray vals(env);
                for (const auto& ja : BEST_SOL)
                    if (ja.second < (int)POOL[ja.first].size()) { sv.add(f[ja.first][(IloInt)ja.second]); vals.add(1.0); }
                if (sv.getSize() > 0) cx.addMIPStart(sv, vals);
                sv.end(); vals.end();
            }
            if (lbCenter && !lbCenter->empty()) {
                IloNumVarArray sv(env); IloNumArray vals(env);
                for (const auto& ja : *lbCenter)
                    if (ja.second < (int)POOL[ja.first].size()) { sv.add(f[ja.first][(IloInt)ja.second]); vals.add(1.0); }
                if (sv.getSize() > 0) cx.addMIPStart(sv, vals);
                sv.end(); vals.end();
            }
        }
        if (cx.solve() || cx.getStatus() == IloAlgorithm::Feasible) {
            double aS = 0; for (int i = 0;i < P;++i) aS += cx.getValue(art[i]);
            if (aS < INT_TOL) {
                IMP_obj = cx.getObjValue(); IMP_gap = cx.getMIPRelativeGap();
                if (IMP_obj < BP_UB - 1e-6) {
                    BP_UB = IMP_obj;
                    BEST_SOL.clear();
                    for (int j = 0;j < U;++j) for (size_t a = 0;a < POOL[j].size();++a)
                        if (cx.getValue(f[j][(IloInt)a]) > 0.5) BEST_SOL.push_back({ j, (int)a });
                }
            }
        }
    }
    catch (IloException& e) { cout << "IMP: " << e.getMessage() << endl; }
    env.end();
    IMP_time = duration<double>(steady_clock::now() - s0).count();
}

// ---- price-and-dive ----
static bool priceAndDive(vector<BDec> decs, int threads, double budget,
    steady_clock::time_point t0, double timelimit) {
    auto s0 = steady_clock::now();
    auto rem = [&]() { return min(budget - elap(s0), timelimit - elap(t0)); };
    ++DIVE_calls;
    set<tuple<int, int, int>> tried;
    vector<vector<BDec>> stepStack;
    int steps = 0, backtracks = 0;
    auto logDive = [&](const char* res, double obj) {
        if (TREE.is_open())
            TREE << "DIVE #" << DIVE_calls << ": pasos=" << steps << " backtracks=" << backtracks
            << " resultado=" << res << (obj < 1e99 ? "  obj=" + fmtB(obj) : string())
            << "  [t=" << (int)elap(t0) << "s]\n" << flush;
    };
    while (steps < 3 * U) {
        if (rem() <= 1) { logDive("TIEMPO", 1e100); return false; }
        RMPOut R = solveRMP(decs, threads, rem());
        if (!R.ok) { logDive("RMP_FALLO", 1e100); return false; }
        for (int itc = 0; itc < 2; ++itc) {
            vector<double> omS = omegaSum(decs, R.om);
            if (CARD_K >= 1) for (int j = 0;j < U;++j) omS[j] += R.mu;
            bool any = false;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(||:any)
#endif
            for (int j = 0;j < U;++j)
                if (greedyPricing(j, decs, R.th, R.la[j], omS[j])) any = true;
            if (!any || rem() <= 1) break;
            R = solveRMP(decs, threads, rem());
            if (!R.ok) { logDive("RMP_FALLO", 1e100); return false; }
        }
        if (R.artSum > INT_TOL) {
            if (backtracks < 5 && !stepStack.empty()) {
                for (size_t k = 0; k < stepStack.back().size(); ++k) decs.pop_back();
                stepStack.pop_back(); ++backtracks; if (steps > 0) --steps;
                continue;
            }
            logDive("MUERTO_ARTIFICIALES", 1e100); return false;
        }
        if (R.integer_) {
            if (R.obj < BP_UB - 1e-6) {
                BP_UB = R.obj; ++DIVE_hits;
                BEST_SOL.clear();
                for (const auto& kv : R.fval) if (kv.second > 0.5) BEST_SOL.push_back(kv.first);
                cout << "   * UB mejorado (price-and-dive): " << fixed << setprecision(2)
                    << BP_UB << "  [t=" << (int)elap(t0) << "s]\n";
                logDive("EXITO_MEJORA_UB", R.obj);
                return true;
            }
            logDive("ENTERO_SIN_MEJORA", R.obj);
            if (!PRIM_OFF && timelimit - elap(t0) > 15) {
                vector<pair<int, int>> ctr;
                for (const auto& kv : R.fval) if (kv.second > 0.5) ctr.push_back(kv.first);
                if (!ctr.empty()) {
                    double ubOld = BP_UB;
                    solveIMP(threads, min(60.0, timelimit - elap(t0) - 5), t0, timelimit, 3, &ctr);
                    if (BP_UB < ubOld - 1e-6) {
                        ++DIVE_hits;
                        cout << "   * UB mejorado (pulido de dive, IMP local): " << fixed << setprecision(2)
                            << BP_UB << "  [t=" << (int)elap(t0) << "s]\n";
                        if (TREE.is_open())
                            TREE << "* UB mejorado por PULIDO DE DIVE (F39, IMP local k=3 centrado en el dive): "
                            << fmtB(ubOld) << " -> " << fmtB(BP_UB) << "  [t=" << (int)elap(t0) << "s]\n" << flush;
                        return true;
                    }
                }
            }
            return false;
        }
        // ---- candidate: fractional w_j first, whole column as fallback ----
        double best = INT_TOL; int bj = -1, ba = -1, blvl = -1;
        map<int, double> wv;
        for (const auto& kv : R.fval) wv[kv.first.first] += kv.second;
        for (const auto& kv : wv) {
            if (kv.second >= 1.0 - INT_TOL || kv.second <= best) continue;
            if (tried.count(make_tuple(0, kv.first, -1))) continue;
            best = kv.second; bj = kv.first; ba = -1; blvl = 0;
        }
        if (blvl < 0) {
            best = INT_TOL;
            for (const auto& kv : R.fval) {
                if (kv.second >= 1.0 - INT_TOL || kv.second <= best) continue;
                if (tried.count(make_tuple(2, kv.first.first, kv.first.second))) continue;
                best = kv.second; bj = kv.first.first; ba = kv.first.second; blvl = 2;
            }
        }
        if (blvl < 0) { logDive("SIN_CANDIDATO", 1e100); return false; }
        vector<BDec> added;
        if (blvl == 0) {
            tried.insert(make_tuple(0, bj, -1));
            added.push_back(BDec{ 0, bj, -1, -1, -1, -1, 1 });
            for (const auto& kv : wv) {
                if (kv.first == bj || kv.second < 0.9 || kv.second >= 1.0 - INT_TOL) continue;
                if (tried.count(make_tuple(0, kv.first, -1))) continue;
                tried.insert(make_tuple(0, kv.first, -1));
                added.push_back(BDec{ 0, kv.first, -1, -1, -1, -1, 1 });
            }
        }
        else {
            tried.insert(make_tuple(2, bj, ba));
            for (const auto& e : POOL[bj][ba].srv) {
                BDec d{ 2, bj, get<1>(e), get<2>(e), get<3>(e), get<4>(e), 1 };
                bool have = false;
                for (const auto& x : decs)
                    if (x.lvl == 2 && x.fix == 1 && x.j == d.j && x.t == d.t &&
                        x.p == d.p && x.s == d.s && x.v == d.v) { have = true; break; }
                if (!have) added.push_back(d);
            }
        }
        for (const auto& d : added) decs.push_back(d);
        stepStack.push_back(added); ++steps;
    }
    logDive("TOPE_PASOS", 1e100);
    return false;
}

// ---- initial columns ----
void initialColumns() {
    POOL.assign(U, {});
    int sBig = 0; for (int s = 1;s < S;++s) if (covd_[s] > covd_[sBig]) sBig = s;
    int t = C - 1, np = pos_[t];
    int chunk = max(1, min(QCAP, np));
    for (int j = 0;j < U;++j)
        for (int start = 0; start < np; start += chunk) {
            vector<Srv> sel;
            for (int p = start; p < min(start + chunk, np); ++p) sel.emplace_back(j, t, p, sBig, 0);
            addColumnIfNew(j, sel, false);
        }
    colsHeur = 0; colsExact = 0;
}

// ---- branch-and-price ----
static void printProgressHeader() {
    cout << "\n";
    cout << "   nodo    cola          UB            LB        gap%    cols     t(s)\n";
    rule('-');
}
static string fmtBound(double x) {
    ostringstream os;
    if (x >= 1e99)  return "        +inf";
    if (x <= -1e99) return "        -inf";
    os << fixed << setprecision(2) << setw(12) << x;
    return os.str();
}
static void printProgressRow(size_t queue, double t) {
    bool haveGap = (BP_UB < 1e99 && BP_UB > 0 && BP_LB > -1e99);
    double gap = haveGap ? (BP_UB - BP_LB) / BP_UB * 100.0 : -1.0;
    cout << fixed;
    cout << "  " << setw(6) << BP_nodesSolved
        << "  " << setw(6) << queue
        << "  " << fmtBound(BP_UB)
        << "  " << fmtBound(BP_LB)
        << "  " << setw(6);
    if (haveGap) cout << setprecision(2) << gap; else cout << "--";
    cout << "  " << setw(6) << (colsHeur + colsExact)
        << "  " << setw(6) << setprecision(0) << t << "\n";
}

static void logNodeBlock(const Node& nd, const CGOut& O, const vector<BDec>& decs,
    long colsNew, double tnow) {
    if (!TREE.is_open()) return;
    map<int, double> wv; map<Srv, double> ypres;
    for (const auto& kv : O.fval) {
        int j = kv.first.first, a = kv.first.second;
        wv[j] += kv.second;
        for (const auto& e : POOL[j][a].srv) ypres[e] += kv.second;
    }
    TREE << fixed << setprecision(4);
    if (BP_nodesSolved > LOG_DETAIL_MAX) {
        TREE << "NODO " << nd.id << " (padre=" << dad[nd.id] << ",prof=" << decs.size() << ")"
            << "  lbHeredada=" << fmtB(nd.lb) << "  z_RMP=" << fmtB(O.rmpObj)
            << "  cotaValida=" << fmtB(O.lbValid) << "  itCG=" << O.its
            << "  colsNuevas=" << colsNew
            << (O.converged ? "  CONVERGIDA" : "  truncada")
            << (O.integerSol ? "  ENTERA" : "") << (O.infeasible ? "  INFACTIBLE" : "")
            << "  | UB=" << fmtB(BP_UB) << " LB=" << fmtB(BP_LB)
            << "  t=" << (int)tnow << "s\n";
        return;
    }
    TREE << "\n" << string(100, '=') << "\n";
    TREE << "NODO " << nd.id;
    if (nd.id == 0) TREE << "  (RAIZ)";
    else TREE << "  (padre=" << dad[nd.id] << ", profundidad=" << decs.size() << ")";
    TREE << "   [t=" << setprecision(1) << tnow << "s]\n" << setprecision(4);
    if (nd.id != 0) {
        const BDec& d = decOf[nd.id];
        TREE << "  Creado por la rama (nivel " << d.lvl << "): " << decName(d) << " = " << d.fix << "\n";
        TREE << "    -> " << (d.fix == 1 ? decConsInc(d) : decConsExc(d)) << "\n";
    }
    if (!decs.empty()) {
        TREE << "  Decisiones activas en la ruta (" << decs.size() << "):";
        for (const auto& d : decs) TREE << "  " << decName(d) << "=" << d.fix;
        TREE << "\n";
    }
    TREE << "  CG: iteraciones=" << O.its << "  columnas nuevas=" << colsNew
        << "  convergida=" << (O.converged ? "SI (z_RMP es el optimo LP del nodo)" : "NO (cota Lagrangiana o heredada)") << "\n";
    TREE << "  Cotas: heredada=" << fmtB(nd.lb) << "  z_RMP=" << fmtB(O.rmpObj)
        << "  cotaValida=" << fmtB(O.lbValid)
        << "   | globales: UB=" << fmtB(BP_UB) << "  LB=" << fmtB(BP_LB) << "\n";
    if (O.fval.empty()) {
        TREE << "  Solucion del maestro: (no disponible: nodo infactible o RMP no resuelto)\n";
    }
    else {
        TREE << "  Solucion del maestro (f[j][a] > 0):\n";
        for (const auto& kv : O.fval) {
            int j = kv.first.first, a = kv.first.second;
            const Column& c = POOL[j][a];
            TREE << "    f[" << j << "][" << a << "] = " << setw(6) << kv.second
                << "   costo=" << setprecision(2) << c.cost << setprecision(4)
                << "   servidores(t,p,s,v)=" << colSrvStr(c) << "\n";
        }
        TREE << "  Solucion en variables del subproblema:\n    ";
        for (const auto& kv : wv) TREE << "w[" << kv.first << "]=" << kv.second << "  ";
        TREE << "\n";
        for (const auto& kv : ypres) {
            const Srv& e = kv.first;
            TREE << "    " << yName(get<0>(e), get<1>(e), get<2>(e), get<3>(e), get<4>(e))
                << " = " << kv.second
                << (fabs(kv.second - round(kv.second)) > INT_TOL ? "   [FRACCIONAL]" : "") << "\n";
        }
        if (!decs.empty()) {
            TREE << "  Chequeo de ramas sobre esta solucion:\n";
            for (const auto& d : decs) {
                double pres = 0.0;
                for (const auto& kv : O.fval)
                    if (kv.first.first == d.j && colMatches(POOL[d.j][kv.first.second], d))
                        pres += kv.second;
                bool ok = (d.fix == 0) ? (pres <= 1e-6) : (pres >= 1.0 - 1e-4);
                TREE << "    " << decName(d) << "=" << d.fix
                    << " : presencia=" << pres << (ok ? "   OK" : "   *** VIOLADA ***") << "\n";
            }
        }
    }
}

void branchAndPrice(const string& inst, int threads, double timelimit, const ModelRes& MR) {
    initialColumns();
    stringstream nb; nb << "Results/BP_CACPR_" << inst << "_Q" << QCAP;
    if (RHO != 15) nb << "_rho" << RHO;
    if (NOROT)     nb << "_noRot";
    if (fabs(BRANCH_TARGET - 0.5) > 1e-9) nb << "_bt" << BRANCH_TARGET;
    if (LAGR_OFF) nb << "_noLagr";
    if (CARD_OFF) nb << "_noCard";
    if (HIER_OFF) nb << "_soloY";
    if (WARM_OFF) nb << "_noWarm";
    if (INTR_OFF) nb << "_noInt";
    if (PRIM_OFF) nb << "_noPrim";
    const string outBase = nb.str();
    TREE.open(outBase + "_tree.log");
    if (TREE.is_open()) {
        TREE << "LOG DEL ARBOL DE BRANCH-AND-PRICE   " << inst
            << "   Q=" << QCAP << " rho=" << RHO << (NOROT ? " noRot" : "")
            << " tl=" << (int)timelimit << "s beta=" << BRANCH_TARGET
            << (LAGR_OFF ? "  [ABLACION F29: COTAS LAGRANGIANAS DESACTIVADAS - nodos no convergidos heredan la cota del padre]" : "")
            << "   " << CODE_VERSION << "\n";
        TREE << string(100, '=') << "\n";
        TREE << "NOTACION\n"
            << "  y(j,t,p,s,v) : servidor tipo s en la posicion p de la configuracion t, rotacion v, ubicacion j\n"
            << "  f[j][a]      : variable del maestro = usar la columna a del pool de la ubicacion j\n"
            << "  presencia    : sum{ f[j][a] : la columna a contiene y(j,t,p,s,v) }  (LHS de la rama fix=1)\n"
            << "  w[j]         : sum_a f[j][a]  (grado de apertura de la ubicacion j en el LP)\n"
            << "  Ramas (F31, jerarquicas" << (HIER_OFF ? ": DESACTIVADAS, solo nivel 2" : "") << "):\n"
            << "    nivel 0  w(j)        : abrir (sum_a f[j][a] >= 1) vs cerrar la ubicacion j\n"
            << "    nivel 1  pos(j,t,p)  : usar la posicion (sum sobre s,v) vs prohibirla completa\n"
            << "    nivel 2  y(j,t,p,s,v): incluir vs excluir el servidor individual\n"
            << "    INCLUIR -> fila >= 1 en el maestro (dual omega al pricing) y SP forzado;\n"
            << "    EXCLUIR -> columnas que la contienen con f=0 y SP prohibido\n"
            << "  Orden de proceso: best-first (menor cota primero); un nodo ENTERO no convergido se RE-ENCOLA\n"
            << "  Detalle completo para los primeros " << LOG_DETAIL_MAX << " nodos; despues 1 linea por nodo.\n";
    }
    priority_queue<Node, vector<Node>, CmpN> Q;
    Q.push({ 0, -1e100 }); dad[0] = 0;
    auto t0 = steady_clock::now();
    banner("BRANCH-AND-PRICE   " + inst);
    cout << "  inicio: " << nowStr() << "     limite maximo de termino: "
        << clockPlus(timelimit) << " (si no converge antes)\n";
    if (!CARD_OFF) {
        double clb = cardLPBound(threads, min(120.0, 0.2 * timelimit), t0, timelimit);
        if (clb > 0.0) { CARD_LP = clb; CARD_K = max(0, (int)ceil(clb - 1e-6)); }
        cout << fixed << setprecision(2);
        if (CARD_K >= 1)
            cout << "  [F30] LP de cardinalidad = " << CARD_LP << "  ->  corte  sum f >= "
            << CARD_K << "   (t=" << elap(t0) << "s)\n";
        else
            cout << "  [F30] LP de cardinalidad no certificado dentro del presupuesto: sin corte\n";
        if (TREE.is_open()) {
            if (CARD_K >= 1)
                TREE << "CORTE DE CARDINALIDAD (F30): LP unitario certificado = " << fmtB(CARD_LP)
                << "  ->  se agrega al maestro de TODOS los nodos:  sum_{j,a} f[j][a] >= " << CARD_K
                << "   (dual mu entra al pricing)   [t=" << fixed << setprecision(1) << elap(t0) << "s]\n";
            else
                TREE << "CORTE DE CARDINALIDAD (F30): no certificado dentro del presupuesto -> sin corte\n";
        }
    }
    printProgressHeader();
    while (!Q.empty()) {
        if (elap(t0) >= timelimit) { outOfTime = true; break; }
        Node nd = Q.top(); Q.pop();
        if (nd.lb >= BP_UB - PRUNE_TOL) {
            if (TREE.is_open())
                TREE << "NODO " << nd.id << ": extraido de la cola y PODADO sin resolver (lb "
                << fmtB(nd.lb) << " >= UB " << fmtB(BP_UB) << " - tol)\n";
            continue;
        }
        if (nd.lb > BP_LB) BP_LB = nd.lb;
        ++BP_nodesSolved;
        bool isRoot = (nd.id == 0);
        if (!isRoot && (BP_nodesSolved <= 3 || BP_nodesSolved % 25 == 0))
            printProgressRow(Q.size(), elap(t0));
        if (!isRoot && BP_nodesSolved % IMP_EVERY == 0 && timelimit - elap(t0) > IMP_BUDGET) {
            double ubOld = BP_UB;
            double pb = (!PRIM_OFF && (size_t)U * (size_t)P >= 30000) ? 90.0 : IMP_BUDGET;
            int lbK = (!PRIM_OFF && !BEST_SOL.empty() && (IMP_ROUND++ % 2 == 0)) ? 3 : 0;
            solveIMP(threads, pb, t0, timelimit, lbK);
            if (BP_UB < ubOld - 1e-6) {
                cout << "   * UB mejorado (IMP periodico" << (lbK ? " LOCAL k=3" : " libre") << "): "
                    << fixed << setprecision(2) << BP_UB << "  [t=" << (int)elap(t0) << "s]\n";
                if (TREE.is_open())
                    TREE << "* UB mejorado por IMP periodico " << (lbK ? "LOCAL(k=3)" : "libre")
                    << " (maestro entero sobre el pool): "
                    << fmtB(ubOld) << " -> " << fmtB(BP_UB) << "  [t=" << (int)elap(t0) << "s]\n";
            }
            if (!PRIM_OFF && (size_t)U * (size_t)P >= 30000 && timelimit - elap(t0) > 10
                && !(OMP_THREADS <= 1 && U >= 200))
                priceAndDive(pathDecs(nd.id), threads, 120.0, t0, timelimit);
        }
        long colsBefore = colsHeur + colsExact;
        CGOut O = runCG(nd.id, nd.lb, threads, timelimit, t0, isRoot);
        vector<BDec> decs = pathDecs(nd.id);
        logNodeBlock(nd, O, decs, colsHeur + colsExact - colsBefore, elap(t0));
        if (isRoot) {
            rootIt = O.its; rootCols = colsHeur + colsExact;
            rootLB = O.lbValid; rootTime = elap(t0);
            if (rootLB > BP_LB) BP_LB = rootLB;
            double budget = rootTime;
            bool bigInst = ((size_t)U * (size_t)P >= 30000);
            if (!PRIM_OFF && bigInst) budget = max(budget, min(120.0, 0.10 * timelimit));
            if (timelimit - elap(t0) > 1) solveIMP(threads, budget, t0, timelimit);
            rootIMP_obj = IMP_obj; rootIMP_gap = IMP_gap; rootIMP_time = IMP_time;
            if (!PRIM_OFF && bigInst && timelimit - elap(t0) > 5)
                priceAndDive(vector<BDec>(), threads, min(300.0, 0.30 * timelimit), t0, timelimit);
            cout << fixed << setprecision(2);
            cout << "\n  [RAIZ]  itCG=" << rootIt << "  cols=" << rootCols
                << "  LB=" << rootLB << "  IMP=" << IMP_obj
                << "  UB=" << BP_UB << "  t=" << elap(t0) << "s\n";
            printProgressHeader();
            if (TREE.is_open())
                TREE << "  RAIZ: LB=" << fmtB(rootLB)
                << "  IMP(maestro entero sobre el pool)=" << fmtB(IMP_obj)
                << "  ->  UB tras la raiz=" << fmtB(BP_UB) << "\n" << flush;
        }
        if (outOfTime) {
            if (TREE.is_open()) TREE << "  RESULTADO: TIEMPO AGOTADO durante el nodo\n" << flush;
            break;
        }
        if (O.infeasible) {
            if (TREE.is_open())
                TREE << "  RESULTADO: INFACTIBLE (CG convergida con artificiales>0, o sin columnas elegibles para una rama y=1) -> PODADO\n" << flush;
            continue;
        }
        if (O.lbValid >= BP_UB - PRUNE_TOL) {
            if (TREE.is_open())
                TREE << "  RESULTADO: PODADO por cota (cotaValida " << fmtB(O.lbValid)
                << " >= UB " << fmtB(BP_UB) << " - tol)\n" << flush;
            continue;
        }
        if (O.integerSol) {
            bool improved = false;
            if (O.rmpObj < BP_UB - 1e-6) {
                BP_UB = O.rmpObj; improved = true;
                BEST_SOL.clear();
                for (const auto& kv : O.fval) if (kv.second > 0.5) BEST_SOL.push_back(kv.first);
                cout << "   * UB mejorado (nodo entero): " << fixed << setprecision(2)
                    << BP_UB << "  [t=" << (int)elap(t0) << "s]\n";
            }
            if (O.converged || O.lbValid >= BP_UB - PRUNE_TOL) {
                if (TREE.is_open())
                    TREE << "  RESULTADO: SOLUCION ENTERA con valor " << fmtB(O.rmpObj)
                    << (improved ? "  * MEJORA EL UB *" : "  (no mejora el UB)")
                    << "  -> PODADO (" << (O.converged ? "CG convergida: es el optimo del nodo" : "cota valida alcanza el UB")
                    << ")\n" << flush;
                continue;
            }
            if (TREE.is_open())
                TREE << "  RESULTADO: SOLUCION ENTERA con valor " << fmtB(O.rmpObj)
                << (improved ? "  * MEJORA EL UB *" : "  (no mejora el UB)")
                << "  pero CG truncada (cotaValida " << fmtB(O.lbValid)
                << " < UB): RE-ENCOLADO para resolverlo hasta converger (F26)\n" << flush;
            Q.push({ nd.id, O.lbValid });
            continue;
        }
        // ---- hierarchical branching candidate ----
        // ---- level 0: location ----
        // ---- level 2: individual server ----
        BDec bd{}; bool found = false; double bestd = 1.0, bestPres = 0.0;
        if (!HIER_OFF) {
            // ---- level 0: location ----
            map<int, double> wv;
            for (const auto& kv : O.fval) wv[kv.first.first] += kv.second;
            for (const auto& kv : wv) {
                if (kv.second <= INT_TOL || kv.second >= 1.0 - INT_TOL) continue;
                bool br = false;
                for (const auto& d : decs) if (d.lvl == 0 && d.j == kv.first) { br = true; break; }
                if (br) continue;
                double dd = fabs(kv.second - BRANCH_TARGET);
                if (dd < bestd - 1e-12) { bestd = dd; bd = { 0, kv.first, -1, -1, -1, -1, 0 }; bestPres = kv.second; found = true; }
            }
            // ---- level 1: aggregated position ----
            if (!found) {
                map<tuple<int, int, int>, double> pp;
                for (const auto& kv : O.fval) {
                    int j = kv.first.first, a = kv.first.second;
                    set<pair<int, int>> ps;
                    for (const auto& e : POOL[j][a].srv) ps.insert(make_pair(get<1>(e), get<2>(e)));
                    for (const auto& tp : ps) pp[make_tuple(j, tp.first, tp.second)] += kv.second;
                }
                for (const auto& kv : pp) {
                    if (kv.second <= INT_TOL || kv.second >= 1.0 - INT_TOL) continue;
                    int j = get<0>(kv.first), t = get<1>(kv.first), p = get<2>(kv.first);
                    bool br = false;
                    for (const auto& d : decs) if (d.lvl == 1 && d.j == j && d.t == t && d.p == p) { br = true; break; }
                    if (br) continue;
                    double dd = fabs(kv.second - BRANCH_TARGET);
                    if (dd < bestd - 1e-12) { bestd = dd; bd = { 1, j, t, p, -1, -1, 0 }; bestPres = kv.second; found = true; }
                }
            }
        }
        // ---- level 2: individual server ----
        if (!found) {
            map<Srv, double> pres;
            for (const auto& kv : O.fval) {
                int j = kv.first.first, a = kv.first.second; double v = kv.second;
                if (v > INT_TOL && v < 1.0 - INT_TOL)
                    for (const auto& e : POOL[j][a].srv) pres[e] += v;
            }
            for (const auto& pv : pres) {
                bool branched = false;
                for (const auto& d : decs)
                    if (d.lvl == 2 && d.j == get<0>(pv.first) && d.t == get<1>(pv.first) && d.p == get<2>(pv.first) &&
                        d.s == get<3>(pv.first) && d.v == get<4>(pv.first)) {
                        branched = true; break;
                    }
                if (branched) continue;
                double dd = fabs(pv.second - BRANCH_TARGET);
                if (dd < bestd - 1e-12) {
                    bestd = dd;
                    bd = { 2, get<0>(pv.first), get<1>(pv.first), get<2>(pv.first), get<3>(pv.first), get<4>(pv.first), 0 };
                    bestPres = pv.second; found = true;
                }
            }
        }
        if (!found) {                                  // no candidate: optimality is no longer claimed
            branchIncomplete = true;
            lbCap = min(lbCap, O.lbValid);
            cout << "   ! AVISO: nodo " << nd.id
                << " fraccional sin candidato de rama (cota " << fixed << setprecision(2)
                << O.lbValid << ")\n";
            if (TREE.is_open())
                TREE << "  RESULTADO: ! FRACCIONAL SIN CANDIDATO DE RAMA (F22) -> descartado; el arbol NO queda probado\n" << flush;
            continue;
        }
        int inc = nextNodeId++, exc = nextNodeId++;
        BDec di = bd; di.fix = 1;
        BDec de = bd; de.fix = 0;
        decOf[inc] = di; decOf[exc] = de;
        dad[inc] = nd.id; dad[exc] = nd.id;
        Q.push({ inc, O.lbValid }); Q.push({ exc, O.lbValid });
        if (TREE.is_open()) {
            TREE << fixed << setprecision(4)
                << "  RESULTADO: RAMIFICA (nivel " << bd.lvl << ") sobre " << decName(bd)
                << "   presencia=" << bestPres << "  (la mas cercana a beta=" << BRANCH_TARGET << ")\n"
                << "    hijo " << inc << " (INCLUIR, lb heredada=" << fmtB(O.lbValid) << "): " << decConsInc(di) << "\n"
                << "    hijo " << exc << " (EXCLUIR, lb heredada=" << fmtB(O.lbValid) << "): " << decConsExc(de) << "\n" << flush;
        }
    }
    double BP_time = elap(t0);
    BP_LB = min(BP_LB, lbCap);
    bool proven = Q.empty() && !outOfTime && !branchIncomplete;
    BP_proven = proven;
    if (proven) BP_LB = BP_UB;
    if (TREE.is_open()) {
        TREE << "\n" << string(100, '=') << "\n";
        TREE << "FIN DEL ARBOL: estado="
            << (proven ? "OPTIMO PROBADO" : (outOfTime ? "LIMITE DE TIEMPO" : (branchIncomplete ? "NO PROBADO (F22)" : "COLA VACIA")))
            << "  UB=" << fmtB(BP_UB) << "  LB=" << fmtB(BP_LB)
            << "  nodos procesados=" << BP_nodesSolved << "  nodos creados=" << nextNodeId
            << "  nodos sin procesar en cola=" << Q.size()
            << "  t=" << fixed << setprecision(2) << BP_time << "s\n";
        TREE.close();
    }
    double gapTree = (BP_UB < 1e99 && BP_UB > 0) ? (BP_UB - BP_LB) / BP_UB : 1.0;
    double gapDir = (MR.feasible && BP_UB < 1e99) ? (BP_UB - MR.Z) / BP_UB : 0.0;
    long totCols = colsHeur + colsExact;

    // ---- final summary ----
    cout << fixed << setprecision(2) << "\n";
    banner("RESUMEN   " + inst);
    cout << "  Parametros      : Q=" << QCAP << "  rho=" << RHO
        << (NOROT ? "  (SIN ROTACION)" : "") << "  tl=" << (int)timelimit << "s  " << CODE_VERSION << "\n";
    cout << "  Estado          : "
        << (proven ? "OPTIMO PROBADO"
            : (outOfTime ? "LIMITE DE TIEMPO"
                : (branchIncomplete ? "NO PROBADO (ver AVISO)" : "COLA VACIA"))) << "\n";
    rule('-');
    cout << "  Modelo compacto : Z=" << setw(12) << (MR.feasible ? fmtB(MR.Z) : "no resuelto")
        << "  gap=" << setw(6) << MR.gap * 100 << "%  t=" << MR.time << "s\n";
    cout << "  Raiz            : itCG=" << rootIt << "  cols=" << rootCols
        << "  LB=" << rootLB << "  t=" << rootTime << "s\n";
    cout << "     IMP raiz     : Obj=" << rootIMP_obj << "  gap=" << rootIMP_gap * 100
        << "%  t=" << rootIMP_time << "s\n";
    cout << "  Arbol           : nodos=" << BP_nodesSolved << "/" << nextNodeId
        << "  itCG=" << CG_totalIt << "  cols=" << totCols
        << " (heur=" << colsHeur << " exactas=" << colsExact << ")\n";
    rule('-');
    cout << "  UB (B&P)        : " << setw(12) << BP_UB << "\n";
    cout << "  LB (B&P)        : " << setw(12) << BP_LB << "\n";
    cout << "  gap arbol       : " << gapTree * 100 << "%    t total=" << BP_time << "s\n";
    cout << "  gap directo F10 : (UB-Z)/UB = " << gapDir * 100 << "%\n";
    rule('=');

    ofstream fo(outBase + ".out"); fo << fixed << setprecision(2);
    fo << inst << " Q:" << QCAP << " rho:" << RHO << (NOROT ? " noRot" : "") << (LAGR_OFF ? " lagrOFF" : "") << (HIER_OFF ? " soloY" : "") << (INTR_OFF ? " intOFF" : "") << " tl:" << (int)timelimit << " " << CODE_VERSION
        << " | estado:" << (proven ? "OPT" : (outOfTime ? "TL" : (branchIncomplete ? "NOPROOF" : "EMPTY")))
        << " | Modelo Z:" << (MR.feasible ? fmtB(MR.Z) : "n/a") << " gap:" << MR.gap * 100 << "% t:" << MR.time
        << " | Raiz it:" << rootIt << " cols:" << rootCols << " LB:" << rootLB << " t:" << rootTime
        << " | IMP Obj:" << rootIMP_obj << " gap:" << rootIMP_gap * 100 << "% t:" << rootIMP_time
        << " | cardLP:" << CARD_LP << " K:" << CARD_K
        << " | dive:" << DIVE_hits << "/" << DIVE_calls
        << " | Arbol nodos:" << BP_nodesSolved << "/" << nextNodeId
        << " itCG:" << CG_totalIt << " colsH:" << colsHeur << " colsE:" << colsExact
        << " UB:" << BP_UB << " LB:" << BP_LB << " gapArbol:" << gapTree * 100 << "% t:" << BP_time
        << " | gapDirecto:" << gapDir * 100 << "%" << endl;
    fo.close();

    ofstream fs(outBase + "_sol.txt"); fs << fixed << setprecision(2);
    fs << inst << " Q:" << QCAP << " rho:" << RHO << (NOROT ? " noRot" : "") << " UB:" << BP_UB
        << " estado:" << (proven ? "OPT" : (outOfTime ? "TL" : (branchIncomplete ? "NOPROOF" : "EMPTY")))
        << " instalaciones:" << BEST_SOL.size() << "\n";
    fs << "# j costo nServidores : t,p,s,v ...\n";
    for (const auto& ja : BEST_SOL) {
        const Column& cc = POOL[ja.first][ja.second];
        fs << ja.first << " " << cc.cost << " " << cc.srv.size() << " :";
        for (const auto& e : cc.srv)
            fs << " " << get<1>(e) << "," << get<2>(e) << "," << get<3>(e) << "," << get<4>(e);
        fs << "\n";
    }
    fs.close();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage: bp_cacpr <instance> <time_limit_s> <threads> [Q=3] [rho=15] [norot=0]"
            " [beta=0.5] [model=1] [reserved=0] [tree_log_nodes=2000]"
            " [lagrangian=1] [cardinality=1] [hierarchical=1] [warm_start=1] [integer_rounding=1] [primal=1]" << endl;
        cout << "  model: 0 = branch-and-price only, 1 = compact model and branch-and-price, 2 = compact model only" << endl;
        cout << "  the last six flags are the ablation switches of the paper (1 = on, 0 = off)" << endl;
        return 1;
    }
    string inst = argv[1];
    double tl = stod(argv[2]);
    int th = stoi(argv[3]);
    if (argc >= 5) QCAP = stoi(argv[4]);
    if (argc >= 6) RHO = stoi(argv[5]);
    if (argc >= 7) NOROT = (stoi(argv[6]) != 0);
    if (argc >= 8) BRANCH_TARGET = stod(argv[7]);
    if (argc >= 9) MODEL_MODE = stoi(argv[8]);
    if (argc >= 11) LOG_DETAIL_MAX = stoi(argv[10]);
    if (argc >= 12) LAGR_OFF = (stoi(argv[11]) == 0);
    if (argc >= 13) CARD_OFF = (stoi(argv[12]) == 0);
    if (argc >= 14) HIER_OFF = (stoi(argv[13]) == 0);
    if (argc >= 15) WARM_OFF = (stoi(argv[14]) == 0);
    if (argc >= 16) INTR_OFF = (stoi(argv[15]) == 0);
    if (argc >= 17) PRIM_OFF = (stoi(argv[16]) == 0);

#ifdef _OPENMP
    OMP_THREADS = omp_get_max_threads();
#endif
    banner(string("BP_CACPR  ") + CODE_VERSION);
    cout << "  Instancia : " << inst << "\n";
    cout << "  Limite    : " << tl << " s     Threads : " << th
#ifdef _OPENMP
        << "     OpenMP : SI (pricing paralelo)\n";
#else
        << "     OpenMP : NO (pricing en serie)\n";
#endif
    cout << "  Q=" << QCAP << "   rho=" << RHO
        << "   branch_target=" << BRANCH_TARGET
        << "   modelo=" << MODEL_MODE
        << "   cotas_lagrangianas=" << (LAGR_OFF ? "OFF (ablacion F29)" : "ON")
        << "   corte_cardinalidad=" << (CARD_OFF ? "OFF" : "ON")
        << "   ramificacion=" << (HIER_OFF ? "solo y" : "jerarquica (w->pos->y)")
        << "   warm_start=" << (WARM_OFF ? "OFF (reconstruccion)" : "ON (RMP/SP persistentes)")
        << "   redondeo_entero=" << (INTR_OFF ? "OFF" : "ON")
        << "   paquete_primal=" << (PRIM_OFF ? "OFF" : "ON (dive+IMP reforzado)")
        << (NOROT ? "   MODO: SIN ROTACION (C-ACP)\n" : "\n");
    cout << "  Hora de inicio : " << nowStr(true) << "\n";
    rule('=');

    if (!readInstance(inst)) return 1;
    if (!NOROT) for (int t = 0;t < C;++t)
        if (conf_[t] % RHO != 0) {
            cout << "Error: la configuracion " << conf_[t]
                << " no es multiplo de rho=" << RHO << endl;
            return 1;
        }
    processData();
    cout << "  Datos     : P=" << P << " puntos, U=" << U << " ubicaciones, C="
        << C << " configs, S=" << S << " servidores   (SRV/ubic=" << SRV << ")\n\n";

    cerr << "[" << inst << "]  inicio " << nowStr(true)
        << "   (P=" << P << " U=" << U << " C=" << C << " S=" << S << " Q=" << QCAP << ")\n";

    ModelRes MR;
    if (MODEL_MODE != 0) {
        cout << ">> [1/2] Resolviendo MODELO COMPACTO ...   (inicio " << nowStr() << ")\n";
        cerr << "   [1/2] resolviendo MODELO COMPACTO ...      " << nowStr() << flush;
        MR = solveCompactModel(th, tl);
        cerr << "  -> Z=" << fixed << setprecision(0) << MR.Z
            << " (" << setprecision(1) << MR.time << "s)\n";
    }
    else {
        cout << ">> [1/2] MODELO COMPACTO OMITIDO (modelo=0)\n";
        cerr << "   [1/2] modelo compacto OMITIDO (test de B&P puro)\n";
    }

    if (MODEL_MODE == 2) {
        stringstream po; po << "Results/MODELO_" << inst << "_Q" << QCAP;
        if (RHO != 15) po << "_rho" << RHO;
        if (NOROT)     po << "_noRot";
        po << ".out";
        ofstream fo(po.str()); fo << fixed << setprecision(2);
        fo << inst << " Q:" << QCAP << " rho:" << RHO << " tl:" << (int)tl << " threads:" << th
            << " " << CODE_VERSION
            << " | MODELO Z:" << (MR.feasible ? fmtB(MR.Z) : "n/a")
            << " gap:" << MR.gap * 100 << "% t:" << MR.time
            << (MR.feasible ? "" : " SIN_SOLUCION") << endl;
        fo.close();
        cout << "\n  Solo modelo compacto (baseline CPLEX con parametros por defecto): resultado en " << po.str() << "\n";
        cout << "  Hora de termino : " << nowStr(true) << "\n";
        cerr << "   -> solo modelo: Z=" << fixed << setprecision(0) << MR.Z
            << "  gap=" << setprecision(2) << MR.gap * 100 << "%  termino " << nowStr() << "\n";
        return 0;
    }

    cout << "\n>> [2/2] Resolviendo con BRANCH-AND-PRICE ...   (inicio " << nowStr() << ")\n";
    cerr << "   [2/2] resolviendo BRANCH-AND-PRICE ...      " << nowStr()
        << "   (tope: " << clockPlus(tl) << ")\n";
    branchAndPrice(inst, th, tl, MR);

    cout << "\n  Hora de termino : " << nowStr(true) << "\n";
    cerr << "   -> UB=" << fixed << setprecision(0) << BP_UB
        << "  estado=" << (BP_proven ? "OPTIMO" : (outOfTime ? "LIMITE-TIEMPO" : "FIN"))
        << "  termino " << nowStr() << "\n\n";
    return 0;
}
