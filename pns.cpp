#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <immintrin.h>

// AVX2/AVX512 aligned memory access requires 256/512 bit alignment. Also Intel
// processors typically use 64 byte (i.e. 512 bit) cache alignment. Hence 512
// bit alignment is a safe choice for both correctness and performance.
#include <boost/align/aligned_allocator.hpp>
template <typename T>
using vec = std::vector<T, boost::alignment::aligned_allocator<T, 512>>;

const int32_t MAX = std::numeric_limits<int32_t>::max() / 2;

enum State : int32_t {
    STATE_UPPER = -1,
    STATE_TREE  =  0,
    STATE_LOWER =  1
};

enum Dir : int32_t {
    DIR_DOWN = -1,
    DIR_UP   =  1
};

static std::chrono::duration<double> find_entering_arc_time;
static std::chrono::duration<double> find_join_node_time;
static std::chrono::duration<double> find_leaving_arc_time;
static std::chrono::duration<double> change_flows_time;
static std::chrono::duration<double> change_states_time;
static std::chrono::duration<double> update_tree_time;
static std::chrono::duration<double> update_pots_time;

struct Graph {
    vec<int32_t> tails;
    vec<int32_t> heads;
    vec<int32_t> flows;
    vec<int32_t> uppers;
    vec<int32_t> costs;
    vec<State>   states;

    vec<int32_t> diffs;
    vec<int32_t> preds;
    vec<Dir>     dirs;
    vec<int32_t> parents;
    vec<int32_t> threads;
    vec<int32_t> rev_threads;
    vec<int32_t> num_succs;
    vec<int32_t> last_succs;
    vec<int32_t> pots;

    int32_t n, N;
    int32_t m, M;

    int32_t root;

    int32_t a_in, i_in, j_in, i_out, j_out;
    int32_t i_join;
    int32_t delta;
    bool    change;

    int32_t a_curr;
    int32_t block_size;

    vec<int32_t> dirty_revs;
};

static Graph
read_dimacs(std::istream& inp)
{
    Graph graph;

    char c;

    std::string p;
    std::string line;

    while (getline(inp, line)) {
        std::istringstream ss(line);
        ss >> c;
        switch (c) {
        case 'p': // problem line
            ss >> p >> graph.n >> graph.m;
            break;
        }
        if (inp.peek() == 'n' || inp.peek() == 'a') { break; }
    }

    graph.N = graph.n + 1;
    graph.M = graph.m + graph.n;

    graph.tails.resize       (graph.M);
    graph.heads.resize       (graph.M);
    graph.flows.resize       (graph.M);
    graph.uppers.resize      (graph.M);
    graph.costs.resize       (graph.M);
    graph.states.resize      (graph.M);

    graph.diffs.resize       (graph.N);
    graph.preds.resize       (graph.N);
    graph.dirs.resize        (graph.N);
    graph.parents.resize     (graph.N);
    graph.threads.resize     (graph.N);
    graph.rev_threads.resize (graph.N);
    graph.num_succs.resize   (graph.N);
    graph.last_succs.resize  (graph.N);
    graph.pots.resize        (graph.N);

    int32_t node;
    int32_t diff;

    int32_t tail, head;
    int32_t lower, upper;
    int32_t cost;

    int32_t curr = 0;
    while (getline(inp, line)) {
        std::istringstream ss(line);
        ss >> c;
        switch (c) {
        case 'n': // node descriptor
            ss >> node >> diff;
            graph.diffs[node-1] = diff;
            break;
        case 'a': // arc descriptor
            ss >> tail >> head >> lower >> upper >> cost;
            graph.tails  [curr] = tail - 1;
            graph.heads  [curr] = head - 1;
            graph.flows  [curr] = 0;
            graph.uppers [curr] = upper;
            graph.costs  [curr] = cost;
            graph.states [curr] = STATE_LOWER;
            ++curr;
            break;
        }
    }

    graph.root = graph.n;

    graph.parents     [graph.root] = -1;
    graph.preds       [graph.root] = -1;
    graph.threads     [graph.root] = 0;
    graph.rev_threads [0]          = graph.root;
    graph.num_succs   [graph.root] = graph.N;
    graph.last_succs  [graph.root] = graph.root - 1;
    graph.diffs       [graph.root] = 0;
    graph.pots        [graph.root] = 0;

    for (int32_t i = 0, a = graph.m; i < graph.n; ++i, ++a) {
        graph.parents     [i]   = graph.root;
        graph.preds       [i]   = a;
        graph.threads     [i]   = i + 1;
        graph.rev_threads [i+1] = i;
        graph.num_succs   [i]   = 1;
        graph.last_succs  [i]   = i;
        graph.uppers      [a]   = MAX;
        graph.states      [a]   = STATE_TREE;
        if (graph.diffs[i] >= 0) {
            graph.dirs  [i] = DIR_UP;
            graph.pots  [i] = 0;
            graph.tails [a] = i;
            graph.heads [a] = graph.root;
            graph.flows [a] = graph.diffs[i];
            graph.costs [a] = 0;
        } else {
            graph.dirs  [i] = DIR_DOWN;
            graph.pots  [i] = MAX;
            graph.tails [a] = graph.root;
            graph.heads [a] = i;
            graph.flows [a] = -graph.diffs[i];
            graph.costs [a] = MAX;
        }
    }

    graph.a_curr = 0;

    return graph;
}

static void
print_solution(Graph& graph, bool flows)
{
    long solution = 0;
    for (int32_t a = 0; a < graph.m; ++a) {
        if (graph.costs[a] > 0) {
            if (flows) {
                std::cout
                    << "f"
                    << " " << graph.tails[a] + 1
                    << " " << graph.heads[a] + 1
                    << " " << graph.flows[a]
                    << "\n";
            }
            solution += (long)graph.flows[a] * (long)graph.costs[a];
        }
    }
    std::cout << "s " << solution << std::endl;
}

struct elem_t {
    int32_t ind;
    int32_t val;
};

#if defined(OMP)
#pragma omp declare \
    reduction(min : elem_t : omp_out = omp_in.val < omp_out.val ? omp_in : omp_out) \
    initializer(omp_priv = {0, 0})

static elem_t
find_entering_arc_range_omp(Graph& graph, int32_t beg_a, int32_t end_a)
{
    elem_t min = {0, 0};

#pragma omp parallel for reduction(min:min)
    for (int32_t a = beg_a; a < end_a; ++a) {
        int32_t c = graph.states[a] *
            (graph.costs[a] +
             graph.pots[graph.tails[a]] -
             graph.pots[graph.heads[a]]);
        if (min.val > c) {
            min.val = c;
            min.ind = a;
        }
    }

    return min;
}
#elif defined(AVX2)
static elem_t
find_entering_arc_range_avx2(Graph& graph, int32_t beg_a, int32_t end_a)
{
    int32_t a;

    elem_t min = {0, 0};

    __m256i incs = _mm256_set1_epi32(8);
    __m256i inds = _mm256_setr_epi32(beg_a, beg_a+1, beg_a+2, beg_a+3, beg_a+4, beg_a+5, beg_a+6, beg_a+7);
    __m256i mininds = inds;
    __m256i minvals = _mm256_set1_epi32(0);

    for (a = beg_a; a < end_a-8; a += 8) {
        __m256i tail  = _mm256_load_si256((__m256i*)&graph.tails[a]);
        __m256i head  = _mm256_load_si256((__m256i*)&graph.heads[a]);
        __m256i state = _mm256_load_si256((__m256i*)&graph.states[a]);
        __m256i cost  = _mm256_load_si256((__m256i*)&graph.costs[a]);
        __m256i pot_tail = _mm256_i32gather_epi32(&graph.pots[0], tail, 4);
        __m256i pot_head = _mm256_i32gather_epi32(&graph.pots[0], head, 4);
        cost = _mm256_add_epi32(cost, pot_tail);
        cost = _mm256_sub_epi32(cost, pot_head);
        cost = _mm256_mullo_epi32(cost, state);
        __m256i mask = _mm256_cmpgt_epi32(minvals, cost);
        mininds = _mm256_blendv_epi8(mininds, inds, mask);
        minvals = _mm256_min_epi32(cost, minvals);
        inds = _mm256_add_epi32(inds, incs);
    }

    int32_t* inds_iptr = (int32_t*)&mininds;
    int32_t* vals_iptr = (int32_t*)&minvals;

    for (int32_t i = 0; i < 8; i++) {
        if (min.val > vals_iptr[i]) {
            min.val = vals_iptr[i];
            min.ind = inds_iptr[i];
        }
    }

    for (; a < end_a; ++a) {
        int32_t c = graph.states[a] *
            (graph.costs[a] +
             graph.pots[graph.tails[a]] -
             graph.pots[graph.heads[a]]);
        if (min.val > c) {
            min.val = c;
            min.ind = a;
        }
    }

    return min;
}
#elif defined(OMP_AVX2)
struct elem_m256i_t {
    __m256i ind;
    __m256i val;
};

static void
min_m256i_f(elem_m256i_t& out, const elem_m256i_t& in)
{
    __m256i mask = _mm256_cmpgt_epi32(out.val, in.val);
    out.ind = _mm256_blendv_epi8(out.ind, in.ind, mask);
    out.val = _mm256_min_epi32(in.val, out.val);
}

#pragma omp declare \
    reduction(min : elem_m256i_t : min_m256i_f(omp_out, omp_in)) \
    initializer(omp_priv = {_mm256_set1_epi32(0), _mm256_set1_epi32(0)})

static elem_t
find_entering_arc_range_omp_avx2(Graph& graph, int32_t beg_a, int32_t end_a)
{
    int32_t a;

    elem_t min = {0, 0};

    elem_m256i_t min_m256i = {_mm256_set1_epi32(0), _mm256_set1_epi32(0)};

#pragma omp parallel for reduction(min:min_m256i) lastprivate(a)
    for (a = beg_a; a < end_a-8; a += 8) {
        __m256i inds  = _mm256_setr_epi32(a, a+1, a+2, a+3, a+4, a+5, a+6, a+7);
        __m256i tail  = _mm256_load_si256((__m256i*)&graph.tails[a]);
        __m256i head  = _mm256_load_si256((__m256i*)&graph.heads[a]);
        __m256i state = _mm256_load_si256((__m256i*)&graph.states[a]);
        __m256i cost  = _mm256_load_si256((__m256i*)&graph.costs[a]);
        __m256i pot_tail = _mm256_i32gather_epi32(&graph.pots[0], tail, 4);
        __m256i pot_head = _mm256_i32gather_epi32(&graph.pots[0], head, 4);
        cost = _mm256_add_epi32(cost, pot_tail);
        cost = _mm256_sub_epi32(cost, pot_head);
        cost = _mm256_mullo_epi32(cost, state);
        min_m256i_f(min_m256i, elem_m256i_t{inds, cost});
    }

    int32_t* inds_iptr = (int32_t*)&min_m256i.ind;
    int32_t* vals_iptr = (int32_t*)&min_m256i.val;

    for (int32_t i = 0; i < 8; i++) {
        if (min.val > vals_iptr[i]) {
            min.val = vals_iptr[i];
            min.ind = inds_iptr[i];
        }
    }

    for (; a < end_a; ++a) {
        int32_t c = graph.states[a] *
            (graph.costs[a] +
             graph.pots[graph.tails[a]] -
             graph.pots[graph.heads[a]]);
        if (min.val > c) {
            min.val = c;
            min.ind = a;
        }
    }

    return min;
}
#elif defined(AVX512)
static elem_t
find_entering_arc_range_avx512(Graph& graph, int32_t beg_a, int32_t end_a)
{
    int32_t a;

    elem_t min = {0, 0};

    __m512i incs = _mm512_set1_epi32(16);
    __m512i inds = _mm512_setr_epi32(beg_a, beg_a+1, beg_a+2, beg_a+3, beg_a+4, beg_a+5, beg_a+6, beg_a+7, beg_a+8, beg_a+9, beg_a+10, beg_a+11, beg_a+12, beg_a+13, beg_a+14, beg_a+15);
    __m512i mininds = inds;
    __m512i minvals = _mm512_set1_epi32(0);

    for (a = beg_a; a < end_a-16; a += 16) {
        __m512i tail  = _mm512_load_si512((__m512i*)&graph.tails[a]);
        __m512i head  = _mm512_load_si512((__m512i*)&graph.heads[a]);
        __m512i state = _mm512_load_si512((__m512i*)&graph.states[a]);
        __m512i cost  = _mm512_load_si512((__m512i*)&graph.costs[a]);
        __m512i pot_tail = _mm512_i32gather_epi32(tail, &graph.pots[0], 4);
        __m512i pot_head = _mm512_i32gather_epi32(head, &graph.pots[0], 4);
        cost = _mm512_add_epi32(cost, pot_tail);
        cost = _mm512_sub_epi32(cost, pot_head);
        cost = _mm512_mullo_epi32(cost, state);
        __mmask16 mask = _mm512_cmpgt_epi32_mask(minvals, cost);
        mininds = _mm512_mask_blend_epi32(mask, mininds, inds);
        minvals = _mm512_min_epi32(cost, minvals);
        inds = _mm512_add_epi32(inds, incs);
    }

    int32_t* inds_iptr = (int32_t*)&mininds;
    int32_t* vals_iptr = (int32_t*)&minvals;

    for (int32_t i = 0; i < 16; i++) {
        if (min.val > vals_iptr[i]) {
            min.val = vals_iptr[i];
            min.ind = inds_iptr[i];
        }
    }

    for (; a < end_a; ++a) {
        int32_t c = graph.states[a] *
            (graph.costs[a] +
             graph.pots[graph.tails[a]] -
             graph.pots[graph.heads[a]]);
        if (min.val > c) {
            min.val = c;
            min.ind = a;
        }
    }

    return min;
}
#elif defined(OMP_AVX512)
struct elem_m512i_t {
    __m512i ind;
    __m512i val;
};

void
min_m512i_f(elem_m512i_t& out, const elem_m512i_t& in)
{
    __mmask16 mask = _mm512_cmpgt_epi32_mask(out.val, in.val);
    out.ind = _mm512_mask_blend_epi32(mask, out.ind, in.ind);
    out.val = _mm512_min_epi32(in.val, out.val);
}

#pragma omp declare \
    reduction(min : elem_m512i_t : min_m512i_f(omp_out, omp_in)) \
    initializer(omp_priv = {_mm512_set1_epi32(0), _mm512_set1_epi32(0)})

static elem_t
find_entering_arc_range_omp_avx512(Graph& graph, int32_t beg_a, int32_t end_a)
{
    int32_t a;

    elem_t min = {0, 0};

    elem_m512i_t min_m512i = {_mm512_set1_epi32(0), _mm512_set1_epi32(0)};

#pragma omp parallel for reduction(min:min_m512i) lastprivate(a)
    for (a = beg_a; a < end_a-16; a += 16) {
        __m512i inds  = _mm512_setr_epi32(a, a+1, a+2, a+3, a+4, a+5, a+6, a+7, a+8, a+9, a+10, a+11, a+12, a+13, a+14, a+15);
        __m512i tail  = _mm512_load_si512((__m512i*)&graph.tails[a]);
        __m512i head  = _mm512_load_si512((__m512i*)&graph.heads[a]);
        __m512i state = _mm512_load_si512((__m512i*)&graph.states[a]);
        __m512i cost  = _mm512_load_si512((__m512i*)&graph.costs[a]);
        __m512i pot_tail = _mm512_i32gather_epi32(tail, &graph.pots[0], 4);
        __m512i pot_head = _mm512_i32gather_epi32(head, &graph.pots[0], 4);
        cost = _mm512_add_epi32(cost, pot_tail);
        cost = _mm512_sub_epi32(cost, pot_head);
        cost = _mm512_mullo_epi32(cost, state);
        min_m512i_f(min_m512i, elem_m512i_t{inds, cost});
    }

    int32_t* inds_iptr = (int32_t*)&min_m512i.ind;
    int32_t* vals_iptr = (int32_t*)&min_m512i.val;

    for (int32_t i = 0; i < 16; i++) {
        if (min.val > vals_iptr[i]) {
            min.val = vals_iptr[i];
            min.ind = inds_iptr[i];
        }
    }

    for (; a < end_a; ++a) {
        int32_t c = graph.states[a] *
            (graph.costs[a] +
             graph.pots[graph.tails[a]] -
             graph.pots[graph.heads[a]]);
        if (min.val > c) {
            min.val = c;
            min.ind = a;
        }
    }

    return min;
}
#else
static elem_t
find_entering_arc_range(Graph& graph, int32_t beg_a, int32_t end_a)
{
    elem_t min = {0, 0};

    for (int32_t a = beg_a; a < end_a; ++a) {
        int32_t c = graph.states[a] *
            (graph.costs[a] +
             graph.pots[graph.tails[a]] -
             graph.pots[graph.heads[a]]);
        if (min.val > c) {
            min.val = c;
            min.ind = a;
        }
    }

    return min;
}
#endif

static bool
find_entering_arc(Graph& graph)
{
    auto beg_time = std::chrono::steady_clock::now();

    int32_t end_a = graph.a_curr;

    elem_t min = {0, 0};

    for (int32_t beg_a = graph.a_curr; beg_a < graph.m; beg_a = end_a) {
        end_a = beg_a + graph.block_size;

        if (end_a > graph.m) {
            end_a = graph.m;
        }

#if defined(OMP)
        min = find_entering_arc_range_omp(graph, beg_a, end_a);
#elif defined(AVX2)
        min = find_entering_arc_range_avx2(graph, beg_a, end_a);
#elif defined(OMP_AVX2)
        min = find_entering_arc_range_omp_avx2(graph, beg_a, end_a);
#elif defined(AVX512)
        min = find_entering_arc_range_avx512(graph, beg_a, end_a);
#elif defined(OMP_AVX512)
        min = find_entering_arc_range_omp_avx512(graph, beg_a, end_a);
#else
        min = find_entering_arc_range(graph, beg_a, end_a);
#endif

        if (min.val < 0) { break; }
    }

    if (min.val >= 0) {
        for (int32_t beg_a = 0; beg_a < graph.a_curr; beg_a = end_a) {
            end_a = beg_a + graph.block_size;

            if (end_a > graph.a_curr) {
                end_a = graph.a_curr;
            }

#if defined(OMP)
            min = find_entering_arc_range_omp(graph, beg_a, end_a);
#elif defined(AVX2)
            min = find_entering_arc_range_avx2(graph, beg_a, end_a);
#elif defined(OMP_AVX2)
            min = find_entering_arc_range_omp_avx2(graph, beg_a, end_a);
#elif defined(AVX512)
            min = find_entering_arc_range_avx512(graph, beg_a, end_a);
#elif defined(OMP_AVX512)
            min = find_entering_arc_range_omp_avx512(graph, beg_a, end_a);
#else
            min = find_entering_arc_range(graph, beg_a, end_a);
#endif

            if (min.val < 0) { break; }
        }
    }

    graph.a_in = min.ind;

    graph.a_curr = end_a;

    find_entering_arc_time += std::chrono::steady_clock::now() - beg_time;

    return min.val < 0;
}

static void
find_join_node(Graph& graph)
{
    auto beg_time = std::chrono::steady_clock::now();

    int32_t i = graph.tails[graph.a_in];
    int32_t j = graph.heads[graph.a_in];

    while (i != j) {
        if (graph.num_succs[i] < graph.num_succs[j]) {
            i = graph.parents[i];
        } else {
            j = graph.parents[j];
        }
    }

    graph.i_join = i;

    find_join_node_time += std::chrono::steady_clock::now() - beg_time;
}

static void
find_leaving_arc(Graph& graph)
{
    auto beg_time = std::chrono::steady_clock::now();

    int32_t first, second;

    if (graph.states[graph.a_in] == STATE_LOWER) {
        first  = graph.tails[graph.a_in];
        second = graph.heads[graph.a_in];
    } else {
        first  = graph.heads[graph.a_in];
        second = graph.tails[graph.a_in];
    }

    graph.delta = graph.uppers[graph.a_in];

    int32_t result = 0;

    for (int32_t i = first; i != graph.i_join; i = graph.parents[i]) {
        int32_t a = graph.preds[i];
        int32_t f = graph.flows[a];
        if (graph.dirs[i] == DIR_DOWN) {
            int32_t u = graph.uppers[a];
            f = u >= MAX ? MAX : u - f;
        }
        if (f < graph.delta) {
            graph.delta = f;
            graph.i_out = i;
            result = 1;
        }
    }

    for (int32_t i = second; i != graph.i_join; i = graph.parents[i]) {
        int32_t a = graph.preds[i];
        int32_t f = graph.flows[a];
        if (graph.dirs[i] == DIR_UP) {
            int32_t u = graph.uppers[a];
            f = u >= MAX ? MAX : u - f;
        }
        if (f <= graph.delta) {
            graph.delta = f;
            graph.i_out = i;
            result = 2;
        }
    }

    if (result == 1) {
        graph.i_in = first;
        graph.j_in = second;
    } else {
        graph.i_in = second;
        graph.j_in = first;
    }

    graph.change = (result != 0);

    find_leaving_arc_time += std::chrono::steady_clock::now() - beg_time;
}

static void
change_flows(Graph& graph)
{
    if (graph.delta == 0) { return; }

    auto beg_time = std::chrono::steady_clock::now();

    int32_t f = graph.states[graph.a_in] * graph.delta;

    graph.flows[graph.a_in] += f;

    for (int32_t i = graph.tails[graph.a_in];
            i != graph.i_join;
            i = graph.parents[i]) {
        graph.flows[graph.preds[i]] -= graph.dirs[i] * f;
    }

    for (int32_t i = graph.heads[graph.a_in];
            i != graph.i_join;
            i = graph.parents[i]) {
        graph.flows[graph.preds[i]] += graph.dirs[i] * f;
    }

    change_flows_time += std::chrono::steady_clock::now() - beg_time;
}

static void
change_states(Graph& graph)
{
    auto beg_time = std::chrono::steady_clock::now();

    int32_t a_in = graph.a_in;
    int32_t a_out = graph.preds[graph.i_out];

    if (graph.change) {
        graph.states[a_in] = STATE_TREE;
        graph.states[a_out] = graph.flows[a_out] == 0 ? STATE_LOWER : STATE_UPPER;
    } else {
        graph.states[a_in] = (State)(-graph.states[a_in]);
    }

    change_states_time += std::chrono::steady_clock::now() - beg_time;
}

// adapted from lemon-1.3.1/lemon/network_simplex.h
static void
update_tree(Graph& graph)
{
    auto beg_time = std::chrono::steady_clock::now();

    int32_t old_rev_thread = graph.rev_threads[graph.i_out];
    int32_t old_succ_num = graph.num_succs[graph.i_out];
    int32_t old_last_succ = graph.last_succs[graph.i_out];
    graph.j_out = graph.parents[graph.i_out];

    // Check if i_in and i_out coincide
    if (graph.i_in == graph.i_out) {
        // Update parent, pred, dir
        graph.parents[graph.i_in] = graph.j_in;
        graph.preds[graph.i_in] = graph.a_in;
        graph.dirs[graph.i_in] = graph.i_in == graph.tails[graph.a_in] ?
            DIR_UP : DIR_DOWN;

        // Update thread and rev_thread
        if (graph.threads[graph.j_in] != graph.i_out) {
            int32_t after = graph.threads[old_last_succ];
            graph.threads[old_rev_thread] = after;
            graph.rev_threads[after] = old_rev_thread;
            after = graph.threads[graph.j_in];
            graph.threads[graph.j_in] = graph.i_out;
            graph.rev_threads[graph.i_out] = graph.j_in;
            graph.threads[old_last_succ] = after;
            graph.rev_threads[after] = old_last_succ;
        }
    } else {
        // Handle the case when old_rev_thread equals to j_in
        // (it also means that join and j_out coincide)
        int32_t thread_continue = old_rev_thread == graph.j_in ?
            graph.threads[old_last_succ] : graph.threads[graph.j_in];

        // Update _thread and _parent along the stem nodes (i.e. the nodes
        // between i_in and i_out, whose parent have to be changed)
        int32_t stem = graph.i_in;                    // the current stem node
        int32_t par_stem = graph.j_in;                // the new parent of stem
        int32_t next_stem;                            // the next stem node
        int32_t last = graph.last_succs[graph.i_in];  // the last successor of stem
        int32_t before, after = graph.threads[last];
        graph.threads[graph.j_in] = graph.i_in;
        graph.dirty_revs.clear();
        graph.dirty_revs.push_back(graph.j_in);
        while (stem != graph.i_out) {
            // Insert the next stem node into the thread list
            next_stem = graph.parents[stem];
            graph.threads[last] = next_stem;
            graph.dirty_revs.push_back(last);

            // Remove the subtree of stem from the thread list
            before = graph.rev_threads[stem];
            graph.threads[before] = after;
            graph.rev_threads[after] = before;

            // Change the parent node and shift stem nodes
            graph.parents[stem] = par_stem;
            par_stem = stem;
            stem = next_stem;

            // Update last and after
            last = graph.last_succs[stem] == graph.last_succs[par_stem] ?
                graph.rev_threads[par_stem] : graph.last_succs[stem];
            after = graph.threads[last];
        }
        graph.parents[graph.i_out] = par_stem;
        graph.threads[last] = thread_continue;
        graph.rev_threads[thread_continue] = last;
        graph.last_succs[graph.i_out] = last;

        // Remove the subtree of i_out from the thread list except for
        // the case when old_rev_thread equals to j_in
        if (old_rev_thread != graph.j_in) {
            graph.threads[old_rev_thread] = after;
            graph.rev_threads[after] = old_rev_thread;
        }

        // Update rev_thread using the new thread values
        for (int32_t i = 0; i != int32_t(graph.dirty_revs.size()); ++i) {
            int32_t u = graph.dirty_revs[i];
            graph.rev_threads[graph.threads[u]] = u;
        }

        // Update pred, dir, last_succ and num_succ for the
        // stem nodes from i_out to i_in
        int32_t tmp_sc = 0, tmp_ls = graph.last_succs[graph.i_out];
        for (int32_t u = graph.i_out, p = graph.parents[u];
                u != graph.i_in;
                u = p, p = graph.parents[u]) {
            graph.preds[u] = graph.preds[p];
            graph.dirs[u] = (Dir)(-graph.dirs[p]);
            tmp_sc += graph.num_succs[u] - graph.num_succs[p];
            graph.num_succs[u] = tmp_sc;
            graph.last_succs[p] = tmp_ls;
        }
        graph.preds[graph.i_in] = graph.a_in;
        graph.dirs[graph.i_in] = graph.i_in == graph.tails[graph.a_in] ?
            DIR_UP : DIR_DOWN;
        graph.num_succs[graph.i_in] = old_succ_num;
    }

    // Update last_succ from j_in towards the root
    int32_t up_limit_out = graph.last_succs[graph.i_join] == graph.j_in ?
        graph.i_join : -1;
    int32_t last_succ_out = graph.last_succs[graph.i_out];
    for (int32_t u = graph.j_in;
            u != -1 && graph.last_succs[u] == graph.j_in;
            u = graph.parents[u]) {
        graph.last_succs[u] = last_succ_out;
    }

    // Update last_succ from j_out towards the root
    if (graph.i_join != old_rev_thread && graph.j_in != old_rev_thread) {
        for (int32_t u = graph.j_out;
                u != up_limit_out && graph.last_succs[u] == old_last_succ;
                u = graph.parents[u]) {
            graph.last_succs[u] = old_rev_thread;
        }
    } else if (last_succ_out != old_last_succ) {
        for (int32_t u = graph.j_out;
                u != up_limit_out && graph.last_succs[u] == old_last_succ;
                u = graph.parents[u]) {
            graph.last_succs[u] = last_succ_out;
        }
    }

    // Update num_succ from j_in to i_join
    for (int32_t u = graph.j_in; u != graph.i_join; u = graph.parents[u]) {
        graph.num_succs[u] += old_succ_num;
    }
    // Update num_succ from j_out to i_join
    for (int32_t u = graph.j_out; u != graph.i_join; u = graph.parents[u]) {
        graph.num_succs[u] -= old_succ_num;
    }

    update_tree_time += std::chrono::steady_clock::now() - beg_time;
}

static void
update_pots(Graph& graph)
{
    auto beg_time = std::chrono::steady_clock::now();

    int32_t sigma = graph.pots[graph.j_in] -
        graph.pots[graph.i_in] -
        graph.dirs[graph.i_in] * graph.costs[graph.a_in];
    int32_t end = graph.threads[graph.last_succs[graph.i_in]];
    for (int32_t i = graph.i_in; i != end; i = graph.threads[i]) {
        graph.pots[i] += sigma;
    }

    update_pots_time += std::chrono::steady_clock::now() - beg_time;
}

static void
usage(const std::string& name)
{
    std::cerr
        << "usage: " << name << " [-k FACTOR] [-f] INPUT\n"
        << "\n"
        << "where\n"
        << "    -k FACTOR\n"
        << "        block size is approximately FACTOR * sqrt(number of arcs)\n"
        << "        and FACTOR is an integer with a default value of 1\n"
        << "    -f\n"
        << "        print arc flow values in the solution output\n"
        << "\n"
        << "and INPUT is a dimacs minimum cost flow file\n"
        << std::endl;
}

int
main(int argc, char* argv[])
{
    int32_t factor = 1;

    bool flows = false;

    std::string inp_name;

    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];
        if (opt == "-h") {
            usage(argv[0]);
            return 0;
        }
        if (opt.find("-k") == 0) {
            std::string val;
            opt.erase(0, 2);
            if (opt.empty()) {
                ++i;
                if (i == argc) {
                    std::cerr << "error: expected block size factor" << std::endl;
                    usage(argv[0]);
                    return 1;
                }
                val = argv[i];
            } else {
                val = opt;
            }
            std::istringstream ss(val);
            ss >> factor;
            if (!ss) {
                std::cerr << "error: expected integer as block size factor but found: " << val << std::endl;
                usage(argv[0]);
                return 1;
            }
        } else if (opt.find("-f") == 0) {
            flows = true;
        } else if (opt.front() == '-') {
            std::cerr << "error: unknown option: " << opt << std::endl;
            usage(argv[0]);
            return 1;
        } else {
            if (!inp_name.empty()) {
                std::cerr << "error: unexpected argument: " << argv[i] << std::endl;
                usage(argv[0]);
                return 1;
            }
            inp_name = argv[i];
        }
    }

    if (inp_name.empty()) {
        std::cerr << "error: expected input filename" << std::endl;
        usage(argv[0]);
        return 1;
    }

    std::cout << "c pns v1.0.0" << std::endl;

    auto beg_time = std::chrono::steady_clock::now();

    std::ifstream inp(inp_name);
    auto graph = read_dimacs(inp);
    inp.close();

    // We use a block size of a multiple of 16 if possible to match with our
    // alignment (i.e. 16 x sizeof(int32_t) = 512 bit) and minimize the number
    // of excess elements for AVX2/AVX512 implementations.
    graph.block_size = std::floor(std::sqrt(graph.m));
    graph.block_size = ((graph.block_size + 15) / 16) * 16;
    graph.block_size = factor * graph.block_size;
    graph.block_size = std::max(graph.block_size, 16);
    graph.block_size = std::min(graph.block_size, graph.m);

    auto end_time = std::chrono::steady_clock::now();

    std::cout << "c # nodes             : " << graph.n << std::endl;
    std::cout << "c # arcs              : " << graph.m << std::endl;
    std::cout << "c # arcs in a block   : " << graph.block_size << std::endl;

    std::cout
        << "c Init Time           : "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - beg_time).count()
        << " ms"
        << std::endl;

    beg_time = std::chrono::steady_clock::now();

    long num_iterations = 0;

    while (find_entering_arc(graph)) {
        // std::cout << "c # iterations : " << num_iterations << "\r" << std::flush;
        ++num_iterations;
        find_join_node(graph);
        find_leaving_arc(graph);
        change_flows(graph);
        change_states(graph);
        if (graph.change) {
            update_tree(graph);
            update_pots(graph);
        }
    }

    end_time = std::chrono::steady_clock::now();

    std::cout << "c # iterations        : " << num_iterations << std::endl;

    std::cout
        << "c Time                : "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - beg_time).count()
        << " ms"
        << std::endl;

    auto total_time = end_time - beg_time;

    std::cout
        << "c - find entering arc : "
        << std::chrono::duration_cast<std::chrono::milliseconds>(find_entering_arc_time).count()
        << " ms"
        << " (" << find_entering_arc_time / total_time << ")"
        << std::endl;

    std::cout
        << "c - find join node    : "
        << std::chrono::duration_cast<std::chrono::milliseconds>(find_join_node_time).count()
        << " ms"
        << " (" << find_join_node_time / total_time << ")"
        << std::endl;

    std::cout
        << "c - find leaving arc  : "
        << std::chrono::duration_cast<std::chrono::milliseconds>(find_leaving_arc_time).count()
        << " ms"
        << " (" << find_leaving_arc_time / total_time << ")"
        << std::endl;

    std::cout
        << "c - change flows      : "
        << std::chrono::duration_cast<std::chrono::milliseconds>(change_flows_time).count()
        << " ms"
        << " (" << change_flows_time / total_time << ")"
        << std::endl;

    std::cout
        << "c - change states     : "
        << std::chrono::duration_cast<std::chrono::milliseconds>(change_states_time).count()
        << " ms"
        << " (" << change_states_time / total_time << ")"
        << std::endl;

    std::cout
        << "c - update tree       : "
        << std::chrono::duration_cast<std::chrono::milliseconds>(update_tree_time).count()
        << " ms"
        << " (" << update_tree_time / total_time << ")"
        << std::endl;

    std::cout
        << "c - update pots       : "
        << std::chrono::duration_cast<std::chrono::milliseconds>(update_pots_time).count()
        << " ms"
        << " (" << update_pots_time / total_time << ")"
        << std::endl;

    print_solution(graph, flows);

    return 0;
}
