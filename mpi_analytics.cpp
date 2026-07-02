#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>
#include <numeric>
#include <string>

using namespace std;

struct Timings
{
    double stats, histogram, sort_, correlation, movingAvg, outlier;
    double commOverhead, total;
};

// Task 1: Local basic-statistics accumulators
struct LocalStats
{
    double sum, sum2, minVal, maxVal;
    long long n;
};

LocalStats computeLocalStats(const vector<double> &v)
{
    LocalStats ls = {0, 0, v[0], v[0], (long long)v.size()};
    for (double x : v)
    {
        ls.sum += x;
        ls.sum2 += x * x;
        if (x < ls.minVal)
            ls.minVal = x;
        if (x > ls.maxVal)
            ls.maxVal = x;
    }
    return ls;
}

// Task 2: Local histogram
vector<long long> computeLocalHistogram(const vector<double> &v,
                                        double gMin, double gMax, int bins)
{
    vector<long long> hist(bins, 0);
    double range = gMax - gMin;
    for (double x : v)
    {
        int b = (int)((x - gMin) / range * bins);
        if (b >= bins)
            b = bins - 1;
        hist[b]++;
    }
    return hist;
}

// Task 3: Sort local chunk
void sortChunk(vector<double> &v) { sort(v.begin(), v.end()); }

// Task 4: Local Pearson sums
struct PearsonSums
{
    double sX, sY, sXY, sX2, sY2;
    long long n;
};

PearsonSums computeLocalPearson(const vector<double> &x, const vector<double> &y)
{
    PearsonSums p = {0, 0, 0, 0, 0, (long long)x.size()};
    for (size_t i = 0; i < x.size(); i++)
    {
        p.sX += x[i];
        p.sY += y[i];
        p.sXY += x[i] * y[i];
        p.sX2 += x[i] * x[i];
        p.sY2 += y[i] * y[i];
    }
    return p;
}

// Task 5: Local moving average (window=100)
vector<double> computeMovingAvg(const vector<double> &v, int w)
{
    size_t n = v.size();
    vector<double> res(n, 0.0);
    double s = 0;
    for (int i = 0; i < w && i < (int)n; i++)
        s += v[i];
    for (size_t i = 0; i < n; i++)
    {
        if (i >= (size_t)w)
            s += v[i] - v[i - w];
        res[i] = s / min((int)i + 1, w);
    }
    return res;
}

// Task 6: Local Z-score outlier count
long long countOutliers(const vector<double> &v, double mean, double stddev)
{
    long long cnt = 0;
    for (double x : v)
        if (fabs((x - mean) / stddev) > 3.0)
            cnt++;
    return cnt;
}

void writeCSV(const string &filename, int P, const Timings &tm)
{
    ofstream f(filename);
    f << "Task,Time_ms\n";
    auto row = [&](const string &name, double t)
    {
        f << name << "," << fixed << setprecision(3) << t << "\n";
    };
    row("BasicStatistics", tm.stats);
    row("Histogram", tm.histogram);
    row("Sort", tm.sort_);
    row("PearsonCorrelation", tm.correlation);
    row("MovingAverage", tm.movingAvg);
    row("OutlierDetection", tm.outlier);
    row("CommOverhead", tm.commOverhead);
    row("TOTAL", tm.total);
    cout << "[LOG] Results -> " << filename << "\n";
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    long long N = 10000000LL;
    if (argc >= 2)
        N = atoll(argv[1]);
    MPI_Bcast(&N, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    vector<int> counts(P), displs(P);
    {
        long long base = N / P, rem = N % P;
        for (int i = 0; i < P; i++)
        {
            counts[i] = (int)(base + (i < rem ? 1 : 0));
            displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        }
    }
    int myCount = counts[rank];

    vector<double> fullA, fullB;
    if (rank == 0)
    {
        fullA.resize(N);
        fullB.resize(N);
        mt19937_64 rng(42);
        uniform_real_distribution<double> dist(0.0, 10000.0);
        for (long long i = 0; i < N; i++)
            fullA[i] = dist(rng);
        mt19937_64 rng2(1234567890ULL);
        for (long long i = 0; i < N; i++)
            fullB[i] = dist(rng2);
    }

    double commStart = MPI_Wtime();
    vector<double> chunkA(myCount), chunkB(myCount);
    MPI_Scatterv(rank == 0 ? fullA.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 chunkA.data(), myCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? fullB.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 chunkB.data(), myCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double commEnd = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    double wallStart = MPI_Wtime();
    double commAccum = (commEnd - commStart) * 1000.0; // scatter overhead
    Timings tm = {};

    // TASK 1 – Basic Statistics
    MPI_Barrier(MPI_COMM_WORLD);
    double ts = MPI_Wtime();

    LocalStats ls = computeLocalStats(chunkA);
    double gSum, gSum2, gMin, gMax;
    long long gN;
    MPI_Reduce(&ls.sum, &gSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ls.sum2, &gSum2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ls.minVal, &gMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ls.maxVal, &gMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ls.n, &gN, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Bcast(&gMin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gMax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double gMean = 0, gStd = 0;
    if (rank == 0)
    {
        gMean = gSum / gN;
        gStd = sqrt(gSum2 / gN - gMean * gMean);
        cout << "[T1] Stats: mean=" << fixed << setprecision(4) << gMean
             << "  stddev=" << gStd
             << "  min=" << gMin << "  max=" << gMax << "\n";
    }
    MPI_Bcast(&gMean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gStd, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    tm.stats = (MPI_Wtime() - ts) * 1000.0;

    // TASK 2 – Histogram
    MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();
    const int BINS = 100;
    auto lhist = computeLocalHistogram(chunkA, gMin, gMax, BINS);
    vector<long long> ghist(BINS, 0);
    MPI_Reduce(lhist.data(), ghist.data(), BINS, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    tm.histogram = (MPI_Wtime() - ts) * 1000.0;
    if (rank == 0)
        cout << "[T2] Histogram: " << BINS << " bins done\n";

    // TASK 3 – Parallel Sample Sort
    MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();
    sortChunk(chunkA);

    vector<double> sortedAll;
    if (rank == 0)
        sortedAll.resize(N);
    MPI_Gatherv(chunkA.data(), myCount, MPI_DOUBLE,
                sortedAll.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        for (int i = 1; i < P; i++)
            inplace_merge(sortedAll.begin(),
                          sortedAll.begin() + displs[i],
                          sortedAll.begin() + displs[i] + counts[i]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    tm.sort_ = (MPI_Wtime() - ts) * 1000.0;
    if (rank == 0)
        cout << "[T3] Sort: merged " << N << " elements\n";

    MPI_Scatterv(rank == 0 ? fullA.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 chunkA.data(), myCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // TASK 4 – Pearson Correlation
    MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();
    PearsonSums pl = computeLocalPearson(chunkA, chunkB);
    double gsX, gsY, gsXY, gsX2, gsY2;
    long long gnP;
    MPI_Reduce(&pl.sX, &gsX, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pl.sY, &gsY, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pl.sXY, &gsXY, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pl.sX2, &gsX2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pl.sY2, &gsY2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pl.n, &gnP, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    tm.correlation = (MPI_Wtime() - ts) * 1000.0;
    if (rank == 0)
    {
        double num = (double)gnP * gsXY - gsX * gsY;
        double den = sqrt(((double)gnP * gsX2 - gsX * gsX) * ((double)gnP * gsY2 - gsY * gsY));
        cout << "[T4] Pearson r = " << (den == 0 ? 0.0 : num / den) << "\n";
    }

    // TASK 5 – Moving Average
    MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();
    auto ma = computeMovingAvg(chunkA, 100);
    vector<double> globalMA;
    if (rank == 0)
        globalMA.resize(N);
    MPI_Gatherv(ma.data(), myCount, MPI_DOUBLE,
                globalMA.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    tm.movingAvg = (MPI_Wtime() - ts) * 1000.0;
    if (rank == 0)
        cout << "[T5] Moving Average (window=100) done\n";

    // TASK 6 – Outlier Detection
    MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();
    long long localOut = countOutliers(chunkA, gMean, gStd);
    long long globalOut = 0;
    MPI_Reduce(&localOut, &globalOut, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    tm.outlier = (MPI_Wtime() - ts) * 1000.0;
    if (rank == 0)
        cout << "[T6] Outliers: " << globalOut << "\n";

    // ── Summary ──────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    tm.total = (MPI_Wtime() - wallStart) * 1000.0;
    tm.commOverhead = commAccum;

    if (rank == 0)
    {
        cout << "\n========================================\n";
        cout << " MPI Analytics Pipeline  P=" << P << "  N=" << N << "\n";
        cout << "----------------------------------------\n";
        cout << " T1 BasicStatistics  : " << fixed << setprecision(1) << tm.stats << " ms\n";
        cout << " T2 Histogram        : " << tm.histogram << " ms\n";
        cout << " T3 Sort             : " << tm.sort_ << " ms\n";
        cout << " T4 Correlation      : " << tm.correlation << " ms\n";
        cout << " T5 MovingAverage    : " << tm.movingAvg << " ms\n";
        cout << " T6 OutlierDetection : " << tm.outlier << " ms\n";
        cout << "----------------------------------------\n";
        cout << " Comm Overhead       : " << tm.commOverhead << " ms\n";
        cout << " TOTAL               : " << tm.total << " ms\n";
        cout << "========================================\n\n";

        string csvFile = "mpi_results_" + to_string(N) + ".csv";
        writeCSV(csvFile, P, tm);
    }

    MPI_Finalize();
    return 0;
}
