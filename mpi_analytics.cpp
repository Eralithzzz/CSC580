#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <string>
#include <sstream>

using namespace std;

struct Timings
{
    double stats, histogram, sort_, correlation, movingAvg, outlier;
    double compTotal, commOverhead, totalExecution, overheadPct;
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
    f << "Task,Time_ms,Time_s\n";
    auto row = [&](const string &name, double t)
    {
        f << name << "," 
          << fixed << setprecision(3) << t << ","
          << fixed << setprecision(6) << (t / 1000.0) << "\n";
    };
    row("BasicStatistics", tm.stats);
    row("Histogram", tm.histogram);
    row("Sort", tm.sort_);
    row("PearsonCorrelation", tm.correlation);
    row("MovingAverage", tm.movingAvg);
    row("OutlierDetection", tm.outlier);
    row("Computation_Total", tm.compTotal);
    row("Communication_Overhead", tm.commOverhead);
    row("Execution_Total", tm.totalExecution);
    f << "Overhead_Percent," << fixed << setprecision(2) << tm.overheadPct << ",-\n";
}

bool loadDataset(const string& filename, vector<double>& dataA, vector<double>& dataB) {
    ifstream file(filename);
    if (!file.is_open()) return false;

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        replace(line.begin(), line.end(), ',', ' ');
        stringstream ss(line);
        double val1, val2;
        
        if (ss >> val1) {
            dataA.push_back(val1);
            if (ss >> val2) dataB.push_back(val2);
            else dataB.push_back(val1);
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 2) {
        if (rank == 0) cerr << "Usage: mpiexec -n <P> " << argv[0] << " <dataset_filename.csv>\n";
        MPI_Finalize();
        return 1;
    }

    string filename = argv[1];
    long long N = 0;
    vector<double> fullA, fullB;

    if (rank == 0) {
        cout << "===========================================================================\n";
        cout << " MPI Data Analytics Pipeline (P=" << P << ")\n";
        cout << " Loading Dataset: " << filename << "\n";
        cout << "===========================================================================\n\n";

        double loadStart = MPI_Wtime();
        if (!loadDataset(filename, fullA, fullB)) {
            cerr << "[ERROR] Could not load dataset.\n";
            N = -1; 
        } else {
            N = fullA.size();
            double loadEnd = MPI_Wtime();
            cout << "[LOAD] Data loading (" << N << " rows) : " 
                 << fixed << setprecision(1) << (loadEnd - loadStart) * 1000.0 << " ms\n\n";
        }
    }

    MPI_Bcast(&N, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    if (N <= 0) {
        MPI_Finalize();
        return 1;
    }

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

    double wallStart = MPI_Wtime(); // Start total execution timer here
    
    vector<double> chunkA(myCount), chunkB(myCount);
    MPI_Scatterv(rank == 0 ? fullA.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 chunkA.data(), myCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? fullB.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 chunkB.data(), myCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
    if (rank == 0) {
        gMean = gSum / gN;
        gStd = sqrt(gSum2 / gN - gMean * gMean);
    }
    MPI_Bcast(&gMean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gStd, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    tm.stats = (MPI_Wtime() - ts) * 1000.0;
    if (rank == 0) {
        cout << "[T1]   Basic Statistics      : " << fixed << setprecision(1) << tm.stats << " ms\n";
        cout << "       Mean=" << gMean << "  StdDev=" << gStd 
             << "  Min=" << gMin << "  Max=" << gMax << "\n\n";
    }

    // TASK 2 – Histogram
    MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();
    const int BINS = 100;
    auto lhist = computeLocalHistogram(chunkA, gMin, gMax, BINS);
    vector<long long> ghist(BINS, 0);
    MPI_Reduce(lhist.data(), ghist.data(), BINS, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    tm.histogram = (MPI_Wtime() - ts) * 1000.0;
    if (rank == 0) {
        cout << "[T2]   Histogram (100 bins)  : " << fixed << setprecision(1) << tm.histogram << " ms\n\n";
    }

    // TASK 3 – Parallel Sample Sort
    MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();
    sortChunk(chunkA);
    vector<double> sortedAll;
    if (rank == 0) sortedAll.resize(N);
    MPI_Gatherv(chunkA.data(), myCount, MPI_DOUBLE,
                sortedAll.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int i = 1; i < P; i++)
            inplace_merge(sortedAll.begin(),
                          sortedAll.begin() + displs[i],
                          sortedAll.begin() + displs[i] + counts[i]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    tm.sort_ = (MPI_Wtime() - ts) * 1000.0;
    if (rank == 0) {
        cout << "[T3]   Sort                  : " << fixed << setprecision(1) << tm.sort_ << " ms\n\n";
    }
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
    if (rank == 0) {
        double num = (double)gnP * gsXY - gsX * gsY;
        double den = sqrt(((double)gnP * gsX2 - gsX * gsX) * ((double)gnP * gsY2 - gsY * gsY));
        double corr = (den == 0 ? 0.0 : num / den);
        
        cout << "[T4]   Pearson Correlation   : " << fixed << setprecision(1) << tm.correlation << " ms\n";
        cout << "       r = " << fixed << setprecision(4) << corr << "\n\n";
    }

    // TASK 5 – Moving Average
    MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();
    auto ma = computeMovingAvg(chunkA, 100);
    vector<double> globalMA;
    if (rank == 0) globalMA.resize(N);
    MPI_Gatherv(ma.data(), myCount, MPI_DOUBLE,
                globalMA.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    tm.movingAvg = (MPI_Wtime() - ts) * 1000.0;
    if (rank == 0) {
        cout << "[T5]   Moving Average (w=100): " << fixed << setprecision(1) << tm.movingAvg << " ms\n\n";
    }

    // TASK 6 – Outlier Detection
    MPI_Barrier(MPI_COMM_WORLD);
    ts = MPI_Wtime();
    long long localOut = countOutliers(chunkA, gMean, gStd);
    long long globalOut = 0;
    MPI_Reduce(&localOut, &globalOut, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    tm.outlier = (MPI_Wtime() - ts) * 1000.0;
    if (rank == 0) {
        cout << "[T6]   Outlier Detection     : " << fixed << setprecision(1) << tm.outlier << " ms\n";
        cout << "       Outliers found: " << globalOut << "\n\n";
    }

    // ── Metric Calculations & Final Summary ──────────────────────────────
    MPI_Barrier(MPI_COMM_WORLD);
    tm.totalExecution = (MPI_Wtime() - wallStart) * 1000.0; 
    
    // Total Computation is the exact sum of the 6 analytical tasks
    tm.compTotal = tm.stats + tm.histogram + tm.sort_ + tm.correlation + tm.movingAvg + tm.outlier;
    
    // Everything else (scatter, gather, barriers) falls under Communication Overhead
    tm.commOverhead = tm.totalExecution - tm.compTotal;
    
    // Prevent negative overhead due to extreme micro-fluctuations in timing
    if (tm.commOverhead < 0) tm.commOverhead = 0; 
    
    // Calculate the percentage
    tm.overheadPct = (tm.commOverhead / tm.totalExecution) * 100.0;

    if (rank == 0)
    {
        cout << "SUMMARY OF EXECUTION TIMES FOR DATASET " << N << " (P=" << P << ")\n";
        cout << string(75, '-') << "\n";
        cout << left << setw(35) << "Task" 
             << right << setw(15) << "Time (ms)" 
             << right << setw(20) << "Time (s)" << "\n";
        cout << string(75, '-') << "\n";

        auto printRow = [&](const string& name, double ms) {
            cout << left << setw(35) << name 
                 << right << setw(15) << fixed << setprecision(3) << ms 
                 << right << setw(20) << fixed << setprecision(6) << (ms / 1000.0) << "\n";
        };

        printRow("1. Basic Statistics", tm.stats);
        printRow("2. Histogram Generation", tm.histogram);
        printRow("3. Parallel Sample Sort", tm.sort_);
        printRow("4. Pearson Correlation", tm.correlation);
        printRow("5. Moving Average", tm.movingAvg);
        printRow("6. Z-Score Outliers Detection", tm.outlier);

        cout << string(75, '-') << "\n";
        
        // Detailed final output block matching your request
        printRow("Total Computation Time", tm.compTotal);
        printRow("Total Comm/Overhead Time", tm.commOverhead);
        
        cout << string(75, '-') << "\n";
        printRow("TOTAL EXECUTION TIME", tm.totalExecution);
        
        cout << "\nOverhead Percentage              : " << fixed << setprecision(2) << tm.overheadPct << " %\n";
        cout << string(75, '=') << "\n\n";

        size_t lastSlash = filename.find_last_of("\\/");
        string justName = (lastSlash == string::npos) ? filename : filename.substr(lastSlash + 1);
        size_t lastdot = justName.find_last_of(".");
        string baseName = (lastdot == string::npos) ? justName : justName.substr(0, lastdot);
        string csvFile = "results_mpi_" + baseName + "_P" + to_string(P) + ".csv";
        
        writeCSV(csvFile, P, tm);
        cout << "[LOG] Timing results saved to " << csvFile << "\n";
    }

    MPI_Finalize();
    return 0;
}