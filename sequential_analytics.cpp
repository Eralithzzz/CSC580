#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

using namespace std;
using namespace std::chrono;

struct Timer
{
    high_resolution_clock::time_point t0;
    void start() { t0 = high_resolution_clock::now(); }
    double elapsed_ms()
    {
        auto t1 = high_resolution_clock::now();
        return duration<double, milli>(t1 - t0).count();
    }
};

// Task 1: Basic Statistics
struct Stats
{
    double mean, variance, stddev, minVal, maxVal;
};

Stats computeStats(const vector<double> &data)
{
    Stats s;
    double sum = 0.0, sum2 = 0.0;
    s.minVal = data[0];
    s.maxVal = data[0];
    for (double x : data)
    {
        sum += x;
        sum2 += x * x;
        if (x < s.minVal)
            s.minVal = x;
        if (x > s.maxVal)
            s.maxVal = x;
    }
    s.mean = sum / data.size();
    s.variance = (sum2 / data.size()) - (s.mean * s.mean);
    s.stddev = sqrt(s.variance);
    return s;
}

// Task 2: Histogram
vector<long long> computeHistogram(const vector<double> &data,
                                   double minVal, double maxVal,
                                   int bins)
{
    vector<long long> hist(bins, 0);
    double range = maxVal - minVal;
    for (double x : data)
    {
        int bin = (int)((x - minVal) / range * bins);
        if (bin >= bins)
            bin = bins - 1;
        hist[bin]++;
    }
    return hist;
}

// Task 3: Sort
void sortData(vector<double> &data)
{
    sort(data.begin(), data.end());
}

// Task 4: Pearson Correlation
double pearsonCorrelation(const vector<double> &x, const vector<double> &y)
{
    size_t n = x.size();
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    for (size_t i = 0; i < n; i++)
    {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
        sumY2 += y[i] * y[i];
    }
    double num = (double)n * sumXY - sumX * sumY;
    double den = sqrt(((double)n * sumX2 - sumX * sumX) *
                      ((double)n * sumY2 - sumY * sumY));
    return (den == 0.0) ? 0.0 : num / den;
}

// Task 5: Moving Average
vector<double> movingAverage(const vector<double> &data, int window)
{
    size_t n = data.size();
    vector<double> result(n, 0.0);
    double windowSum = 0.0;
    for (int i = 0; i < window && i < (int)n; i++)
        windowSum += data[i];
    for (size_t i = 0; i < n; i++)
    {
        if (i >= (size_t)window)
            windowSum += data[i] - data[i - window];
        result[i] = windowSum / min((int)i + 1, window);
    }
    return result;
}

// Task 6: Outlier Detection
long long detectOutliers(const vector<double> &data, double mean, double stddev)
{
    long long count = 0;
    for (double x : data)
    {
        if (fabs((x - mean) / stddev) > 3.0)
            count++;
    }
    return count;
}

// UPDATED: Now writes both ms and s
void writeCSV(const string &filename,
              const vector<pair<string, double>> &rows)
{
    ofstream f(filename);
    f << "Task,Time_ms,Time_s\n";
    for (auto &r : rows)
    {
        f << r.first << ","
          << fixed << setprecision(3) << r.second << ","
          << fixed << setprecision(6) << (r.second / 1000.0) << "\n";
    }
    cout << "[LOG] Timing results saved to " << filename << "\n";
}

bool loadDataset(const string &filename, vector<double> &dataA, vector<double> &dataB)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Could not open file " << filename << "\n";
        return false;
    }

    string line;
    while (getline(file, line))
    {
        if (line.empty())
            continue;

        replace(line.begin(), line.end(), ',', ' ');
        stringstream ss(line);
        double val1, val2;

        if (ss >> val1)
        {
            dataA.push_back(val1);
            if (ss >> val2)
            {
                dataB.push_back(val2);
            }
            else
            {
                dataB.push_back(val1);
            }
        }
    }
    return true;
}

void runPipeline(const string &filename)
{
    cout << "\n========================================\n";
    cout << " Loading Dataset: " << filename << "\n";
    cout << "========================================\n\n";

    Timer t;
    t.start();

    vector<double> dataA, dataB;

    if (!loadDataset(filename, dataA, dataB))
    {
        cout << "[SKIP] Skipping " << filename << " as it could not be loaded.\n";
        return;
    }

    long long N = dataA.size();
    if (N == 0)
    {
        cerr << "Error: Dataset is empty or incorrectly formatted.\n";
        return;
    }

    double loadTime = t.elapsed_ms();
    cout << "[LOAD] Data loading (" << N << " rows) : " << fixed << setprecision(1) << loadTime << " ms\n\n";

    vector<pair<string, double>> timings;

    // Task 1: Basic Statistics
    t.start();
    Stats s = computeStats(dataA);
    double t1 = t.elapsed_ms();
    timings.push_back({"BasicStatistics", t1});
    cout << "[T1]   Basic Statistics       : " << t1 << " ms\n";

    // Task 2: Histogram
    t.start();
    auto hist = computeHistogram(dataA, s.minVal, s.maxVal, 100);
    double t2 = t.elapsed_ms();
    timings.push_back({"Histogram", t2});
    cout << "[T2]   Histogram (100 bins)   : " << t2 << " ms\n";

    // Task 3: Sort
    vector<double> sortBuf(dataA);
    t.start();
    sortData(sortBuf);
    double t3 = t.elapsed_ms();
    timings.push_back({"Sort", t3});
    cout << "[T3]   Sort                   : " << t3 << " ms\n";

    // Task 4: Pearson Correlation
    t.start();
    double corr = pearsonCorrelation(dataA, dataB);
    double t4 = t.elapsed_ms();
    timings.push_back({"PearsonCorrelation", t4});
    cout << "[T4]   Pearson Correlation    : " << t4 << " ms\n";

    // Task 5: Moving Average
    t.start();
    auto ma = movingAverage(dataA, 100);
    double t5 = t.elapsed_ms();
    timings.push_back({"MovingAverage", t5});
    cout << "[T5]   Moving Average (w=100) : " << t5 << " ms\n";

    // Task 6: Outlier Detection
    t.start();
    long long outliers = detectOutliers(dataA, s.mean, s.stddev);
    double t6 = t.elapsed_ms();
    timings.push_back({"OutlierDetection", t6});
    cout << "[T6]   Outlier Detection      : " << t6 << " ms\n\n";

    // Total
    double total = t1 + t2 + t3 + t4 + t5 + t6;
    timings.push_back({"TOTAL", total});
    cout << "========================================\n";
    cout << " TOTAL analytical time       : " << total << " ms\n";
    cout << "========================================\n\n";

    size_t lastdot = filename.find_last_of(".");
    string baseName = (lastdot == string::npos) ? filename : filename.substr(0, lastdot);
    string csvFile = "results_sequential_" + baseName + ".csv";

    writeCSV(csvFile, timings);
}

int main(int argc, char *argv[])
{
    cout << "=================================================\n";
    cout << " Starting Automated Sequential Analytics Batch\n";
    cout << "=================================================\n";

    vector<string> datasets = {
        "dataset_small.csv",
        "dataset_medium.csv",
        "dataset_large.csv"};

    for (const string &ds : datasets)
    {
        runPipeline(ds);
    }

    cout << "\n[DONE] All datasets processed.\n";
    return 0;
}