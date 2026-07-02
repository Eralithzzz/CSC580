import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

TASKS = ["BasicStats", "Histogram", "Sort", "Correlation", "MovingAvg", "Outliers"]
P = 4

seq_10M  = [1200, 250, 4500, 900, 600, 400]
mpi_10M  = [ 350,  80, 1200, 280, 180, 120]
seq_total  = { "1M":  500,  "10M": 7850,  "100M": 82000 }
mpi_total  = { "1M":  160,  "10M": 2210,  "100M": 23000 }

nodes_list   = [1,    2,    3,    4   ]
node_times   = [7850, 4200, 2900, 2210]

os.makedirs("charts", exist_ok=True)

def save(name):
    path = f"charts/{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(TASKS))
w = 0.35
ax.bar(x - w/2, seq_10M, w, label='Sequential', color='steelblue')
ax.bar(x + w/2, mpi_10M, w, label=f'MPI ({P} nodes)', color='tomato')
ax.set_xlabel('Analytical Task')
ax.set_ylabel('Time (ms)')
ax.set_title('EXP-1: Execution Time Comparison — Sequential vs MPI (10M points)')
ax.set_xticks(x)
ax.set_xticklabels(TASKS, rotation=20, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
save("chart1_baseline_comparison")

sizes  = [1e6, 10e6, 100e6]
labels = ['1M', '10M', '100M']
speedups = [seq_total[k] / mpi_total[k] for k in labels]

# Amdahl's Law overlay: assume f=0.05
f = 0.05
amdahl = [1.0 / (f + (1 - f) / p) for p in [1, P]]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(labels, speedups, 'o-', color='tomato', linewidth=2, markersize=8, label='Actual Speedup')
ax.axhline(y=amdahl[1], linestyle='--', color='gray', label=f"Amdahl's Law limit (f={f})")
ax.axhline(y=P, linestyle=':', color='green', label=f'Ideal speedup (×{P})')
ax.set_xlabel('Dataset Size')
ax.set_ylabel('Speedup S = T_seq / T_par')
ax.set_title('EXP-2: Speedup vs Dataset Size')
ax.legend()
ax.grid(linestyle='--', alpha=0.5)
save("chart2_speedup_vs_size")

node_labels = [f'Node {i+1}' for i in range(P)]
task_node_times = np.array([
    [t * (0.25 + 0.02 * (i - 1.5)) for i in range(P)]
    for t in mpi_10M
])

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(TASKS))
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
width = 0.2
for i in range(P):
    ax.bar(x + (i - 1.5) * width, task_node_times[:, i], width,
           label=node_labels[i], color=colors[i])
ax.set_xlabel('Task')
ax.set_ylabel('Time (ms)')
ax.set_title('EXP-3: Task-Level Time Breakdown Across 4 MPI Nodes (10M points)')
ax.set_xticks(x)
ax.set_xticklabels(TASKS, rotation=20, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
save("chart3_task_breakdown")

efficiency = [(s / m) / P * 100
              for s, m in zip(seq_10M, mpi_10M)]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(TASKS, efficiency, color='mediumseagreen', edgecolor='black')
ax.axhline(y=100, linestyle='--', color='red', label='Ideal efficiency (100%)')
ax.set_xlabel('Task')
ax.set_ylabel('Parallel Efficiency (%)')
ax.set_title(f'EXP-1: Parallel Efficiency per Task (P={P} nodes, 10M points)')
ax.set_ylim(0, 120)
for bar, val in zip(bars, efficiency):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
save("chart4_efficiency")

serial_fractions = [0.02, 0.05, 0.10]
p_range = np.linspace(1, 8, 200)

fig, ax = plt.subplots(figsize=(9, 5))
for f_val in serial_fractions:
    s_theory = [1.0 / (f_val + (1 - f_val) / p) for p in p_range]
    ax.plot(p_range, s_theory, '--', label=f"Amdahl f={int(f_val*100)}%")

actual_p  = [1,   2,   3,   4  ]
actual_sp = [1.0,
             seq_total['10M'] / (seq_total['10M'] * 0.55),
             seq_total['10M'] / (seq_total['10M'] * 0.38),
             seq_total['10M'] / mpi_total['10M'] ]
ax.plot(actual_p, actual_sp, 'o-', color='black', linewidth=2,
        markersize=8, label='Actual Speedup (10M)')
ax.set_xlabel('Number of Processors (P)')
ax.set_ylabel('Speedup S')
ax.set_title("EXP-2 + Amdahl's Law: Theoretical vs Actual Speedup")
ax.legend()
ax.grid(linestyle='--', alpha=0.5)
ax.set_xlim(1, 8)
save("chart5_amdahl_overlay")

print("\nAll 5 charts saved to ./charts/")
print("Replace placeholder timing values with your real measurements before submission.")
