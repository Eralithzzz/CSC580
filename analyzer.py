# analyzer.py - Complete Performance Analysis Tool for MPI Assignment
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# ============================================
# FIX: Set working directory to project folder
# ============================================
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root (since script is in 'scripts/')
project_path = os.path.join(script_dir, '..')

# Change to project directory
os.chdir(project_path)

# Print working directory for debugging
print("="*60)
print("🚀 MPI PERFORMANCE ANALYZER")
print("="*60)
print(f"📁 Working Directory: {os.getcwd()}")
print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# ============================================
# PERFORMANCE ANALYZER CLASS
# ============================================

class PerformanceAnalyzer:
    def __init__(self):
        self.data = None
        self.metrics = None
        
    def load_data(self):
        """Load experiment data from Excel"""
        try:
            file_path = 'data/performance_data.xlsx'
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"❌ File not found: {os.getcwd()}\\{file_path}")
                return False
                
            print(f"📂 Loading: {file_path}")
            self.data = pd.read_excel(file_path, sheet_name='Raw_Data')
            print(f"✅ Loaded {len(self.data)} experiment runs")
            print(f"📊 Datasets: {list(self.data['Dataset_Size'].unique())}")
            print(f"📊 Run Types: {list(self.data['Run_Type'].unique())}")
            return True
        except FileNotFoundError:
            print("❌ Data file not found!")
            print(f"📝 Please create: {os.getcwd()}\\data\\performance_data.xlsx")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def calculate_metrics(self):
        """Calculate speedup, efficiency, overhead"""
        if self.data is None:
            print("❌ Load data first!")
            return None
            
        seq_data = self.data[self.data['Run_Type'] == 'Seq']
        mpi_data = self.data[self.data['Run_Type'] == 'MPI']
        
        results = []
        
        for size in ['1M', '10M', '100M']:
            t_seq = seq_data[seq_data['Dataset_Size'] == size]['Total_Time'].values
            
            if len(t_seq) == 0:
                print(f"⚠️ No sequential data for {size}")
                continue
                
            t_seq = t_seq[0]
            
            for nodes in [2, 3, 4]:
                mpi_subset = mpi_data[(mpi_data['Dataset_Size'] == size) & 
                                     (mpi_data['Nodes'] == nodes)]
                
                if len(mpi_subset) == 0:
                    continue
                    
                t_par = mpi_subset['Total_Time'].values[0]
                speedup = t_seq / t_par
                efficiency = (speedup / nodes) * 100
                overhead = (nodes * t_par) - t_seq
                
                results.append({
                    'Dataset': size,
                    'Nodes': nodes,
                    'T_Seq': round(t_seq, 3),
                    'T_Par': round(t_par, 3),
                    'Speedup': round(speedup, 3),
                    'Efficiency_%': round(efficiency, 1),
                    'Overhead_s': round(overhead, 3)
                })
        
        self.metrics = pd.DataFrame(results)
        self.metrics.to_excel('data/calculated_metrics.xlsx', index=False)
        print("\n✅ Metrics calculated and saved!")
        print("\n📊 METRICS SUMMARY:")
        print(self.metrics.to_string())
        return self.metrics
    
    def generate_chart1(self):
        """Chart 1: Sequential vs MPI Execution Time"""
        if self.data is None:
            print("❌ No data loaded!")
            return
            
        data_10m = self.data[self.data['Dataset_Size'] == '10M']
        
        if len(data_10m) == 0:
            print("❌ No 10M data found!")
            return
            
        tasks = ['Basic_Stats', 'Histogram', 'Sorting', 'Correlation', 'Moving_Avg', 'Outlier']
        task_labels = ['Basic Stats', 'Histogram', 'Sorting', 'Correlation', 'Moving Avg', 'Outlier']
        
        seq_data = data_10m[data_10m['Run_Type'] == 'Seq']
        mpi_data = data_10m[data_10m['Run_Type'] == 'MPI']
        
        if len(seq_data) == 0 or len(mpi_data) == 0:
            print("❌ Missing sequential or MPI data for 10M")
            return
            
        seq_times = seq_data[tasks].values[0]
        mpi_times = mpi_data[tasks].values[0]
        
        x = np.arange(len(tasks))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, seq_times, width, label='Sequential', color='#3498db')
        bars2 = ax.bar(x + width/2, mpi_times, width, label='MPI (4 Nodes)', color='#e74c3c')
        
        ax.set_xlabel('Analytical Tasks', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.set_title('Chart 1: Sequential vs MPI Execution Time (10M dataset)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('charts/Chart1_Execution_Time_Comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Chart 1 saved: charts/Chart1_Execution_Time_Comparison.png")
    
    def generate_chart2(self):
        """Chart 2: Speedup vs Dataset Size"""
        if self.data is None:
            print("❌ No data loaded!")
            return
            
        # Calculate metrics if not done
        if self.metrics is None:
            self.calculate_metrics()
            
        if self.metrics is None or len(self.metrics) == 0:
            print("❌ No metrics calculated!")
            return
            
        sizes = ['1M', '10M', '100M']
        speedups = []
        
        for size in sizes:
            speedup = self.metrics[self.metrics['Dataset'] == size]['Speedup'].values
            if len(speedup) > 0:
                speedups.append(speedup[0])
            else:
                speedups.append(0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sizes, speedups, marker='o', linewidth=2, markersize=10, color='#2ecc71')
        ax.axhline(y=4, color='red', linestyle='--', label='Ideal Speedup (4 nodes)')
        
        ax.set_xlabel('Dataset Size', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title('Chart 2: Speedup vs Dataset Size (4 nodes)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, s in enumerate(speedups):
            if s > 0:
                ax.text(i, s + 0.1, f'{s:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('charts/Chart2_Speedup_vs_Dataset.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Chart 2 saved: charts/Chart2_Speedup_vs_Dataset.png")
    
    def generate_chart3(self):
        """Chart 3: Task-Level Time Breakdown for MPI"""
        if self.data is None:
            print("❌ No data loaded!")
            return
            
        data_10m = self.data[self.data['Dataset_Size'] == '10M']
        
        if len(data_10m) == 0:
            print("❌ No 10M data found!")
            return
            
        tasks = ['Basic_Stats', 'Histogram', 'Sorting', 'Correlation', 'Moving_Avg', 'Outlier']
        task_labels = ['Basic Stats', 'Histogram', 'Sorting', 'Correlation', 'Moving Avg', 'Outlier']
        
        seq_data = data_10m[data_10m['Run_Type'] == 'Seq']
        mpi_data = data_10m[data_10m['Run_Type'] == 'MPI']
        
        if len(seq_data) == 0 or len(mpi_data) == 0:
            print("❌ Missing sequential or MPI data for 10M")
            return
            
        seq_times = seq_data[tasks].values[0]
        mpi_times = mpi_data[tasks].values[0]
        
        x = np.arange(len(tasks))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, seq_times, width, label='Sequential', color='#3498db')
        bars2 = ax.bar(x + width/2, mpi_times, width, label='MPI (4 Nodes)', color='#e74c3c')
        
        ax.set_xlabel('Analytical Tasks', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.set_title('Chart 3: Task-Level Time Breakdown (10M dataset)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('charts/Chart3_Task_Breakdown.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Chart 3 saved: charts/Chart3_Task_Breakdown.png")
    
    def generate_chart4(self):
        """Chart 4: Efficiency per Task"""
        if self.data is None:
            print("❌ No data loaded!")
            return
            
        data_10m = self.data[self.data['Dataset_Size'] == '10M']
        
        if len(data_10m) == 0:
            print("❌ No 10M data found!")
            return
            
        tasks = ['Basic_Stats', 'Histogram', 'Sorting', 'Correlation', 'Moving_Avg', 'Outlier']
        task_labels = ['Basic Stats', 'Histogram', 'Sorting', 'Correlation', 'Moving Avg', 'Outlier']
        
        seq_data = data_10m[data_10m['Run_Type'] == 'Seq']
        mpi_data = data_10m[data_10m['Run_Type'] == 'MPI']
        
        if len(seq_data) == 0 or len(mpi_data) == 0:
            print("❌ Missing sequential or MPI data for 10M")
            return
            
        seq_times = seq_data[tasks].values[0]
        mpi_times = mpi_data[tasks].values[0]
        
        # Calculate efficiency for each task
        efficiencies = []
        for i in range(len(tasks)):
            if mpi_times[i] > 0:
                eff = (seq_times[i] / (mpi_times[i] * 4)) * 100
                efficiencies.append(eff)
            else:
                efficiencies.append(0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(task_labels, efficiencies, color='#9b59b6')
        ax.axhline(y=100, color='red', linestyle='--', label='Ideal Efficiency (100%)')
        
        ax.set_xlabel('Analytical Tasks', fontsize=12)
        ax.set_ylabel('Efficiency (%)', fontsize=12)
        ax.set_title('Chart 4: Parallel Efficiency per Task (10M dataset)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('charts/Chart4_Efficiency.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Chart 4 saved: charts/Chart4_Efficiency.png")
    
    def generate_chart5(self):
        """Chart 5: Amdahl's Law - Theoretical vs Actual Speedup"""
        if self.data is None:
            print("❌ No data loaded!")
            return
            
        # Calculate metrics if not done
        if self.metrics is None:
            self.calculate_metrics()
            
        if self.metrics is None or len(self.metrics) == 0:
            print("❌ No metrics calculated!")
            return
        
        # Get speedup for 10M dataset
        speedup_data = self.metrics[self.metrics['Dataset'] == '10M']
        
        if len(speedup_data) == 0:
            print("❌ No 10M speedup data found!")
            return
            
        nodes = speedup_data['Nodes'].values
        actual_speedups = speedup_data['Speedup'].values
        
        # Calculate sequential fraction using Amdahl's Law
        # S = 1 / (f + (1-f)/P)
        # Rearranged: f = (1/S - 1/P) / (1 - 1/P)
        if len(actual_speedups) >= 1 and nodes[0] > 1:
            # Use the 4-node speedup to estimate f
            s4 = actual_speedups[-1] if len(actual_speedups) >= 2 else actual_speedups[0]
            p4 = nodes[-1] if len(nodes) >= 2 else nodes[0]
            f = (1/s4 - 1/p4) / (1 - 1/p4)
            f = max(0, min(1, f))  # Clamp between 0 and 1
            
            # Calculate theoretical speedups
            theoretical = []
            for p in nodes:
                if p == 1:
                    theoretical.append(1.0)
                else:
                    s = 1 / (f + (1-f)/p)
                    theoretical.append(round(s, 3))
        else:
            # Default if not enough data
            f = 0.1
            theoretical = [1 / (f + (1-f)/p) for p in nodes]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot actual speedups
        ax.plot(nodes, actual_speedups, marker='o', linewidth=2, 
                markersize=10, color='#3498db', label='Actual Speedup')
        
        # Plot theoretical speedups
        ax.plot(nodes, theoretical, marker='s', linewidth=2, 
                markersize=10, color='#e74c3c', linestyle='--', label='Theoretical (Amdahl)')
        
        ax.set_xlabel('Number of Nodes', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title('Chart 5: Amdahl\'s Law - Theoretical vs Actual Speedup (10M dataset)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (x, s, t) in enumerate(zip(nodes, actual_speedups, theoretical)):
            ax.text(x, s + 0.1, f'{s:.2f}x', ha='center', va='bottom', fontweight='bold', color='#3498db')
            ax.text(x, t - 0.3, f'{t:.2f}x', ha='center', va='top', fontweight='bold', color='#e74c3c')
        
        # Add f value text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, f'Sequential Fraction (f) = {f:.3f}', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('charts/Chart5_Amdahl_Overlay.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Chart 5 saved: charts/Chart5_Amdahl_Overlay.png")
    
    def generate_all_charts(self):
        """Generate all 5 charts"""
        if self.data is None:
            print("❌ Load data first!")
            return
        
        print("\n📊 Generating Charts...")
        
        # Make sure charts folder exists
        os.makedirs('charts', exist_ok=True)
        
        self.generate_chart1()
        self.generate_chart2()
        self.generate_chart3()
        self.generate_chart4()
        self.generate_chart5()
        
        print("\n✅ All charts generated!")
        print(f"📁 Charts saved in: {os.getcwd()}\\charts\\")

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    
    print("\n" + "="*60)
    print("📋 HOW TO USE:")
    print("="*60)
    print("1. Create data/performance_data.xlsx with your experiment data")
    print("2. Fill in the 3 sheets (Raw_Data, Node_Scalability, Communication_Overhead)")
    print("3. Run this script again: python scripts/analyzer.py")
    print("4. Your charts will appear in the 'charts' folder!")
    print("="*60)
    
    # Try to load data if it exists
    if analyzer.load_data():
        analyzer.calculate_metrics()
        analyzer.generate_all_charts()
    else:
        print("\n💡 TIP: Create the Excel template first!")
        print(f"📝 Excel file should be at: {os.getcwd()}\\data\\performance_data.xlsx")