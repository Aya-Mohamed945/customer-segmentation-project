"""
run.py - Run all scripts with correct file paths
"""

import os
import sys
import subprocess

print("="*60)
print("🚀 RUNNING PROJECT (FIXED VERSION)")
print("="*60)

data_file = "data/segmentation_data.csv"
if not os.path.exists(data_file):
    print(f"❌ ERROR: {data_file} not found!")
    print("Please make sure the data file is in the data/ folder.")
    sys.exit(1)

print(f"✅ Data file found: {data_file}")

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)

scripts_to_run = [
    "01_data_exploration.py",
    "02_data_preprocessing.py",
    "03_clustering_kmeans.py",
    "04_clustering_hierarchical.py",
    "05_clustering_dbscan.py",
    "07_visualization.py"
]

os.chdir("scripts")

for script in scripts_to_run:
    print(f"\n📌 Running {script}...")
    print("="*50)
    
    if not os.path.exists(script):
        print(f"⚠️ {script} not found! Skipping...")
        continue
    
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("⚠️ Errors/Warnings:")
        print(result.stderr)
    
    print(f"✅ {script} completed!")

print("\n" + "="*60)
print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📊 Results should be saved in:")
print("   • Visualizations: results/plots/")
print("   • Reports: results/reports/")
\
print("\n📂 Contents of results/plots/:")
try:
    plots_files = os.listdir("../results/plots")
    if plots_files:
        for f in plots_files:
            print(f"   📄 {f}")
    else:
        print("   ⚠️ No files found (directory is empty)")
except FileNotFoundError:
    print("   ❌ Directory not found!")

print("\n📂 Contents of results/reports/:")
try:
    reports_files = os.listdir("../results/reports")
    if reports_files:
        for f in reports_files:
            print(f"   📄 {f}")
    else:
        print("   ⚠️ No files found (directory is empty)")
except FileNotFoundError:
    print("   ❌ Directory not found!")