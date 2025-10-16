"""
Анализ результатов сравнения Baseline vs Bifrost.
Читает CSV файлы из base_eval/ и генерирует отчет.

Usage:
    python scripts/analyze_bifrost_results.py
"""

import os
import glob
import pandas as pd
from nanochat.common import get_base_dir

def find_csv_files(pattern):
    base_dir = get_base_dir()
    eval_dir = os.path.join(base_dir, "base_eval")
    files = glob.glob(os.path.join(eval_dir, pattern))
    return sorted(files)

def parse_csv(filepath):
    """Parse evaluation CSV and return dict of metrics"""
    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip() and ',' in line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    task, accuracy, centered = parts[0], parts[1], parts[2]
                    try:
                        results[task] = {
                            'accuracy': float(accuracy),
                            'centered': float(centered)
                        }
                    except ValueError:
                        pass # Skip header or invalid lines
    return results

def main():
    print("=" * 80)
    print("Bifrost vs Baseline Comparison Report")
    print("=" * 80)
    print()
    
    # Find all baseline and bifrost CSV files
    baseline_files = find_csv_files("*baseline*.csv")
    bifrost_files = find_csv_files("*bifrost*.csv")
    
    if not baseline_files:
        print("❌ No baseline evaluation files found")
        return
    if not bifrost_files:
        print("❌ No bifrost evaluation files found")
        return
    
    print(f"Found {len(baseline_files)} baseline evaluation(s)")
    print(f"Found {len(bifrost_files)} bifrost evaluation(s)")
    print()
    
    # Parse the most recent files
    baseline_results = parse_csv(baseline_files[-1])
    bifrost_results = parse_csv(bifrost_files[-1])
    
    # Get all tasks
    all_tasks = sorted(set(baseline_results.keys()) | set(bifrost_results.keys()))
    
    # Print comparison table
    print(f"{'Task':<35} | {'Baseline':<10} | {'Bifrost':<10} | {'Delta':<10}")
    print("-" * 80)
    
    improvements = []
    for task in all_tasks:
        baseline_centered = baseline_results.get(task, {}).get('centered', 0.0)
        bifrost_centered = bifrost_results.get(task, {}).get('centered', 0.0)
        delta = bifrost_centered - baseline_centered
        improvements.append(delta)
        
        # Color code the delta
        delta_str = f"{delta:+.4f}"
        if delta > 0:
            delta_display = f"✅ {delta_str}"
        elif delta < 0:
            delta_display = f"❌ {delta_str}"
        else:
            delta_display = f" {delta_str}"
        
        print(f"{task:<35} | {baseline_centered:<10.4f} | {bifrost_centered:<10.4f} | {delta_display}")
    
    print("-" * 80)
    
    # Calculate average improvement
    if 'CORE' in all_tasks and improvements:
        core_idx = all_tasks.index('CORE')
        core_improvement = improvements[core_idx]
        print()
        print(f"📊 CORE Metric Improvement: {core_improvement:+.4f}")
        print()
        
        if core_improvement > 0:
            print("✅ Bifrost показывает УЛУЧШЕНИЕ над baseline!")
        elif core_improvement < 0:
            print("❌ Bifrost показывает УХУДШЕНИЕ относительно baseline")
        else:
            print("➖ Bifrost и baseline показывают одинаковые результаты")
    
    print()
    print("=" * 80)
    print("Detailed files:")
    print(f" Baseline: {baseline_files[-1]}")
    print(f" Bifrost: {bifrost_files[-1]}")
    print("=" * 80)

if __name__ == "__main__":
    main()