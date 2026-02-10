#!/usr/bin/env python3
"""
Standalone script to create detailed comparison visualization from CSV.
Usage: python visualize_comparison.py path/to/regime_comparison_BTCUSDT_1m.csv
"""

import sys
import pandas as pd
from pathlib import Path

# Import the visualizer
from regime_visualizer import RegimeVisualizer

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_comparison.py <path_to_comparison_csv>")
        print("\nExample:")
        print("  python visualize_comparison.py results/regimes/regime_comparison_BTCUSDT_1m.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"Loading comparison data from {csv_path}...")
    comparison_df = pd.read_csv(csv_path)
    
    print(f"Found {len(comparison_df)} methods to compare:")
    for method in comparison_df['method']:
        print(f"  - {method}")
    
    # Output path
    output_dir = csv_path.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "detailed_comparison.png"
    
    print(f"\nCreating detailed comparison visualization...")
    
    # Create visualizer and plot
    viz = RegimeVisualizer()
    summary = viz.plot_detailed_comparison(comparison_df, output_path=output_path)
    
    print(f"\n{'='*60}")
    print("✓ VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output saved to: {output_path}")
    
    print(f"\nMethod Rankings (Best to Worst):")
    for i, row in enumerate(summary, 1):
        print(f"  {i}. {row[0]:<12} Avg Rank: {row[5]}")
    
    print(f"\n⭐ RECOMMENDED: {summary[0][0]}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()