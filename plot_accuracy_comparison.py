"""
Script to compare accuracies across multiple experiment folders.
Parses log files to extract best accuracies and creates comparison plots.
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple

def parse_log_file(log_file_path: str) -> float:
    """
    Parse a log file to extract the best accuracy.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        Best accuracy as float, or None if not found
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for patterns like "Best Accuracy: 85.67%" or "Best accuracy: 85.67%"
        patterns = [
            r'Best Accuracy:\s*(\d+\.?\d*)%',
            r'Best accuracy:\s*(\d+\.?\d*)%',
            r'🔥 Best model saved with Test Accuracy:\s*(\d+\.?\d*)%',
            r'Best model saved with Test Accuracy:\s*(\d+\.?\d*)%',
            r'Best Test Accuracy:\s*(\d+\.?\d*)%',
            r'Best test accuracy:\s*(\d+\.?\d*)%'
        ]
        
        best_acc = None
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Convert to float and find the highest value
                accuracies = [float(match) for match in matches]
                if accuracies:
                    best_acc = max(accuracies)
                    break
        
        return best_acc
        
    except Exception as e:
        print(f"Error parsing {log_file_path}: {e}")
        return None

def find_log_files(folder_path: str) -> List[str]:
    """
    Find all log files in a folder and its subfolders.
    
    Args:
        folder_path: Path to the main folder
        
    Returns:
        List of paths to log files
    """
    log_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return log_files
    
    # Look for log files in the main folder and subfolders
    for log_file in folder.rglob("logfile.txt"):
        log_files.append(str(log_file))
    
    # Also look for other common log file names
    for log_file in folder.rglob("*.log"):
        log_files.append(str(log_file))
    
    return log_files

def extract_accuracies_from_folder(folder_path: str) -> Tuple[List[float], float, float]:
    """
    Extract all best accuracies from a folder and calculate statistics.
    
    Args:
        folder_path: Path to the folder
        
    Returns:
        Tuple of (list of accuracies, mean, std)
    """
    log_files = find_log_files(folder_path)
    
    if not log_files:
        print(f"No log files found in {folder_path}")
        return [], 0.0, 0.0
    
    accuracies = []
    for log_file in log_files:
        acc = parse_log_file(log_file)
        if acc is not None:
            accuracies.append(acc)
    
    if not accuracies:
        print(f"No valid accuracies found in {folder_path}")
        return [], 0.0, 0.0
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"Folder: {folder_path}")
    print(f"  Found {len(accuracies)} valid accuracies")
    print(f"  Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Best: {max(accuracies):.2f}%")
    print(f"  Worst: {min(accuracies):.2f}%")
    print()
    
    return accuracies, mean_acc, std_acc

def create_comparison_plot(folder_paths: List[str], output_path: str = "accuracy_comparison.png", folder_name_mapping: Dict[str, str] = None):
    """
    Create a comparison plot of accuracies across folders.
    
    Args:
        folder_paths: List of folder paths to compare
        output_path: Path to save the output plot
        folder_name_mapping: Dictionary to map folder names to display names
    """
    folder_names = [Path(folder).name for folder in folder_paths]
    
    # Apply folder name mapping if provided
    if folder_name_mapping:
        display_names = []
        for folder_name in folder_names:
            display_name = folder_name_mapping.get(folder_name, folder_name)
            display_names.append(display_name)
        folder_names = display_names
    
    # Extract data from each folder
    all_data = []
    means = []
    stds = []
    bests = []
    
    for folder_path in folder_paths:
        accuracies, mean_acc, std_acc = extract_accuracies_from_folder(folder_path)
        if accuracies:
            all_data.append(accuracies)
            means.append(mean_acc)
            stds.append(std_acc)
            bests.append(max(accuracies))
        else:
            all_data.append([])
            means.append(0.0)
            stds.append(0.0)
            bests.append(0.0)
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Normal box plots
    bp = ax1.boxplot(all_data, labels=folder_names, patch_artist=True)
    
    # Color the box plots
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax1.set_title('Distribution of Accuracies Across Training Sample Percentages')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xlabel('Percent of Training Samples')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar plot of means and best accuracies
    x = np.arange(len(folder_names))
    width = 0.35
    
    # Plot mean accuracies with error bars
    bars1 = ax2.bar(x - width/2, means, width, label='Mean Accuracy', 
                    yerr=stds, capsize=5, alpha=0.8, color='skyblue')
    
    # Plot best accuracies
    bars2 = ax2.bar(x + width/2, bests, width, label='Best Accuracy', 
                    alpha=0.8, color='orange')
    
    ax2.set_title('Mean vs Best Accuracies')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xlabel('Percent of Training Samples')
    ax2.set_xticks(x)
    ax2.set_xticklabels(folder_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Line plot connecting best accuracies
    x_positions = np.arange(len(folder_names))
    ax3.plot(x_positions, bests, 'o-', linewidth=3, markersize=8, color='red', 
             label='Best Accuracy Trend', markerfacecolor='red', markeredgecolor='darkred')
    
    # Add value labels on the line plot
    for i, (x, y) in enumerate(zip(x_positions, bests)):
        ax3.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
    
    ax3.set_title('Best Accuracy Trend Across Training Sample Percentages')
    ax3.set_ylabel('Best Accuracy (%)')
    ax3.set_xlabel('Percent of Training Samples')
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(folder_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Set y-axis limits to show the trend better
    if bests:
        y_min = min(bests) - 2
        y_max = max(bests) + 2
        ax3.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_path}")
    plt.show()
    
    # Create summary table
    summary_data = []
    for i, folder_name in enumerate(folder_names):
        if all_data[i]:
            summary_data.append({
                'Folder': folder_name,
                'Num_Experiments': len(all_data[i]),
                'Mean_Accuracy': f"{means[i]:.2f}%",
                'Std_Accuracy': f"{stds[i]:.2f}%",
                'Best_Accuracy': f"{bests[i]:.2f}%",
                'Worst_Accuracy': f"{min(all_data[i]):.2f}%"
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        print(df.to_string(index=False))
        
        # Save summary to CSV
        csv_path = output_path.replace('.png', '_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSummary saved as: {csv_path}")

# Add a simple function to run with predefined folders
def run_with_predefined_folders():
    """Run the comparison with the predefined k-fold testing folders."""
    folder_paths = [
        "k3fold_testing",
        "k4fold_testing", 
        "final_k5fold_testing",
        "k10fold_testing"
    ]
    
    # Define folder name mappings
    folder_name_mapping = {
        'k3fold_testing': '66%',
        'k4fold_testing': '75%', 
        'final_k5fold_testing': '80%',
        'k5fold_testing': '80%',
        'k10fold_testing': '90%'
    }
    
    print("="*80)
    print("ACCURACY COMPARISON TOOL - PREDEFINED FOLDERS")
    print("="*80)
    print("Comparing k-fold testing folders:")
    for folder in folder_paths:
        print(f"  - {folder} -> {folder_name_mapping.get(folder, folder)}")
    print()
    
    create_comparison_plot(folder_paths, "kfold_accuracy_comparison.png", folder_name_mapping)

def main():
    parser = argparse.ArgumentParser(description='Compare accuracies across multiple experiment folders')
    parser.add_argument('folders', nargs='+', help='Paths to folders containing experiment results')
    parser.add_argument('--output', '-o', default='accuracy_comparison.png', 
                       help='Output path for the plot (default: accuracy_comparison.png)')
    
    args = parser.parse_args()
    
    # Define folder name mappings
    folder_name_mapping = {
        'k3fold_testing': '66%',
        'k4fold_testing': '75%', 
        'final_k5fold_testing': '80%',
        'k10fold_testing': '90%'
    }
    
    print("="*80)
    print("ACCURACY COMPARISON TOOL")
    print("="*80)
    print(f"Comparing {len(args.folders)} folders:")
    for folder in args.folders:
        print(f"  - {folder}")
    print()
    
    create_comparison_plot(args.folders, args.output, folder_name_mapping)

if __name__ == "__main__":
    # Check if command line arguments are provided
    import sys
    if len(sys.argv) > 1:
        # Use command line arguments
        main()
    else:
        # Use predefined folders
        run_with_predefined_folders() 