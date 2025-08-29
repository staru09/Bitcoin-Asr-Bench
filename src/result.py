import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List
import argparse
import json

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ASRResultsPlotter:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.results = {}
        
    def add_result(self, model_name: str, wer: float, accuracy: float, precision: float, 
                   recall: float, f1_score: float, substitutions: int, deletions: int, 
                   insertions: int, total_words: int):
        self.results[model_name] = {
            'WER': wer,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Substitution Rate': (substitutions / total_words) * 100,
            'Deletion Rate': (deletions / total_words) * 100,
            'Insertion Rate': (insertions / total_words) * 100,
            'Total Errors': substitutions + deletions + insertions,
            'Substitutions': substitutions,
            'Deletions': deletions,
            'Insertions': insertions
        }
    
    def plot_wer_comparison(self, save_path: str = None):
        models = list(self.results.keys())
        wer_values = [self.results[model]['WER'] for model in models]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.bar(models, wer_values, 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        for bar, value in zip(bars, wer_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Word Error Rate (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('ASR Models', fontsize=12, fontweight='bold')
        ax.set_title('Word Error Rate (WER) Comparison Across ASR Models', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(wer_values) * 1.2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"WER comparison saved to: {save_path}")
        
        plt.show()
    
    def plot_accuracy_metrics(self, save_path: str = None):
        models = list(self.results.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(models))
        width = 0.2
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            bars = ax.bar(x + i * width, values, width, 
                         label=metric, color=colors[i], alpha=0.8, edgecolor='black')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('ASR Models', fontsize=12, fontweight='bold')
        ax.set_title('ASR Model Performance Metrics Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy metrics saved to: {save_path}")
        
        plt.show()
    
    def plot_error_breakdown(self, save_path: str = None):
        models = list(self.results.keys())
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(models))
        width = 0.25
        
        substitutions = [self.results[model]['Substitution Rate'] for model in models]
        deletions = [self.results[model]['Deletion Rate'] for model in models]
        insertions = [self.results[model]['Insertion Rate'] for model in models]
        
        bars1 = ax.bar(x - width, substitutions, width, label='Substitutions', 
                      color='#ff9999', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, deletions, width, label='Deletions', 
                      color='#66b3ff', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, insertions, width, label='Insertions', 
                      color='#99ff99', alpha=0.8, edgecolor='black')
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.1:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('ASR Models', fontsize=12, fontweight='bold')
        ax.set_title('Error Type Breakdown by ASR Model', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error breakdown saved to: {save_path}")
        
        plt.show()
    
    def plot_combined_dashboard(self, save_path: str = None):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(self.results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        wer_values = [self.results[model]['WER'] for model in models]
        bars1 = ax1.bar(models, wer_values, color=colors, alpha=0.8, edgecolor='black')
        for bar, value in zip(bars1, wer_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        ax1.set_title('Word Error Rate (WER)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('WER (%)')
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        f1_values = [self.results[model]['F1-Score'] for model in models]
        bars2 = ax2.bar(models, f1_values, color=colors, alpha=0.8, edgecolor='black')
        for bar, value in zip(bars2, f1_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax2.set_title('F1-Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1-Score (%)')
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        substitutions = [self.results[model]['Substitution Rate'] for model in models]
        deletions = [self.results[model]['Deletion Rate'] for model in models]
        insertions = [self.results[model]['Insertion Rate'] for model in models]
        
        ax3.bar(models, substitutions, label='Substitutions', color='#ff9999', alpha=0.8)
        ax3.bar(models, deletions, bottom=substitutions, label='Deletions', color='#66b3ff', alpha=0.8)
        ax3.bar(models, insertions, bottom=np.array(substitutions) + np.array(deletions), 
               label='Insertions', color='#99ff99', alpha=0.8)
        ax3.set_title('Error Types (Stacked)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Error Rate (%)')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        model_wer_pairs = [(model, self.results[model]['WER']) for model in models]
        model_wer_pairs.sort(key=lambda x: x[1])
        
        ranked_models = [pair[0] for pair in model_wer_pairs]
        ranked_wer = [pair[1] for pair in model_wer_pairs]
        
        rank_colors = ['green', 'orange', 'red', 'darkred'][:len(ranked_models)]
        
        bars4 = ax4.barh(ranked_models, ranked_wer, color=rank_colors, alpha=0.7)
        for bar, value in zip(bars4, ranked_wer):
            width = bar.get_width()
            ax4.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{value:.2f}%', ha='left', va='center', fontweight='bold')
        ax4.set_title('Model Ranking (by WER)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('WER (%)')
        ax4.grid(axis='x', alpha=0.3)
        
        plt.suptitle('ASR Model Performance Dashboard', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        
        plt.show()
    
    def generate_all_plots(self, prefix: str = "asr_results"):
        self.plot_wer_comparison(f"{prefix}_wer_comparison.png")
        self.plot_accuracy_metrics(f"{prefix}_accuracy_metrics.png")
        self.plot_error_breakdown(f"{prefix}_error_breakdown.png")
        self.plot_combined_dashboard(f"{prefix}_dashboard.png")
    
    def save_results_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {filepath}")

def main():
    
    plotter = ASRResultsPlotter(figsize=(12, 8))
    
    plotter.add_result(
        model_name="Canary",
        wer=16.60,
        accuracy=97.94,
        precision=85.80,
        recall=97.94,
        f1_score=91.47,
        substitutions=60,
        deletions=14,
        insertions=523,
        total_words=3597
    )
    
    plotter.add_result(
        model_name="Granite",
        wer=97.80,
        accuracy=77.93,
        precision=45.93,
        recall=77.93,
        f1_score=57.79,
        substitutions=576,
        deletions=218,
        insertions=2724,
        total_words=3597
    )
    
    plotter.add_result(
        model_name="Moshi",
        wer=4.31,
        accuracy=97.94,
        precision=96.65,
        recall=97.94,
        f1_score=97.29,
        substitutions=41,
        deletions=33,
        insertions=81,
        total_words=3597
    )
    
    plotter.add_result(
        model_name="Parakeet",
        wer=9.06,
        accuracy=98.86,
        precision=92.10,
        recall=98.86,
        f1_score=95.36,
        substitutions=20,
        deletions=21,
        insertions=285,
        total_words=3597
    )
    
    print("Generating ASR comparison plots...")
    plotter.generate_all_plots("asr_model_comparison")
    
    plotter.save_results_json("asr_results.json")
    
    print("\nASR Model Performance Summary:")
    print("=" * 50)
    for model in plotter.results:
        wer = plotter.results[model]['WER']
        f1 = plotter.results[model]['F1-Score']
        print(f"{model:<12} WER: {wer:6.2f}%  F1: {f1:5.1f}%")
    
    best_model = min(plotter.results.keys(), key=lambda x: plotter.results[x]['WER'])
    print(f"\nBest performing model: {best_model} (WER: {plotter.results[best_model]['WER']:.2f}%)")

if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Install with: pip install matplotlib seaborn numpy")
        exit(1)
    
    main()