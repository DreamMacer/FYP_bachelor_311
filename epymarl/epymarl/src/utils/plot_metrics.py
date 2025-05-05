import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_training_metrics(csv_file, output_dir=None):
    """
    Visualize training metrics
    Args:
        csv_file: Path to metrics.csv file
        output_dir: Directory to save the plot, if None the plot will be displayed
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # 1. Plot metrics in 0-1 range
    ax1.plot(df['episode'], df['completion_rate'], 'g-', label='Completion Rate')
    ax1.plot(df['episode'], df['truncated_rate'], 'r-', label='Truncation Rate')
    ax1.plot(df['episode'], df['average_battery'], 'b-', label='Average Battery')
    ax1.plot(df['episode'], df['task_success'].astype(float), 'y-', label='Task Success')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Rate')
    ax1.set_title('Ratio Metrics (0-1 Range)')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(-0.1, 1.1)
    
    # 2. Plot simulation time
    ax2.plot(df['episode'], df['simulation_time'], 'm-')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Simulation Time')
    ax2.grid(True)
    
    # 3. Plot mean reward
    ax3.plot(df['episode'], df['mean_reward'], 'c-')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.set_title('Mean Reward')
    ax3.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'training_metrics.png')
        print(f"Metrics plot saved to: {output_dir / 'training_metrics.png'}")
    else:
        plt.show()
    
    plt.close()

def plot_metrics_with_std(csv_file, window_size=10, output_dir=None):
    """
    Plot metrics with moving average and standard deviation
    Args:
        csv_file: Path to metrics.csv file
        window_size: Size of the moving window
        output_dir: Directory to save the plot, if None the plot will be displayed
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f'Training Metrics ({window_size}-Episode Moving Average)', fontsize=16)
    
    def rolling_stats(data):
        rolling_mean = data.rolling(window=window_size, min_periods=1).mean()
        rolling_std = data.rolling(window=window_size, min_periods=1).std()
        return rolling_mean, rolling_std
    
    # 1. Plot metrics in 0-1 range
    metrics_01 = ['completion_rate', 'truncated_rate', 'average_battery', 'task_success']
    colors = ['g', 'r', 'b', 'y']
    labels = ['Completion Rate', 'Truncation Rate', 'Average Battery', 'Task Success']
    
    for metric, color, label in zip(metrics_01, colors, labels):
        mean, std = rolling_stats(df[metric])
        ax1.plot(df['episode'], mean, color=color, label=label)
        ax1.fill_between(df['episode'], mean-std, mean+std, color=color, alpha=0.2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Rate')
    ax1.set_title('Ratio Metrics (0-1 Range)')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(-0.1, 1.1)
    
    # 2. Plot simulation time
    mean, std = rolling_stats(df['simulation_time'])
    ax2.plot(df['episode'], mean, 'm-')
    ax2.fill_between(df['episode'], mean-std, mean+std, color='m', alpha=0.2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Simulation Time')
    ax2.grid(True)
    
    # 3. Plot mean reward
    mean, std = rolling_stats(df['mean_reward'])
    ax3.plot(df['episode'], mean, 'c-')
    ax3.fill_between(df['episode'], mean-std, mean+std, color='c', alpha=0.2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.set_title('Mean Reward')
    ax3.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'training_metrics_smooth_{window_size}.png')
        print(f"Smoothed metrics plot saved to: {output_dir / f'training_metrics_smooth_{window_size}.png'}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    # Example usage
    csv_file = "path/to/your/metrics.csv"
    output_dir = "path/to/output/dir"
    
    # Plot raw data
    plot_training_metrics(csv_file, output_dir)
    
    # Plot smoothed data (10-episode moving window)
    plot_metrics_with_std(csv_file, window_size=10, output_dir=output_dir) 