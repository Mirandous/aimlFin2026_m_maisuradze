#!/usr/bin/env python3
"""
Enhanced DDoS Attack Visualization
Creates publication-quality 8-panel comprehensive analysis dashboard
Author: m_maisuradze_47631
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import re
import sys

sns.set_style("whitegrid")

def load_and_process_logs(filepath):
    """Load and process log file"""
    log_pattern = r'(\d+\.\d+\.\d+\.\d+)\s+-\s+-\s+\[([^\]]+)\]\s+"(\w+)\s+([^\s]+)\s+HTTP/[^"]+"\s+(\d+)\s+(\d+)\s+"([^"]*)"\s+"([^"]*)"\s+(\d+)'
    
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(log_pattern, line.strip())
            if match:
                ip, timestamp, method, endpoint, status, size, referrer, user_agent, response_time = match.groups()
                entries.append({
                    'ip': ip,
                    'timestamp': pd.to_datetime(timestamp),
                    'method': method,
                    'status': int(status),
                    'response_time': int(response_time)
                })
    
    df = pd.DataFrame(entries)
    
    # Create time window aggregations
    df_agg = df.copy()
    df_agg['time_window'] = df_agg['timestamp'].dt.floor('1min')
    
    agg_data = df_agg.groupby('time_window').agg({
        'ip': ['count', 'nunique'],
        'response_time': 'mean',
        'status': lambda x: (x >= 400).sum()
    })
    agg_data.columns = ['total_requests', 'unique_ips', 'avg_response_time', 'error_count']
    agg_data = agg_data.reset_index()
    
    return df, agg_data

def detect_attacks(agg_data, threshold=2.0):
    """Detect attack periods using regression"""
    agg_data['minutes_elapsed'] = (agg_data['time_window'] - agg_data['time_window'].min()).dt.total_seconds() / 60
    
    X = agg_data[['minutes_elapsed']].values
    y = agg_data['total_requests'].values
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    agg_data['predicted'] = model.predict(X_poly)
    agg_data['residual'] = agg_data['total_requests'] - agg_data['predicted']
    
    mean_res = agg_data['residual'].mean()
    std_res = agg_data['residual'].std()
    agg_data['z_score'] = (agg_data['residual'] - mean_res) / std_res
    agg_data['is_anomaly'] = np.abs(agg_data['z_score']) > threshold
    
    # Find attack intervals
    attack_periods = agg_data[agg_data['is_anomaly']]
    
    return agg_data, model, attack_periods

def create_enhanced_visualization(df, agg_data, attack_periods, output_path='enhanced_ddos_analysis.png'):
    """Create comprehensive 8-panel dashboard"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Calculate attack time ranges for shading
    if len(attack_periods) > 0:
        attack_start = attack_periods['time_window'].min()
        attack_end = attack_periods['time_window'].max()
        baseline_mean = agg_data['predicted'].mean()
        peak_traffic = attack_periods['total_requests'].max()
        max_z = attack_periods['z_score'].abs().max()
    
    # 1. Main Timeline with Regression (Large, top)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(agg_data['time_window'], agg_data['total_requests'], 
             linewidth=2, label='Actual Traffic', color='#2E86AB')
    ax1.plot(agg_data['time_window'], agg_data['predicted'], 
             '--', linewidth=2, label='Regression Baseline (2nd degree polynomial)', 
             color='#A23B72', alpha=0.8)
    
    # Shade attack periods
    if len(attack_periods) > 0:
        for _, period in attack_periods.iterrows():
            ax1.axvspan(period['time_window'], 
                       period['time_window'] + pd.Timedelta(minutes=1),
                       alpha=0.2, color='red')
    
    # Add annotation for detected attack
    if len(attack_periods) > 0:
        severity_text = "Critical" if max_z > 4 else "High" if max_z > 3 else "Moderate"
        wave_info = f"Wave #1: {severity_text}\n{int(peak_traffic):,} req/min\nZ-score: {max_z:.2f}"
        
        ax1.annotate(wave_info,
                    xy=(attack_end, peak_traffic),
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='#FFB5A7', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                    fontsize=9, fontweight='bold')
    
    ax1.axhline(y=baseline_mean, color='gray', linestyle=':', linewidth=1.5, 
                label=f'Baseline: {baseline_mean:.0f} req/min', alpha=0.7)
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Requests per Minute', fontsize=11)
    ax1.set_title('ENHANCED DDoS ATTACK ANALYSIS - HIGH ACCURACY VISUALIZATION\nLog File: m_maisuradze_47631_server.log  |  Analysis Date: February 13, 2026', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 2. Regression Residuals Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ['red' if x else 'green' for x in agg_data['is_anomaly']]
    ax2.bar(range(len(agg_data)), agg_data['residual'], color=colors, alpha=0.7, width=0.8)
    ax2.axhline(y=2*agg_data['residual'].std(), color='red', linestyle='--', 
                linewidth=2, label='±2σ threshold')
    ax2.axhline(y=-2*agg_data['residual'].std(), color='red', linestyle='--', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Time Period (Actual - Predicted)', fontsize=10)
    ax2.set_ylabel('Residual (Actual - Predicted)', fontsize=10)
    ax2.set_title('Regression Residuals Distribution', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Statistical Anomaly Detection (Z-Score Analysis)
    ax3 = fig.add_subplot(gs[1, 1])
    colors_z = ['red' if x else 'blue' for x in agg_data['is_anomaly']]
    ax3.scatter(agg_data['minutes_elapsed'], agg_data['z_score'], 
               c=colors_z, alpha=0.6, s=50)
    ax3.axhline(y=2.0, color='red', linestyle='--', linewidth=2, 
                label='Anomaly Threshold (±2.00)')
    ax3.axhline(y=-2.0, color='red', linestyle='--', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Annotate critical points
    if len(attack_periods) > 0:
        max_z_idx = agg_data['z_score'].abs().idxmax()
        ax3.scatter(agg_data.loc[max_z_idx, 'minutes_elapsed'], 
                   agg_data.loc[max_z_idx, 'z_score'],
                   s=200, facecolors='none', edgecolors='red', linewidths=2)
        ax3.annotate(f'Max: {agg_data.loc[max_z_idx, "z_score"]:.2f}',
                    xy=(agg_data.loc[max_z_idx, 'minutes_elapsed'], 
                        agg_data.loc[max_z_idx, 'z_score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='red', fontweight='bold')
    
    ax3.set_xlabel('Time (Minutes)', fontsize=10)
    ax3.set_ylabel('Z-Score (Standard Deviations)', fontsize=10)
    ax3.set_title('Statistical Anomaly Detection (Z-Score Analysis)', 
                 fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Traffic Source Distribution & Error Rate
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Separate normal vs attack traffic
    if len(attack_periods) > 0:
        attack_times = attack_periods['time_window'].values
        df['is_attack'] = df['timestamp'].dt.floor('1min').isin(attack_times)
        
        normal_ips = df[~df['is_attack']]['ip'].nunique()
        attack_ips = df[df['is_attack']]['ip'].nunique()
        
        # Calculate error rates
        normal_errors = df[~df['is_attack']]['status'].apply(lambda x: 1 if x >= 400 else 0).sum()
        normal_total = len(df[~df['is_attack']])
        attack_errors = df[df['is_attack']]['status'].apply(lambda x: 1 if x >= 400 else 0).sum()
        attack_total = len(df[df['is_attack']])
        
        normal_error_rate = (normal_errors / normal_total * 100) if normal_total > 0 else 0
        attack_error_rate = (attack_errors / attack_total * 100) if attack_total > 0 else 0
    else:
        normal_ips = df['ip'].nunique()
        attack_ips = 0
        normal_error_rate = df['status'].apply(lambda x: 1 if x >= 400 else 0).mean() * 100
        attack_error_rate = 0
    
    # Create grouped bar chart
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, [normal_ips, attack_ips], width, 
                   label='Unique IPs', color='#2E86AB', alpha=0.8)
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, [normal_error_rate, attack_error_rate], width,
                        label='Error Rate (%)', color='#E63946', alpha=0.8)
    
    ax4.set_ylabel('Unique IP Addresses', fontsize=10, color='#2E86AB')
    ax4_twin.set_ylabel('HTTP Error Rate (%)', fontsize=10, color='#E63946')
    ax4.set_title('Traffic Source Distribution & Error Rate', fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Normal Traffic', 'Attack Traffic'])
    ax4.tick_params(axis='y', labelcolor='#2E86AB')
    ax4_twin.tick_params(axis='y', labelcolor='#E63946')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax4_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Server Response Time During Attack
    ax5 = fig.add_subplot(gs[1, 3])
    
    # Plot response time with confidence interval
    response_time_series = agg_data.set_index('time_window')['avg_response_time']
    ax5.plot(response_time_series.index, response_time_series.values, 
            color='#7209B7', linewidth=2, label='Mean Response Time')
    
    # Calculate and plot std deviation envelope
    df['time_window_resp'] = df['timestamp'].dt.floor('1min')
    response_std = df.groupby('time_window_resp')['response_time'].std()
    
    ax5.fill_between(response_time_series.index, 
                     response_time_series - response_std, 
                     response_time_series + response_std,
                     alpha=0.2, color='#7209B7', label='±1 Std Dev')
    
    # Shade attack period
    if len(attack_periods) > 0:
        ax5.axvspan(attack_start, attack_end + pd.Timedelta(minutes=1),
                   alpha=0.15, color='red', label='Attack Period')
    
    ax5.set_xlabel('Time', fontsize=10)
    ax5.set_ylabel('Response Time (ms)', fontsize=10)
    ax5.set_title('Server Response Time During Attack', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 6. Traffic Distribution: Normal vs Attack
    ax6 = fig.add_subplot(gs[2, :2])
    
    if len(attack_periods) > 0:
        normal_traffic = agg_data[~agg_data['is_anomaly']]['total_requests']
        attack_traffic = agg_data[agg_data['is_anomaly']]['total_requests']
        
        # Create histogram
        bins = np.linspace(0, max(agg_data['total_requests'].max(), 15000), 30)
        ax6.hist(normal_traffic, bins=bins, alpha=0.7, label='Normal Traffic', 
                color='#4CAF50', edgecolor='black')
        ax6.hist(attack_traffic, bins=bins, alpha=0.7, label='Attack Traffic', 
                color='#F44336', edgecolor='black')
        
        # Add statistics
        ax6.axvline(normal_traffic.mean(), color='#4CAF50', linestyle='--', 
                   linewidth=2, label=f'Normal Mean: {normal_traffic.mean():.0f}')
        ax6.axvline(attack_traffic.mean(), color='#F44336', linestyle='--', 
                   linewidth=2, label=f'Attack Mean: {attack_traffic.mean():.0f}')
    
    ax6.set_xlabel('Requests per Minute', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Traffic Distribution: Normal vs Attack', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. HTTP Error Rate Over Time
    ax7 = fig.add_subplot(gs[2, 2:])
    
    # Calculate error rate percentage
    agg_data['error_rate'] = (agg_data['error_count'] / agg_data['total_requests'] * 100)
    
    ax7.plot(agg_data['time_window'], agg_data['error_rate'], 
            linewidth=2, color='#E63946', marker='o', markersize=4)
    ax7.axhline(y=50, color='red', linestyle='--', linewidth=2, 
                label='Critical Threshold (50%)', alpha=0.7)
    
    # Shade attack period
    if len(attack_periods) > 0:
        ax7.axvspan(attack_start, attack_end + pd.Timedelta(minutes=1),
                   alpha=0.15, color='red', label='Attack Period')
    
    ax7.set_xlabel('Time', fontsize=10)
    ax7.set_ylabel('Error Rate (%)', fontsize=10)
    ax7.set_title('HTTP Error Rate Over Time', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 8. Attack Summary Box
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    # Create summary text
    if len(attack_periods) > 0:
        first_attack_error_rate = (attack_periods.iloc[0]['error_count'] / attack_periods.iloc[0]['total_requests'] * 100) if attack_periods.iloc[0]['total_requests'] > 0 else 0
        summary_text = f"""
MULTI-WAVE DDoS ATTACK TIMELINE

ATTACK WAVE #1 (Moderate Intensity)
├─ Time: {attack_periods.iloc[0]['time_window'].strftime('%H:%M:%S')} - {attack_periods.iloc[min(1, len(attack_periods)-1)]['time_window'].strftime('%H:%M:%S')} ({attack_periods.iloc[0]['time_window'].strftime('%Y-%m-%d')})
├─ Peak Traffic: {int(attack_periods.iloc[0]['total_requests']):,} requests/min
├─ Baseline: {int(baseline_mean):,} requests/min
├─ Amplification: {attack_periods.iloc[0]['total_requests']/baseline_mean:.1f}x (±{((attack_periods.iloc[0]['total_requests']/baseline_mean - 1) * 100):.0f}% above normal)
├─ Unique IPs: {attack_periods.iloc[0]['unique_ips']:.0f}  →  {agg_data[~agg_data['is_anomaly']]['unique_ips'].mean():.0f}
├─ HTTP Errors: {int(attack_periods.iloc[0]['error_count']):,}  →  {int(first_attack_error_rate)}% error rate
└─ Error Rate: ~{first_attack_error_rate:.0f}%

OVERALL ATTACK CHARACTERISTICS
• Total Duration: {len(attack_periods)} minutes (with gaps)
• Total Requests: {attack_periods['total_requests'].sum():,}
• Attack Severity: CRITICAL ({max_z:.0f}/11 score)
• Pattern: {'Single sustained burst' if len(attack_periods) <= 2 else 'Multi-wave coordinated botnet assault'}
• Server Impact: {attack_error_rate:.0f}% error rate, complete overload
        """
    else:
        summary_text = "✓ No DDoS attacks detected in the analyzed period"
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nEnhanced visualization saved to: {output_path}")
    print(f"Resolution: 300 DPI (publication quality)")

def main():
    """Main execution"""
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = 'm_maisuradze_47631_server.log'
    
    print("Creating Enhanced DDoS Attack Visualization...\n")
    print(f"Loading: {log_file}")
    
    # Load and process data
    df, agg_data = load_and_process_logs(log_file)
    print(f"Processed {len(df):,} log entries into {len(agg_data)} time windows")
    
    # Detect attacks
    agg_data, model, attack_periods = detect_attacks(agg_data)
    print(f"Detected {len(attack_periods)} anomalous periods")
    
    # Create visualization
    create_enhanced_visualization(df, agg_data, attack_periods)
    
    print("\nEnhanced visualization complete!")

if __name__ == "__main__":
    main()
