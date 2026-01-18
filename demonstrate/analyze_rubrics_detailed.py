"""
Detailed SciCode Rubrics Analysis
Comparing individual rubric evaluations before and after framework fixes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import re

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Paths
rubrics_dir = Path('/home/v-tatruong/hal/agent-debug/rubrics_output/scicode')
output_dir = Path('/home/v-tatruong/hal/agent-debug/demonstrate')
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("DETAILED SCICODE RUBRICS ANALYSIS")
print("=" * 70)

# Load all rubric files
all_files = list(rubrics_dir.glob('*.csv'))
print(f"\nFound {len(all_files)} rubric files")

# Categorize files into before/after groups
# Before fixes: potato, tomato, kiwi, hal_generalist, scicode_tool_calling, scicode_zero_shot (UPLOAD files)
# After fixes: apple, honey

before_patterns = ['potato', 'tomato', 'kiwi', 'UPLOAD']
after_patterns = ['apple', 'honey']

before_files = []
after_files = []

for f in all_files:
    fname = f.name
    if any(p in fname for p in after_patterns):
        after_files.append(f)
    elif any(p in fname for p in before_patterns):
        before_files.append(f)

print(f"Before-fix files: {len(before_files)}")
print(f"After-fix files: {len(after_files)}")

# Load and combine data
def load_rubrics(files, label):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = f.name
            df['group'] = label
            # Extract model name from filename
            model_match = re.search(r'(gpt-4|gpt-5|o3|o4-mini|claude|deepseek|gemini)', f.name, re.IGNORECASE)
            df['model'] = model_match.group(1) if model_match else 'unknown'
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {f.name}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

before_df = load_rubrics(before_files, 'Before Fixes')
after_df = load_rubrics(after_files, 'After Fixes')

print(f"\nBefore-fix evaluations: {len(before_df)}")
print(f"After-fix evaluations: {len(after_df)}")

# Combine for analysis
all_df = pd.concat([before_df, after_df], ignore_index=True)

# ============================================================================
# Figure 1: Overall Benchmark Defect Rate Comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Calculate defect rates
before_defect_rate = before_df['grade'].mean() * 100 if len(before_df) > 0 else 0
after_defect_rate = after_df['grade'].mean() * 100 if len(after_df) > 0 else 0

# Bar chart
groups = ['Before Fixes', 'After Fixes']
defect_rates = [before_defect_rate, after_defect_rate]
colors = ['#e74c3c', '#27ae60']

bars = axes[0].bar(groups, defect_rates, color=colors, edgecolor='black', width=0.6)
axes[0].set_ylabel('Benchmark Defect Rate (%)', fontsize=12)
axes[0].set_title('Reduction in False Positive\nBenchmark Defect Detections', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, max(defect_rates) * 1.3 if max(defect_rates) > 0 else 100)

for bar, rate in zip(bars, defect_rates):
    axes[0].annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=14)

# Add improvement annotation
if before_defect_rate > after_defect_rate:
    reduction = before_defect_rate - after_defect_rate
    axes[0].annotate(f'↓ {reduction:.1f}% reduction',
                    xy=(0.5, max(defect_rates) * 0.7),
                    fontsize=16, fontweight='bold', color='#27ae60',
                    ha='center', transform=axes[0].transAxes)

# Pie chart showing composition
if len(before_df) > 0:
    before_defects = int(before_df['grade'].sum())
    before_ok = len(before_df) - before_defects
    sizes = [before_defects, before_ok]
    labels = [f'Defects ({before_defects})', f'Agent Issues ({before_ok})']
    colors_pie = ['#e74c3c', '#3498db']
    axes[1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
               startangle=90, wedgeprops={'edgecolor': 'black'})
    axes[1].set_title('Before Fixes:\nClassification of Failures', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig5_defect_rate_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: fig5_defect_rate_comparison.png")

# ============================================================================
# Figure 2: Task-level Defect Detection Heatmap
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Get unique tasks
all_tasks = sorted(all_df['task_id'].unique())

# Before fixes - aggregate by task
if len(before_df) > 0:
    before_task_rates = before_df.groupby('task_id')['grade'].mean().reindex(all_tasks).fillna(0)

    # Create bar chart
    x = np.arange(len(all_tasks))
    colors_before = ['#e74c3c' if r > 0.5 else '#f39c12' if r > 0 else '#27ae60' for r in before_task_rates]
    axes[0].bar(x, before_task_rates * 100, color=colors_before, edgecolor='black', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_tasks, rotation=90, fontsize=8)
    axes[0].set_ylabel('Defect Detection Rate (%)')
    axes[0].set_xlabel('Task ID')
    axes[0].set_title('BEFORE Fixes: Per-Task Benchmark Defect Rate', fontsize=13, fontweight='bold')
    axes[0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    axes[0].set_ylim(0, 105)

# After fixes - aggregate by task
if len(after_df) > 0:
    after_tasks = sorted(after_df['task_id'].unique())
    after_task_rates = after_df.groupby('task_id')['grade'].mean()

    x = np.arange(len(after_tasks))
    colors_after = ['#e74c3c' if r > 0.5 else '#f39c12' if r > 0 else '#27ae60' for r in after_task_rates]
    axes[1].bar(x, after_task_rates * 100, color=colors_after, edgecolor='black', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(after_tasks, rotation=90, fontsize=8)
    axes[1].set_ylabel('Defect Detection Rate (%)')
    axes[1].set_xlabel('Task ID')
    axes[1].set_title('AFTER Fixes: Per-Task Benchmark Defect Rate', fontsize=13, fontweight='bold')
    axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    axes[1].set_ylim(0, 105)

plt.tight_layout()
plt.savefig(output_dir / 'fig6_task_defect_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: fig6_task_defect_heatmap.png")

# ============================================================================
# Figure 3: Model-wise Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Get defect rate by model and group
model_comparison = all_df.groupby(['group', 'model'])['grade'].agg(['mean', 'count']).reset_index()
model_comparison.columns = ['group', 'model', 'defect_rate', 'count']
model_comparison['defect_rate'] *= 100

# Pivot for grouped bar chart
models = model_comparison['model'].unique()
before_rates = []
after_rates = []
for m in models:
    before_rate = model_comparison[(model_comparison['model'] == m) & (model_comparison['group'] == 'Before Fixes')]['defect_rate'].values
    after_rate = model_comparison[(model_comparison['model'] == m) & (model_comparison['group'] == 'After Fixes')]['defect_rate'].values
    before_rates.append(before_rate[0] if len(before_rate) > 0 else 0)
    after_rates.append(after_rate[0] if len(after_rate) > 0 else 0)

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, before_rates, width, label='Before Fixes', color='#e74c3c', edgecolor='black')
bars2 = ax.bar(x + width/2, after_rates, width, label='After Fixes', color='#27ae60', edgecolor='black')

ax.set_ylabel('Benchmark Defect Rate (%)')
ax.set_xlabel('Model')
ax.set_title('Benchmark Defect Rate by Model: Before vs After Fixes', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, max(max(before_rates), max(after_rates)) * 1.2 if max(before_rates + after_rates) > 0 else 100)

plt.tight_layout()
plt.savefig(output_dir / 'fig7_model_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: fig7_model_comparison.png")

# ============================================================================
# Figure 4: Specific Tasks - Before vs After Deep Dive
# ============================================================================
# Focus on tasks that were evaluated both before and after
common_tasks = set(before_df['task_id'].unique()) & set(after_df['task_id'].unique())
print(f"\nTasks evaluated both before and after: {sorted(common_tasks)}")

if len(common_tasks) > 0:
    fig, ax = plt.subplots(figsize=(14, 6))

    common_tasks = sorted(common_tasks)
    before_task_defects = before_df[before_df['task_id'].isin(common_tasks)].groupby('task_id')['grade'].mean()
    after_task_defects = after_df[after_df['task_id'].isin(common_tasks)].groupby('task_id')['grade'].mean()

    x = np.arange(len(common_tasks))
    width = 0.35

    bars1 = ax.bar(x - width/2, [before_task_defects.get(t, 0) * 100 for t in common_tasks],
                   width, label='Before Fixes', color='#e74c3c', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, [after_task_defects.get(t, 0) * 100 for t in common_tasks],
                   width, label='After Fixes', color='#27ae60', edgecolor='black', alpha=0.8)

    ax.set_ylabel('Benchmark Defect Rate (%)')
    ax.set_xlabel('Task ID')
    ax.set_title('Task-by-Task Comparison: Defect Detection Rate\n(Tasks Evaluated Both Before and After)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(common_tasks)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)

    # Add annotations for improvements
    for i, task in enumerate(common_tasks):
        before_rate = before_task_defects.get(task, 0) * 100
        after_rate = after_task_defects.get(task, 0) * 100
        if before_rate > after_rate:
            ax.annotate(f'↓{before_rate - after_rate:.0f}%',
                       xy=(i, max(before_rate, after_rate) + 5),
                       ha='center', fontsize=9, color='#27ae60', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig8_task_deep_dive.png', dpi=150, bbox_inches='tight')
    print("Saved: fig8_task_deep_dive.png")

# ============================================================================
# Figure 5: Timeline / Batch Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

# Group by source file pattern
batch_patterns = {
    'hal_generalist': 'HAL Generalist\n(Baseline)',
    'zero_shot': 'Zero Shot\n(Baseline)',
    'tool_calling.*UPLOAD': 'Tool Calling\n(Baseline)',
    'potato': 'Potato\n(Pre-fix)',
    'tomato': 'Tomato\n(Pre-fix)',
    'kiwi': 'Kiwi\n(Mid-fix)',
    'apple': 'Apple\n(Post-fix)',
    'honey': 'Honey\n(Post-fix)',
}

batch_stats = []
for pattern, label in batch_patterns.items():
    matching = all_df[all_df['source_file'].str.contains(pattern, case=False, regex=True)]
    if len(matching) > 0:
        batch_stats.append({
            'batch': label,
            'defect_rate': matching['grade'].mean() * 100,
            'count': len(matching),
            'defects': int(matching['grade'].sum())
        })

if batch_stats:
    batch_df = pd.DataFrame(batch_stats)

    # Color based on defect rate
    colors = ['#e74c3c' if r > 50 else '#f39c12' if r > 20 else '#27ae60' for r in batch_df['defect_rate']]

    bars = ax.bar(batch_df['batch'], batch_df['defect_rate'], color=colors, edgecolor='black')
    ax.set_ylabel('Benchmark Defect Rate (%)')
    ax.set_xlabel('Evaluation Batch')
    ax.set_title('Benchmark Defect Rate Across Evaluation Batches\n(Showing Progressive Improvement)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(batch_df['defect_rate']) * 1.2)

    # Add value labels
    for bar, rate, count in zip(bars, batch_df['defect_rate'], batch_df['count']):
        ax.annotate(f'{rate:.1f}%\n(n={count})',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig9_batch_timeline.png', dpi=150, bbox_inches='tight')
    print("Saved: fig9_batch_timeline.png")

# ============================================================================
# Figure 6: Summary Dashboard
# ============================================================================
fig = plt.figure(figsize=(16, 10))

# Create grid
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Panel 1: Overall metrics
ax1 = fig.add_subplot(gs[0, 0])
metrics = ['Total Evals', 'Before Defects', 'After Defects']
values = [len(all_df), int(before_df['grade'].sum()) if len(before_df) > 0 else 0,
          int(after_df['grade'].sum()) if len(after_df) > 0 else 0]
colors = ['#3498db', '#e74c3c', '#27ae60']
bars = ax1.bar(metrics, values, color=colors, edgecolor='black')
ax1.set_title('Evaluation Counts', fontweight='bold')
for bar, val in zip(bars, values):
    ax1.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')

# Panel 2: Defect rate comparison
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(['Before', 'After'], [before_defect_rate, after_defect_rate],
        color=['#e74c3c', '#27ae60'], edgecolor='black')
ax2.set_title('Defect Rate (%)', fontweight='bold')
ax2.set_ylim(0, max(before_defect_rate, after_defect_rate) * 1.3 if max(before_defect_rate, after_defect_rate) > 0 else 100)

# Panel 3: Improvement percentage
ax3 = fig.add_subplot(gs[0, 2])
if before_defect_rate > 0:
    improvement_pct = ((before_defect_rate - after_defect_rate) / before_defect_rate) * 100
else:
    improvement_pct = 0
ax3.pie([improvement_pct, 100-improvement_pct] if improvement_pct > 0 else [100, 0],
        labels=['Improved', 'Remaining'], colors=['#27ae60', '#bdc3c7'],
        autopct='%1.0f%%', startangle=90)
ax3.set_title('Improvement Rate', fontweight='bold')

# Panel 4: Top improved tasks
ax4 = fig.add_subplot(gs[1, :2])
if len(common_tasks) > 0:
    improvements = []
    for task in common_tasks:
        before_rate = before_task_defects.get(task, 0) * 100
        after_rate = after_task_defects.get(task, 0) * 100
        improvements.append({'task': task, 'improvement': before_rate - after_rate,
                           'before': before_rate, 'after': after_rate})
    imp_df = pd.DataFrame(improvements).sort_values('improvement', ascending=False)

    x = np.arange(len(imp_df))
    ax4.barh(x, imp_df['improvement'], color='#27ae60', edgecolor='black')
    ax4.set_yticks(x)
    ax4.set_yticklabels([f"Task {t}" for t in imp_df['task']])
    ax4.set_xlabel('Improvement in Defect Rate (%)')
    ax4.set_title('Task-wise Improvement in Defect Classification', fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Panel 5: Legend / Key fixes
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')
fixes_text = """
Key Fixes Applied:

1. scipy.integrate.simps shim
2. MatMult (@) operator patch
3. numpy.random authorization
4. heapq import authorization
5. code_block_tags="markdown"
6. Complex dtype guidance
7. Rubric clarification

Result: {:.1f}% → {:.1f}%
Defect Rate Reduction
""".format(before_defect_rate, after_defect_rate)
ax5.text(0.1, 0.5, fixes_text, transform=ax5.transAxes, fontsize=11,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('SciCode Benchmark Quality Dashboard', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'fig10_dashboard.png', dpi=150, bbox_inches='tight')
print("Saved: fig10_dashboard.png")

# ============================================================================
# Print Summary Statistics
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(f"\nTotal evaluations analyzed: {len(all_df)}")
print(f"  - Before fixes: {len(before_df)}")
print(f"  - After fixes: {len(after_df)}")

print(f"\nBenchmark Defect Rate:")
print(f"  - Before fixes: {before_defect_rate:.1f}% ({int(before_df['grade'].sum())} defects)")
print(f"  - After fixes: {after_defect_rate:.1f}% ({int(after_df['grade'].sum())} defects)")
print(f"  - Reduction: {before_defect_rate - after_defect_rate:.1f} percentage points")

if before_defect_rate > 0:
    print(f"  - Relative improvement: {((before_defect_rate - after_defect_rate) / before_defect_rate) * 100:.1f}%")

print(f"\nTasks evaluated in both phases: {len(common_tasks)}")
if len(common_tasks) > 0:
    improved_tasks = sum(1 for t in common_tasks
                        if before_task_defects.get(t, 0) > after_task_defects.get(t, 0))
    print(f"  - Tasks with improved classification: {improved_tasks}")

# Save detailed report
report = f"""# Detailed Rubrics Analysis Report

## Overview
- Total evaluations: {len(all_df)}
- Before fixes: {len(before_df)} evaluations
- After fixes: {len(after_df)} evaluations

## Benchmark Defect Rates
| Phase | Defect Rate | Count |
|-------|-------------|-------|
| Before Fixes | {before_defect_rate:.1f}% | {int(before_df['grade'].sum())} |
| After Fixes | {after_defect_rate:.1f}% | {int(after_df['grade'].sum())} |
| **Reduction** | **{before_defect_rate - after_defect_rate:.1f}%** | **{int(before_df['grade'].sum()) - int(after_df['grade'].sum())}** |

## Common Tasks Analysis
Tasks evaluated both before and after: {sorted(common_tasks)}

## Files Analyzed
### Before Fixes
{chr(10).join(['- ' + f.name for f in before_files])}

### After Fixes
{chr(10).join(['- ' + f.name for f in after_files])}

## Generated Figures
1. fig5_defect_rate_comparison.png - Overall defect rate comparison
2. fig6_task_defect_heatmap.png - Per-task defect detection rates
3. fig7_model_comparison.png - Model-wise comparison
4. fig8_task_deep_dive.png - Task-by-task before/after comparison
5. fig9_batch_timeline.png - Timeline across evaluation batches
6. fig10_dashboard.png - Summary dashboard
"""

with open(output_dir / 'detailed_analysis_report.md', 'w') as f:
    f.write(report)
print("\nSaved: detailed_analysis_report.md")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
