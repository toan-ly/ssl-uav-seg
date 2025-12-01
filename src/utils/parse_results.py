"""
Parse results from the provided table and create visualizations for the research poster.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = Path('analysis/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# UAVid class names
UAVID_CLS = [
    'Model',
    'Clutter',
    'Building',
    'Road',
    'Tree',
    'Vegetation',
    'Moving Car',
    'Static Car',
    'Human',
    'mIoU',
    'F1',
    'OA'
]

# Results data from the table
results_data = {
    'Model': [
        'DeepLabV3 + SSL',
        'DeepLabV3 + ImageNet',
        'UNet + SSL',
        'UNet++ + SSL',
        'UNet + ImageNet',
        'UNet++ + ImageNet',
        'UNet + RandomInit'
    ],
    'Clutter': [66.02, 65.45, 64.96, 63.81, 61.98, 58.57, 46.25],
    'Building': [86.49, 86.00, 85.56, 84.17, 83.76, 80.03, 66.39],
    'Road': [78.84, 78.86, 77.47, 77.14, 74.86, 71.56, 62.80],
    'Tree': [78.83, 78.25, 78.50, 78.42, 76.41, 76.28, 59.53],
    'Vegetation': [62.69, 60.11, 61.57, 60.87, 58.79, 59.17, 36.05],
    'Moving Car': [70.49, 68.33, 66.72, 69.13, 66.22, 63.51, 42.48],
    'Static Car': [48.82, 51.09, 48.72, 48.20, 46.53, 45.00, 20.01],
    'Human': [28.78, 26.71, 29.68, 30.19, 25.47, 28.13, 0.0],
    'mIoU': [65.12, 64.35, 64.15, 63.99, 61.75, 60.28, 41.69],
    'F1': [77.34, 76.70, 76.72, 76.64, 74.72, 74.84, 62.97],
    'OA': [85.53, 85.06, 84.94, 84.49, 83.22, 82.00, 69.96]
}

order = [
    'DeepLabV3 + SSL',
    'DeepLabV3 + ImageNet',
    'UNet + SSL',
    'UNet++ + SSL',
    'UNet + ImageNet',
    'UNet++ + ImageNet',
    'UNet + RandomInit'
]
# Sort results_data according to the specified order
results_data = {key: [results_data[key][order.index(model)] for model in order]
                for key in results_data}

df = pd.DataFrame(results_data)

# Save to CSV
df.to_csv(OUTPUT_DIR / 'results_table.csv', index=False)
print(f"Saved results table to {OUTPUT_DIR / 'results_table.csv'}")

# ============================================================================
# Figure 1: mIoU Comparison Bar Chart
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
models = df['Model'].values
miou = df['mIoU'].values

# Color coding: SSL models in blue, ImageNet in orange, Random in red
colors = ['#3498db', '#e74c3c', '#3498db', '#3498db', '#e74c3c', '#e74c3c', '#95a5a6']

bars = ax.barh(models, miou, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, miou)):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Mean IoU (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Model Configuration', fontsize=14, fontweight='bold')
ax.set_title('Mean IoU Comparison Across Model Configurations',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0, 70)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='black', label='SSL Pretraining'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='ImageNet Pretraining'),
    Patch(facecolor='#95a5a6', edgecolor='black', label='Random Init')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'miou_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'miou_comparison.pdf', bbox_inches='tight')
print(f"Saved mIoU comparison chart")
plt.close()

# ============================================================================
# Figure 2: Per-Class IoU Heatmap
# ============================================================================
class_cols = ['Clutter', 'Building', 'Road', 'Tree', 'Vegetation',
              'Moving Car', 'Static Car', 'Human']
class_data = df[class_cols].values

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(class_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=90)

# Set ticks and labels
ax.set_xticks(np.arange(len(class_cols)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(class_cols, fontsize=11, rotation=45, ha='right')
ax.set_yticklabels(models, fontsize=11)

# Add text annotations
for i in range(len(models)):
    for j in range(len(class_cols)):
        text = ax.text(j, i, f'{class_data[i, j]:.1f}',
                      ha="center", va="center", color="black",
                      fontsize=9, fontweight='bold')

ax.set_title('Per-Class IoU Performance (%)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Class', fontsize=14, fontweight='bold')
ax.set_ylabel('Model Configuration', fontsize=14, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('IoU (%)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'per_class_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'per_class_heatmap.pdf', bbox_inches='tight')
print(f"Saved per-class heatmap")
plt.close()

# ============================================================================
# Figure 3: Overall Metrics Comparison (mIoU, F1, OA)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, df['mIoU'], width, label='mIoU',
               color='#3498db', edgecolor='black', linewidth=1)
bars2 = ax.bar(x, df['F1'], width, label='F1 Score',
               color='#2ecc71', edgecolor='black', linewidth=1)
bars3 = ax.bar(x + width, df['OA'], width, label='Overall Accuracy',
               color='#e74c3c', edgecolor='black', linewidth=1)

ax.set_xlabel('Model Configuration', fontsize=14, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax.set_title('Overall Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=12, loc='lower left')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'overall_metrics.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'overall_metrics.pdf', bbox_inches='tight')
print(f"Saved overall metrics comparison")
plt.close()

# ============================================================================
# Figure 4: SSL vs ImageNet Direct Comparison
# ============================================================================
ssl_models = ['DeepLabV3 + SSL', 'UNet++ + SSL', 'UNet + SSL']
imagenet_models = ['DeepLabV3 + ImageNet', 'UNet++ + ImageNet', 'UNet + ImageNet']
architecture_names = ['DeepLabV3', 'UNet++', 'UNet']

ssl_miou = [df[df['Model'] == m]['mIoU'].values[0] for m in ssl_models]
imagenet_miou = [df[df['Model'] == m]['mIoU'].values[0] for m in imagenet_models]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(architecture_names))
width = 0.35

bars1 = ax.bar(x - width/2, ssl_miou, width, label='SSL (MoCo v2)',
               color='#3498db', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, imagenet_miou, width, label='ImageNet',
               color='#e74c3c', edgecolor='black', linewidth=1.2)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.2f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

# Calculate improvement
improvements = [(ssl - img) for ssl, img in zip(ssl_miou, imagenet_miou)]
for i, imp in enumerate(improvements):
    color = 'green' if imp > 0 else 'red'
    ax.text(i, max(ssl_miou[i], imagenet_miou[i]) + 2,
            f'{"+" if imp > 0 else ""}{imp:.2f}%',
            ha='center', fontsize=10, fontweight='bold', color=color)

ax.set_xlabel('Architecture', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean IoU (%)', fontsize=14, fontweight='bold')
ax.set_title('SSL vs ImageNet Pretraining Comparison',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(architecture_names, fontsize=12)
ax.legend(fontsize=12, loc='lower right')
ax.set_ylim(0, 70)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ssl_vs_imagenet.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'ssl_vs_imagenet.pdf', bbox_inches='tight')
print(f"Saved SSL vs ImageNet comparison")
plt.close()

# ============================================================================
# Figure 5: Architecture Comparison (DeepLabV3 vs UNet vs UNet++)
# ============================================================================
# Group by initialization method
init_methods = ['SSL', 'ImageNet']
deeplab_scores = []
unetpp_scores = []
unet_scores = []

for method in init_methods:
    deeplab = df[df['Model'].str.contains(f'DeepLabV3.*{method}')]['mIoU'].values
    unetpp = df[df['Model'].str.contains(f'UNet\\+\\+.*{method}')]['mIoU'].values
    unet = df[(df['Model'].str.contains(f'UNet.*{method}')) &
              (~df['Model'].str.contains('UNet\\+\\+'))]['mIoU'].values

    deeplab_scores.append(deeplab[0] if len(deeplab) > 0 else 0)
    unetpp_scores.append(unetpp[0] if len(unetpp) > 0 else 0)
    unet_scores.append(unet[0] if len(unet) > 0 else 0)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(init_methods))
width = 0.25

bars1 = ax.bar(x - width, deeplab_scores, width, label='DeepLabV3',
               color='#9b59b6', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x, unetpp_scores, width, label='UNet++',
               color='#3498db', edgecolor='black', linewidth=1.2)
bars3 = ax.bar(x + width, unet_scores, width, label='UNet',
               color='#e74c3c', edgecolor='black', linewidth=1.2)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.2f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

ax.set_xlabel('Initialization Method', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean IoU (%)', fontsize=14, fontweight='bold')
ax.set_title('Architecture Comparison Across Initialization Methods',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(init_methods, fontsize=12)
ax.legend(fontsize=12, loc='lower right')
ax.set_ylim(0, 70)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'architecture_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'architecture_comparison.pdf', bbox_inches='tight')
print(f"Saved architecture comparison")
plt.close()

# ============================================================================
# Statistical Summary
# ============================================================================
print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)

print("\n1. Best Overall Model:")
best_idx = df['mIoU'].idxmax()
print(f"   {df.loc[best_idx, 'Model']} - mIoU: {df.loc[best_idx, 'mIoU']:.2f}%")

print("\n2. SSL vs ImageNet Average Performance:")
ssl_avg = df[df['Model'].str.contains('SSL')]['mIoU'].mean()
imagenet_avg = df[df['Model'].str.contains('ImageNet')]['mIoU'].mean()
print(f"   SSL Average mIoU: {ssl_avg:.2f}%")
print(f"   ImageNet Average mIoU: {imagenet_avg:.2f}%")
print(f"   Difference: {ssl_avg - imagenet_avg:.2f}% ({'SSL better' if ssl_avg > imagenet_avg else 'ImageNet better'})")

print("\n3. Architecture Performance (averaged across init methods):")
for arch in ['DeepLabV3', 'UNet++', 'UNet']:
    if arch == 'UNet++':
        mask = df['Model'].str.contains('UNet\\+\\+')
    elif arch == 'UNet':
        mask = (df['Model'].str.contains('UNet')) & (~df['Model'].str.contains('UNet\\+\\+'))
    else:
        mask = df['Model'].str.contains(arch)

    arch_avg = df[mask]['mIoU'].mean()
    print(f"   {arch}: {arch_avg:.2f}%")

print("\n4. Most Challenging Classes (lowest avg IoU):")
class_cols = ['Clutter', 'Building', 'Road', 'Tree', 'Vegetation',
              'Moving Car', 'Static Car', 'Human']
class_avgs = df[class_cols].mean().sort_values()
print("   " + "\n   ".join([f"{cls}: {val:.2f}%" for cls, val in class_avgs.head(3).items()]))

print("\n5. Best Performing Classes (highest avg IoU):")
print("   " + "\n   ".join([f"{cls}: {val:.2f}%" for cls, val in class_avgs.tail(3).items()]))

print("\n" + "="*80)
print(f"All figures saved to: {OUTPUT_DIR}")
print("="*80)
