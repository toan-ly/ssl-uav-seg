"""
Comprehensive visualization script for UAV Semantic Segmentation Research Poster
Generates detailed methodology diagrams, training dynamics, and in-depth analysis plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts in PDF

# Create output directory
ROOT = Path(__file__).resolve().parent.parent.parent
print(f"ROOT directory: {ROOT}")

OUTPUT_DIR = ROOT / 'analysis/new_findings'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Results data
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

df = pd.DataFrame(results_data)

print("="*80)
print("GENERATING COMPREHENSIVE VISUALIZATIONS FOR RESEARCH POSTER")
print("="*80)

# ============================================================================
# Figure 1: Two-Phase Training Strategy Flowchart
# ============================================================================
print("\n[1/10] Creating Two-Phase Training Strategy Flowchart...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(7, 7.5, 'Two-Phase Fine-Tuning Strategy',
        fontsize=18, fontweight='bold', ha='center')

# Phase 1: Frozen Encoder
phase1_box = FancyBboxPatch((0.5, 4.5), 6, 2.5,
                            boxstyle="round,pad=0.1",
                            edgecolor='#3498db', facecolor='#ebf5fb', linewidth=2.5)
ax.add_patch(phase1_box)
ax.text(3.5, 6.5, 'Phase 1: Frozen Encoder Training',
        fontsize=14, fontweight='bold', ha='center', color='#3498db')
ax.text(3.5, 6.0, '• Epochs: 10', fontsize=11, ha='center')
ax.text(3.5, 5.7, '• Encoder: Frozen (requires_grad=False)', fontsize=11, ha='center')
ax.text(3.5, 5.4, '• Decoder: Trainable', fontsize=11, ha='center')
ax.text(3.5, 5.1, '• Learning Rate: 4e-4 (decoder only)', fontsize=11, ha='center')
ax.text(3.5, 4.8, '• Early Stopping: Patience 2', fontsize=11, ha='center')

# Phase 2: Unfrozen Encoder
phase2_box = FancyBboxPatch((7.5, 4.5), 6, 2.5,
                            boxstyle="round,pad=0.1",
                            edgecolor='#e74c3c', facecolor='#fadbd8', linewidth=2.5)
ax.add_patch(phase2_box)
ax.text(10.5, 6.5, 'Phase 2: End-to-End Fine-Tuning',
        fontsize=14, fontweight='bold', ha='center', color='#e74c3c')
ax.text(10.5, 6.0, '• Epochs: 100 (with early stopping)', fontsize=11, ha='center')
ax.text(10.5, 5.7, '• Encoder: Unfrozen (requires_grad=True)', fontsize=11, ha='center')
ax.text(10.5, 5.4, '• Decoder LR: 4e-4 | Encoder LR: 1e-4', fontsize=11, ha='center')
ax.text(10.5, 5.1, '• Differential Learning Rates', fontsize=11, ha='center')
ax.text(10.5, 4.8, '• Early Stopping: Patience 7', fontsize=11, ha='center')

# Arrow between phases
arrow = FancyArrowPatch((6.5, 5.75), (7.5, 5.75),
                       arrowstyle='->', mutation_scale=30,
                       linewidth=3, color='#2c3e50')
ax.add_patch(arrow)

# Initialization boxes at top
ssl_box = FancyBboxPatch((1, 2.5), 3.5, 1.5,
                         boxstyle="round,pad=0.08",
                         edgecolor='#3498db', facecolor='#d4e6f1', linewidth=2)
ax.add_patch(ssl_box)
ax.text(2.75, 3.5, 'SSL Initialization', fontsize=12, fontweight='bold', ha='center')
ax.text(2.75, 3.1, 'MoCo v2 Pretrained', fontsize=10, ha='center')
ax.text(2.75, 2.85, 'ResNet-50 Encoder', fontsize=10, ha='center')

imagenet_box = FancyBboxPatch((5.25, 2.5), 3.5, 1.5,
                              boxstyle="round,pad=0.08",
                              edgecolor='#e74c3c', facecolor='#fadbd8', linewidth=2)
ax.add_patch(imagenet_box)
ax.text(7, 3.5, 'ImageNet Initialization', fontsize=12, fontweight='bold', ha='center')
ax.text(7, 3.1, 'Standard Pretrained', fontsize=10, ha='center')
ax.text(7, 2.85, 'ResNet-50 Encoder', fontsize=10, ha='center')

random_box = FancyBboxPatch((9.5, 2.5), 3.5, 1.5,
                            boxstyle="round,pad=0.08",
                            edgecolor='#95a5a6', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(random_box)
ax.text(11.25, 3.5, 'Random Initialization', fontsize=12, fontweight='bold', ha='center')
ax.text(11.25, 3.1, 'No Pretraining', fontsize=10, ha='center')
ax.text(11.25, 2.85, '(Baseline)', fontsize=10, ha='center')

# Arrows from initialization to Phase 1
for x in [2.75, 7, 11.25]:
    arrow = FancyArrowPatch((x, 2.5), (3.5, 4.5),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='#7f8c8d', alpha=0.6)
    ax.add_patch(arrow)

# Output box
output_box = FancyBboxPatch((4.5, 0.5), 5, 1.2,
                            boxstyle="round,pad=0.08",
                            edgecolor='#2ecc71', facecolor='#d5f4e6', linewidth=2.5)
ax.add_patch(output_box)
ax.text(7, 1.4, 'Final Segmentation Model', fontsize=13, fontweight='bold', ha='center', color='#2ecc71')
ax.text(7, 0.95, 'Optimized for UAV Scene Understanding', fontsize=10, ha='center')

# Arrow from Phase 2 to output
arrow = FancyArrowPatch((10.5, 4.5), (7, 1.7),
                       arrowstyle='->', mutation_scale=25,
                       linewidth=3, color='#2c3e50')
ax.add_patch(arrow)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_strategy_flowchart.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'training_strategy_flowchart.pdf', bbox_inches='tight')
print(f"   [OK] Saved training strategy flowchart")
plt.close()

# ============================================================================
# Figure 2: Data Augmentation Pipeline
# ============================================================================
print("\n[2/10] Creating Data Augmentation Pipeline Diagram...")

fig, ax = plt.subplots(figsize=(15, 6))
ax.set_xlim(0, 15)
ax.set_ylim(0, 6)
ax.axis('off')

# Title
ax.text(7.5, 5.5, 'Training Data Augmentation Pipeline',
        fontsize=16, fontweight='bold', ha='center')

# Stage 1: Original Image
stage1 = FancyBboxPatch((0.5, 3), 2.5, 1.8,
                        boxstyle="round,pad=0.08",
                        edgecolor='#3498db', facecolor='#ebf5fb', linewidth=2)
ax.add_patch(stage1)
ax.text(1.75, 4.4, 'Original Image', fontsize=11, fontweight='bold', ha='center')
ax.text(1.75, 3.95, '4K Resolution', fontsize=9, ha='center')
ax.text(1.75, 3.65, 'UAVid Dataset', fontsize=9, ha='center')
ax.text(1.75, 3.35, '(with RGB mask)', fontsize=9, ha='center')

# Stage 2: Random Crop
stage2 = FancyBboxPatch((3.5, 3), 2.5, 1.8,
                        boxstyle="round,pad=0.08",
                        edgecolor='#9b59b6', facecolor='#f4ecf7', linewidth=2)
ax.add_patch(stage2)
ax.text(4.75, 4.4, 'Random Crop', fontsize=11, fontweight='bold', ha='center')
ax.text(4.75, 3.95, 'Patch: 512×512', fontsize=9, ha='center')
ax.text(4.75, 3.65, '6 samples/image', fontsize=9, ha='center')
ax.text(4.75, 3.35, 'MONAI Transform', fontsize=9, ha='center')

# Stage 3: Geometric Augmentation
stage3 = FancyBboxPatch((6.5, 3), 2.5, 1.8,
                        boxstyle="round,pad=0.08",
                        edgecolor='#e67e22', facecolor='#fdebd0', linewidth=2)
ax.add_patch(stage3)
ax.text(7.75, 4.4, 'Geometric Aug', fontsize=11, fontweight='bold', ha='center')
ax.text(7.75, 3.95, 'Rotation (±15°)', fontsize=9, ha='center')
ax.text(7.75, 3.65, 'H/V Flip (p=0.5)', fontsize=9, ha='center')
ax.text(7.75, 3.35, 'Albumentations', fontsize=9, ha='center')

# Stage 4: Photometric Augmentation
stage4 = FancyBboxPatch((9.5, 3), 2.5, 1.8,
                        boxstyle="round,pad=0.08",
                        edgecolor='#16a085', facecolor='#d1f2eb', linewidth=2)
ax.add_patch(stage4)
ax.text(10.75, 4.4, 'Photometric Aug', fontsize=11, fontweight='bold', ha='center')
ax.text(10.75, 3.95, 'Color Jitter', fontsize=9, ha='center')
ax.text(10.75, 3.65, 'Brightness/Contrast', fontsize=9, ha='center')
ax.text(10.75, 3.35, 'Weather Effects', fontsize=9, ha='center')

# Stage 5: Copy-Paste Augmentation
stage5 = FancyBboxPatch((12.5, 3), 2.5, 1.8,
                        boxstyle="round,pad=0.08",
                        edgecolor='#c0392b', facecolor='#fadbd8', linewidth=2)
ax.add_patch(stage5)
ax.text(13.75, 4.4, 'Copy-Paste Aug', fontsize=11, fontweight='bold', ha='center')
ax.text(13.75, 3.95, 'Rare class boost', fontsize=9, ha='center')
ax.text(13.75, 3.65, 'Human: 50%', fontsize=9, ha='center')
ax.text(13.75, 3.35, 'Cars: 25-30%', fontsize=9, ha='center')

# Arrows between stages
for i, x_start in enumerate([3, 6, 9, 12]):
    arrow = FancyArrowPatch((x_start, 3.9), (x_start+0.5, 3.9),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color='#34495e')
    ax.add_patch(arrow)

# Output
output = FancyBboxPatch((5, 0.5), 5, 1.5,
                        boxstyle="round,pad=0.08",
                        edgecolor='#27ae60', facecolor='#d5f4e6', linewidth=2.5)
ax.add_patch(output)
ax.text(7.5, 1.6, 'Augmented Training Batch', fontsize=12, fontweight='bold', ha='center', color='#27ae60')
ax.text(7.5, 1.2, 'Shape: (B=6, C=3, H=512, W=512)', fontsize=10, ha='center')
ax.text(7.5, 0.85, 'Label: (B=6, 1, H=512, W=512) - class indices', fontsize=10, ha='center')

# Arrow to output
arrow = FancyArrowPatch((7.5, 3), (7.5, 2),
                       arrowstyle='->', mutation_scale=25,
                       linewidth=3, color='#2c3e50')
ax.add_patch(arrow)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'augmentation_pipeline.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'augmentation_pipeline.pdf', bbox_inches='tight')
print(f"   [OK] Saved data augmentation pipeline")
plt.close()

# ============================================================================
# Figure 3: Loss Function Breakdown
# ============================================================================
print("\n[3/10] Creating Loss Function Breakdown...")

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')

# Title
ax.text(6, 6.5, 'Combined Dice + Cross-Entropy Loss with Class Weights',
        fontsize=16, fontweight='bold', ha='center')

# Main loss box
main_box = FancyBboxPatch((1, 4.2), 10, 1.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='#2c3e50', facecolor='#ecf0f1', linewidth=2.5)
ax.add_patch(main_box)
ax.text(6, 5.6, r'$\mathcal{L}_{total} = \mathcal{L}_{Dice} + \mathcal{L}_{CE}$',
        fontsize=14, ha='center', fontweight='bold')
ax.text(6, 5.15, 'Combines region overlap (Dice) with pixel-wise classification (CE)',
        fontsize=10, ha='center', style='italic')
ax.text(6, 4.75, 'Class Weights: [BG:2.0, Std:1.0, Moving Car:3.0, Static Car:3.0, Human:5.0]',
        fontsize=9, ha='center')

# Dice Loss component
dice_box = FancyBboxPatch((0.5, 2), 5, 1.8,
                          boxstyle="round,pad=0.08",
                          edgecolor='#3498db', facecolor='#ebf5fb', linewidth=2)
ax.add_patch(dice_box)
ax.text(3, 3.5, 'Dice Loss', fontsize=13, fontweight='bold', ha='center', color='#3498db')
ax.text(3, 3.1, r'$\mathcal{L}_{Dice} = 1 - \frac{2|P \cap G|}{|P| + |G|}$',
        fontsize=11, ha='center')
ax.text(3, 2.7, '• Region-based overlap metric', fontsize=9, ha='center')
ax.text(3, 2.45, '• Handles class imbalance', fontsize=9, ha='center')
ax.text(3, 2.2, '• Optimizes for segmentation quality', fontsize=9, ha='center')

# Cross-Entropy component
ce_box = FancyBboxPatch((6.5, 2), 5, 1.8,
                        boxstyle="round,pad=0.08",
                        edgecolor='#e74c3c', facecolor='#fadbd8', linewidth=2)
ax.add_patch(ce_box)
ax.text(9, 3.5, 'Cross-Entropy Loss', fontsize=13, fontweight='bold', ha='center', color='#e74c3c')
ax.text(9, 3.1, r'$\mathcal{L}_{CE} = -\sum_{c} w_c \cdot y_c \log(\hat{y}_c)$',
        fontsize=11, ha='center')
ax.text(9, 2.7, '• Pixel-wise classification', fontsize=9, ha='center')
ax.text(9, 2.45, '• Weighted for rare classes', fontsize=9, ha='center')
ax.text(9, 2.2, '• Optimizes class boundaries', fontsize=9, ha='center')

# Arrows from components to main
arrow1 = FancyArrowPatch((3, 3.8), (4, 4.2),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2.5, color='#7f8c8d')
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((9, 3.8), (8, 4.2),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2.5, color='#7f8c8d')
ax.add_patch(arrow2)

# Class weights table
weights = [
    ('Background', 2.0, '#bdc3c7'),
    ('Building', 1.0, '#3498db'),
    ('Road', 1.0, '#9b59b6'),
    ('Tree', 1.0, '#27ae60'),
    ('Vegetation', 1.0, '#f39c12'),
    ('Moving Car', 3.0, '#e74c3c'),
    ('Static Car', 3.0, '#c0392b'),
    ('Human', 5.0, '#8e44ad'),
]

ax.text(6, 1.5, 'Class Weights Rationale:', fontsize=11, fontweight='bold', ha='center')
y_pos = 1.1
for i, (cls, weight, color) in enumerate(weights):
    if i % 4 == 0:
        y_pos -= 0.25
        x_start = 1
    x = x_start + (i % 4) * 2.75
    ax.text(x, y_pos, f'{cls}: {weight:.1f}', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'loss_function_breakdown.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'loss_function_breakdown.pdf', bbox_inches='tight')
print(f"   [OK] Saved loss function breakdown")
plt.close()

# ============================================================================
# Figure 4: Model Architecture Comparison
# ============================================================================
print("\n[4/10] Creating Model Architecture Comparison...")

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# Title
fig.suptitle('Segmentation Architecture Comparison: DeepLabV3 vs UNet vs UNet++',
             fontsize=18, fontweight='bold', y=0.98)

# DeepLabV3
ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, 16)
ax1.set_ylim(0, 3)
ax1.axis('off')
ax1.text(8, 2.7, 'DeepLabV3: Atrous Spatial Pyramid Pooling (ASPP)',
         fontsize=14, fontweight='bold', ha='center', color='#9b59b6')

# DeepLabV3 components
encoder = Rectangle((0.5, 1), 3, 1.2, facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
ax1.add_patch(encoder)
ax1.text(2, 1.6, 'ResNet-50\nEncoder', fontsize=10, ha='center', fontweight='bold')

aspp = Rectangle((4.5, 0.8), 7, 1.6, facecolor='#fff3cd', edgecolor='#ffc107', linewidth=2)
ax1.add_patch(aspp)
ax1.text(8, 2.1, 'ASPP Module', fontsize=11, ha='center', fontweight='bold')
ax1.text(5.5, 1.5, '1×1 conv', fontsize=8, ha='center')
ax1.text(6.8, 1.5, '3×3, r=6', fontsize=8, ha='center')
ax1.text(8.1, 1.5, '3×3, r=12', fontsize=8, ha='center')
ax1.text(9.4, 1.5, '3×3, r=18', fontsize=8, ha='center')
ax1.text(10.7, 1.5, 'Global\nPool', fontsize=8, ha='center')

decoder = Rectangle((12.5, 1), 3, 1.2, facecolor='#d1ecf1', edgecolor='#17a2b8', linewidth=2)
ax1.add_patch(decoder)
ax1.text(14, 1.6, 'Decoder\n(1×1 conv)', fontsize=10, ha='center', fontweight='bold')

# Arrows
for x in [(3.5, 4.5), (11.5, 12.5)]:
    arrow = FancyArrowPatch((x[0], 1.6), (x[1], 1.6),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='#2c3e50')
    ax1.add_patch(arrow)

ax1.text(8, 0.3, '[OK] Multi-scale context | [OK] Robust to initialization | [OK] Best overall performance (65.32%)',
         fontsize=9, ha='center', style='italic')

# UNet
ax2 = fig.add_subplot(gs[1, :])
ax2.set_xlim(0, 16)
ax2.set_ylim(0, 3)
ax2.axis('off')
ax2.text(8, 2.7, 'UNet: Symmetric Encoder-Decoder with Skip Connections',
         fontsize=14, fontweight='bold', ha='center', color='#3498db')

# UNet encoder path (downsampling)
for i, y in enumerate([1.8, 1.3, 0.8]):
    size = 1.5 - i*0.3
    enc_box = Rectangle((1, y), size, 0.4, facecolor='#cfe2ff', edgecolor='#0d6efd', linewidth=1.5)
    ax2.add_patch(enc_box)
    ax2.text(1 + size/2, y+0.2, f'Enc {i+1}', fontsize=8, ha='center')

# UNet decoder path (upsampling)
for i, y in enumerate([0.8, 1.3, 1.8]):
    size = 1.2 + i*0.3
    dec_box = Rectangle((14-size, y), size, 0.4, facecolor='#d1e7dd', edgecolor='#198754', linewidth=1.5)
    ax2.add_patch(dec_box)
    ax2.text(14 - size/2, y+0.2, f'Dec {3-i}', fontsize=8, ha='center')

# Skip connections
for y in [1.0, 1.5, 2.0]:
    skip = FancyArrowPatch((3, y), (13, y),
                          arrowstyle='<->', mutation_scale=15, linewidth=1.5,
                          color='#6610f2', linestyle='dashed', alpha=0.7)
    ax2.add_patch(skip)

ax2.text(8, 2.4, 'Skip Connections (Feature Concatenation)', fontsize=9,
         ha='center', color='#6610f2', fontweight='bold')
ax2.text(8, 0.3, '[OK] Preserves spatial information | [OK] Simple architecture | • Benefits from SSL (+3.71%)',
         fontsize=9, ha='center', style='italic')

# UNet++
ax3 = fig.add_subplot(gs[2, :])
ax3.set_xlim(0, 16)
ax3.set_ylim(0, 3)
ax3.axis('off')
ax3.text(8, 2.7, 'UNet++: Nested Skip Pathways with Deep Supervision',
         fontsize=14, fontweight='bold', ha='center', color='#e74c3c')

# UNet++ encoder
for i, y in enumerate([1.8, 1.3, 0.8]):
    size = 1.5 - i*0.3
    enc_box = Rectangle((1, y), size, 0.4, facecolor='#f8d7da', edgecolor='#dc3545', linewidth=1.5)
    ax3.add_patch(enc_box)

# UNet++ decoder with nested connections
for i, y in enumerate([0.8, 1.3, 1.8]):
    size = 1.2 + i*0.3
    dec_box = Rectangle((14-size, y), size, 0.4, facecolor='#d1e7dd', edgecolor='#198754', linewidth=1.5)
    ax3.add_patch(dec_box)

# Nested skip connections (more complex)
for y1, y2 in [(0.8, 1.0), (1.3, 1.5), (1.8, 2.0), (1.0, 1.2), (1.5, 1.7)]:
    skip = FancyArrowPatch((4, y1), (12, y2),
                          arrowstyle='->', mutation_scale=12, linewidth=1.2,
                          color='#fd7e14', alpha=0.6)
    ax3.add_patch(skip)

ax3.text(8, 2.4, 'Nested Skip Pathways + Deep Supervision', fontsize=9,
         ha='center', color='#fd7e14', fontweight='bold')
ax3.text(8, 0.3, '[OK] Dense feature propagation | [OK] Best SSL performance (+2.60%) | • Higher capacity',
         fontsize=9, ha='center', style='italic')

plt.savefig(OUTPUT_DIR / 'architecture_comparison_detailed.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'architecture_comparison_detailed.pdf', bbox_inches='tight')
print(f"   [OK] Saved detailed architecture comparison")
plt.close()


# ============================================================================
# Figure 6: Class-Wise Performance Analysis
# ============================================================================
print("\n[6/10] Creating Class-Wise Performance Deep Dive...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Per-Class Performance Analysis: Strengths and Challenges',
             fontsize=18, fontweight='bold')

class_cols = ['Clutter', 'Building', 'Road', 'Tree', 'Vegetation',
              'Moving Car', 'Static Car', 'Human']

# SSL vs ImageNet per class
ssl_models = ['DeepLabV3 + SSL', 'UNet++ + SSL', 'UNet + SSL']
imagenet_models = ['DeepLabV3 + ImageNet', 'UNet++ + ImageNet', 'UNet + ImageNet']

ssl_avg = df[df['Model'].isin(ssl_models)][class_cols].mean()
imagenet_avg = df[df['Model'].isin(imagenet_models)][class_cols].mean()
difference = ssl_avg - imagenet_avg

# Plot 1: SSL vs ImageNet by Class
ax = axes[0, 0]
x = np.arange(len(class_cols))
width = 0.35

bars1 = ax.bar(x - width/2, ssl_avg, width, label='SSL Average',
               color='#3498db', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, imagenet_avg, width, label='ImageNet Average',
               color='#e74c3c', edgecolor='black', linewidth=1.2)

ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Average IoU (%)', fontsize=12, fontweight='bold')
ax.set_title('(A) SSL vs ImageNet: Average Performance by Class', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_cols, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 100)

# Annotate differences
for i, diff in enumerate(difference):
    color = 'green' if diff > 0 else 'red'
    y_pos = max(ssl_avg[i], imagenet_avg[i]) + 3
    ax.text(i, y_pos, f'{diff:+.1f}', ha='center', fontsize=9,
            fontweight='bold', color=color)

# Plot 2: Performance Distribution Box Plot
ax = axes[0, 1]
class_data = []
for cls in class_cols:
    values = df[df['Model'] != 'UNet+Random'][cls].values
    class_data.append(values)

bp = ax.boxplot(class_data, labels=class_cols, patch_artist=True,
                showmeans=True, meanline=True)

for patch, cls in zip(bp['boxes'], class_cols):
    if cls in ['Human', 'Static Car']:
        patch.set_facecolor('#fadbd8')
        patch.set_edgecolor('#c0392b')
    elif cls in ['Building', 'Road', 'Tree']:
        patch.set_facecolor('#d5f4e6')
        patch.set_edgecolor('#27ae60')
    else:
        patch.set_facecolor('#fff3cd')
        patch.set_edgecolor('#f39c12')

ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('IoU (%) Distribution', fontsize=12, fontweight='bold')
ax.set_title('(B) Performance Variability Across Models', fontsize=13, fontweight='bold')
ax.set_xticklabels(class_cols, rotation=45, ha='right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=50, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Challenge Threshold')
ax.legend(fontsize=10)

# Plot 3: Class Difficulty Ranking
ax = axes[1, 0]
overall_avg = df[df['Model'] != 'UNet+Random'][class_cols].mean().sort_values()
colors = ['#c0392b' if x < 50 else ('#f39c12' if x < 70 else '#27ae60')
          for x in overall_avg.values]

bars = ax.barh(range(len(overall_avg)), overall_avg.values, color=colors,
               edgecolor='black', linewidth=1.2)

ax.set_yticks(range(len(overall_avg)))
ax.set_yticklabels(overall_avg.index, fontsize=11)
ax.set_xlabel('Average IoU (%)', fontsize=12, fontweight='bold')
ax.set_title('(C) Class Difficulty Ranking (Easiest to Hardest)', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add difficulty labels
for i, (idx, val) in enumerate(overall_avg.items()):
    difficulty = 'Hard' if val < 50 else ('Medium' if val < 70 else 'Easy')
    ax.text(val + 2, i, f'{val:.1f}% ({difficulty})', va='center', fontsize=9, fontweight='bold')

ax.axvline(x=50, color='red', linestyle=':', linewidth=2, alpha=0.5)
ax.axvline(x=70, color='orange', linestyle=':', linewidth=2, alpha=0.5)

# Plot 4: Best Model by Class
ax = axes[1, 1]
best_models = []
best_scores = []
for cls in class_cols:
    best_model_idx = df[df['Model'] != 'UNet+Random'][cls].idxmax()
    best_model = df.loc[best_model_idx, 'Model']
    best_score = df.loc[best_model_idx, cls]
    best_models.append(best_model)
    best_scores.append(best_score)

# Color code by initialization
colors = []
for model in best_models:
    if 'SSL' in model:
        colors.append('#3498db')
    else:
        colors.append('#e74c3c')

bars = ax.bar(range(len(class_cols)), best_scores, color=colors,
              edgecolor='black', linewidth=1.2)

ax.set_xticks(range(len(class_cols)))
ax.set_xticklabels(class_cols, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Best IoU (%)', fontsize=12, fontweight='bold')
ax.set_title('(D) Best Model Performance by Class', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add model labels
for i, (score, model) in enumerate(zip(best_scores, best_models)):
    model_short = model.replace('DeepLabV3+', 'DL3+').replace('UNet++', 'U++').replace('UNet+', 'U+')
    ax.text(i, score + 2, model_short, ha='center', fontsize=8, rotation=90, va='bottom')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='black', label='SSL Wins'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='ImageNet Wins')
]
ax.legend(handles=legend_elements, fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'class_wise_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'class_wise_analysis.pdf', bbox_inches='tight')
print(f"   [OK] Saved class-wise performance analysis")
plt.close()

# ============================================================================
# Figure 7: Sliding Window Inference Visualization
# ============================================================================
print("\n[7/10] Creating Sliding Window Inference Diagram...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(7, 7.5, 'Sliding Window Inference Strategy for High-Resolution Images',
        fontsize=16, fontweight='bold', ha='center')

# Full image
full_img = Rectangle((0.5, 4), 5, 3, facecolor='#e8f4f8',
                     edgecolor='#3498db', linewidth=3)
ax.add_patch(full_img)
ax.text(3, 7.3, 'Full Test Image (4K)', fontsize=11, fontweight='bold', ha='center')
ax.text(3, 3.7, '(H, W, 3)', fontsize=10, ha='center', style='italic')

# Sliding windows
window_positions = [(1, 5.5), (2, 5.5), (3, 5.5), (1, 4.5), (2, 4.5), (3, 4.5)]
for i, (x, y) in enumerate(window_positions):
    window = Rectangle((x, y), 1.2, 0.8, facecolor='#fff3cd',
                       edgecolor='#f39c12', linewidth=2, alpha=0.7)
    ax.add_patch(window)
    ax.text(x + 0.6, y + 0.4, f'{i+1}', fontsize=9, ha='center', fontweight='bold')

ax.text(3, 4.2, '↑ Overlapping 512×512 patches (50% overlap)',
        fontsize=10, ha='center', style='italic')

# Processing
arrow1 = FancyArrowPatch((5.5, 5.5), (6.5, 5.5),
                        arrowstyle='->', mutation_scale=30,
                        linewidth=3, color='#2c3e50')
ax.add_patch(arrow1)

process_box = FancyBboxPatch((6.5, 4.5), 3, 2,
                             boxstyle="round,pad=0.1",
                             edgecolor='#9b59b6', facecolor='#f4ecf7', linewidth=2.5)
ax.add_patch(process_box)
ax.text(8, 6.2, 'Segmentation Model', fontsize=12, fontweight='bold', ha='center', color='#9b59b6')
ax.text(8, 5.7, 'Process each patch', fontsize=10, ha='center')
ax.text(8, 5.35, 'Batch size: 64', fontsize=9, ha='center')
ax.text(8, 5.0, 'Mode: Gaussian', fontsize=9, ha='center')
ax.text(8, 4.7, 'weighting', fontsize=9, ha='center', style='italic')

# Gaussian weighting visualization
ax2 = fig.add_axes([0.51, 0.62, 0.12, 0.12])
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2) / 2)
ax2.contourf(X, Y, Z, levels=20, cmap='YlOrRd', alpha=0.7)
ax2.set_title('Gaussian Weight', fontsize=8)
ax2.axis('off')

# Aggregation
arrow2 = FancyArrowPatch((9.5, 5.5), (10.5, 5.5),
                        arrowstyle='->', mutation_scale=30,
                        linewidth=3, color='#2c3e50')
ax.add_patch(arrow2)

# Final output
output_img = Rectangle((10.5, 4), 3, 3, facecolor='#d5f4e6',
                       edgecolor='#27ae60', linewidth=3)
ax.add_patch(output_img)
ax.text(12, 7.3, 'Final Prediction', fontsize=11, fontweight='bold', ha='center')
ax.text(12, 3.7, 'Full Resolution Mask', fontsize=10, ha='center', style='italic')

# Parameters box
params_box = FancyBboxPatch((1, 1), 12, 2,
                            boxstyle="round,pad=0.1",
                            edgecolor='#34495e', facecolor='#ecf0f1', linewidth=2)
ax.add_patch(params_box)
ax.text(7, 2.6, 'Inference Configuration', fontsize=12, fontweight='bold', ha='center')

params_text = [
    '• ROI Size: 512 × 512 pixels',
    '• Overlap: 50% (prevents boundary artifacts)',
    '• Batch Size: 64 patches processed simultaneously',
    '• Weighting: Gaussian (center pixels weighted more)',
    '• Aggregation: Weighted average for overlapping regions'
]
y_pos = 2.2
for param in params_text:
    ax.text(7, y_pos, param, fontsize=10, ha='center')
    y_pos -= 0.25

ax.text(7, 1.1, '[OK] Handles arbitrary image sizes | [OK] Reduces memory footprint | [OK] Improved boundary predictions',
        fontsize=9, ha='center', style='italic', color='#27ae60', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sliding_window_inference.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'sliding_window_inference.pdf', bbox_inches='tight')
print(f"   [OK] Saved sliding window inference diagram")
plt.close()

# ============================================================================
# Figure 8: Optimizer and Scheduler Configuration
# ============================================================================
print("\n[8/10] Creating Optimizer and Scheduler Visualization...")

fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

fig.suptitle('Training Configuration: Optimizer and Learning Rate Scheduling',
             fontsize=16, fontweight='bold')

# Learning rate schedule
ax1 = fig.add_subplot(gs[0, :])
epochs = np.arange(0, 100)
lr_decoder = 4e-4 * (0.5 * (1 + np.cos(np.pi * epochs / 100)))
lr_encoder = 1e-4 * (0.5 * (1 + np.cos(np.pi * epochs / 100)))

ax1.plot(epochs, lr_decoder, label='Decoder LR', color='#3498db', linewidth=2.5)
ax1.plot(epochs, lr_encoder, label='Encoder LR (Phase 2)', color='#e74c3c', linewidth=2.5, linestyle='--')
ax1.axvline(x=10, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Phase 2 Starts')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
ax1.set_title('Cosine Annealing Learning Rate Schedule', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_yscale('log')

# Add annotations
ax1.annotate('Initial: 4e-4', xy=(0, 4e-4), xytext=(15, 6e-4),
            arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5),
            fontsize=10, color='#3498db', fontweight='bold')
ax1.annotate('Min: ~1e-6', xy=(99, lr_decoder[-1]), xytext=(80, 1e-5),
            arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5),
            fontsize=10, color='#3498db', fontweight='bold')

# Optimizer details
ax2 = fig.add_subplot(gs[1, 0])
ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

optim_box = FancyBboxPatch((0.5, 2), 9, 7,
                           boxstyle="round,pad=0.15",
                           edgecolor='#9b59b6', facecolor='#f4ecf7', linewidth=2.5)
ax2.add_patch(optim_box)

ax2.text(5, 8.3, 'AdamW Optimizer Configuration', fontsize=13, fontweight='bold', ha='center', color='#9b59b6')

params = [
    ('Parameter', 'Value', 'Rationale'),
    ('Optimizer', 'AdamW', 'Decoupled weight decay'),
    ('Base LR', '4e-4', 'Decoder learning rate'),
    ('Encoder LR', '1e-4', 'Lower for pretrained weights'),
    ('Weight Decay', '1e-2', 'Regularization'),
    ('Betas', '(0.9, 0.999)', 'Adam momentum terms'),
    ('Epsilon', '1e-8', 'Numerical stability'),
]

y_start = 7.5
for i, (param, value, rationale) in enumerate(params):
    y = y_start - i * 0.7
    if i == 0:  # Header
        ax2.text(1.5, y, param, fontsize=10, fontweight='bold', ha='left')
        ax2.text(4.5, y, value, fontsize=10, fontweight='bold', ha='left')
        ax2.text(7, y, rationale, fontsize=10, fontweight='bold', ha='left')
    else:
        ax2.text(1.5, y, param, fontsize=9, ha='left')
        ax2.text(4.5, y, value, fontsize=9, ha='left', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax2.text(7, y, rationale, fontsize=8, ha='left', style='italic', color='#7f8c8d')

ax2.text(5, 2.4, '[OK] Differential learning rates for encoder/decoder\n[OK] Cosine annealing with warm restarts\n[OK] Early stopping prevents overfitting',
         fontsize=9, ha='center', style='italic', bbox=dict(boxstyle='round,pad=0.5', facecolor='#d5f4e6', alpha=0.7))

# Training strategy summary
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)

strategy_box = FancyBboxPatch((0.5, 2), 9, 7,
                              boxstyle="round,pad=0.15",
                              edgecolor='#e74c3c', facecolor='#fadbd8', linewidth=2.5)
ax3.add_patch(strategy_box)

ax3.text(5, 8.3, 'Training Strategy Summary', fontsize=13, fontweight='bold', ha='center', color='#e74c3c')

strategies = [
    '1. Phase 1 (10 epochs):',
    '   • Freeze encoder weights',
    '   • Train decoder only (LR=4e-4)',
    '   • Early stopping patience: 2',
    '',
    '2. Phase 2 (100 epochs):',
    '   • Unfreeze encoder',
    '   • Differential LR (Enc:1e-4, Dec:4e-4)',
    '   • Early stopping patience: 7',
    '',
    '3. Regularization:',
    '   • Weight decay: 1e-2',
    '   • Data augmentation',
    '   • Early stopping',
]

y = 7.5
for strategy in strategies:
    if strategy == '':
        y -= 0.3
    elif strategy[0].isdigit():
        ax3.text(1, y, strategy, fontsize=10, fontweight='bold', ha='left')
        y -= 0.5
    else:
        ax3.text(1.5, y, strategy, fontsize=9, ha='left')
        y -= 0.4

ax3.text(5, 2.3, 'Best model saved based on validation mIoU',
         fontsize=9, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3cd', alpha=0.8))

plt.savefig(OUTPUT_DIR / 'optimizer_scheduler_config.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'optimizer_scheduler_config.pdf', bbox_inches='tight')
print(f"   [OK] Saved optimizer and scheduler configuration")
plt.close()

# ============================================================================
# Figure 9: Evaluation Metrics Formulas
# ============================================================================
print("\n[9/10] Creating Evaluation Metrics Reference...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'Evaluation Metrics: Definitions and Formulas',
        fontsize=18, fontweight='bold', ha='center')

metrics = [
    {
        'name': 'Mean Intersection over Union (mIoU)',
        'formula': r'$mIoU = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FP_c + FN_c}$',
        'description': 'Average IoU across all classes. Primary evaluation metric.',
        'color': '#3498db',
        'interpretation': 'Higher is better. Measures overlap between prediction and ground truth.',
    },
    {
        'name': 'F1 Score',
        'formula': r'$F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall} = \frac{2TP}{2TP + FP + FN}$',
        'description': 'Harmonic mean of precision and recall.',
        'color': '#2ecc71',
        'interpretation': 'Balances false positives and false negatives.',
    },
    {
        'name': 'Overall Accuracy (OA)',
        'formula': r'$OA = \frac{\sum_{c=1}^{C} TP_c}{\sum_{c=1}^{C} (TP_c + FP_c + FN_c + TN_c)}$',
        'description': 'Pixel-wise classification accuracy across all classes.',
        'color': '#e74c3c',
        'interpretation': 'Can be misleading with class imbalance (dominated by large classes).',
    },
]

y_start = 8.5
box_height = 2.2

for i, metric in enumerate(metrics):
    y = y_start - i * (box_height + 0.3)

    # Metric box
    metric_box = FancyBboxPatch((0.5, y), 13, box_height,
                                boxstyle="round,pad=0.1",
                                edgecolor=metric['color'],
                                facecolor=f"{metric['color']}20",
                                linewidth=2.5)
    ax.add_patch(metric_box)

    # Name
    ax.text(1, y + box_height - 0.3, metric['name'],
            fontsize=13, fontweight='bold', color=metric['color'])

    # Formula
    ax.text(7, y + box_height - 0.7, metric['formula'],
            fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

    # Description
    ax.text(1, y + 0.8, f"Definition: {metric['description']}",
            fontsize=10, style='italic')

    # Interpretation
    ax.text(1, y + 0.3, f"Interpretation: {metric['interpretation']}",
            fontsize=9, color='#7f8c8d')

# Confusion matrix reference
cm_y = 0.8
cm_box = FancyBboxPatch((0.5, cm_y), 6, 1.5,
                        boxstyle="round,pad=0.1",
                        edgecolor='#9b59b6', facecolor='#f4ecf7', linewidth=2)
ax.add_patch(cm_box)

ax.text(3.5, cm_y + 1.2, 'Confusion Matrix Terms', fontsize=11, fontweight='bold', ha='center', color='#9b59b6')
ax.text(1, cm_y + 0.7, 'TP: True Positive', fontsize=9)
ax.text(1, cm_y + 0.3, 'FP: False Positive', fontsize=9)
ax.text(4, cm_y + 0.7, 'TN: True Negative', fontsize=9)
ax.text(4, cm_y + 0.3, 'FN: False Negative', fontsize=9)

# Per-class computation note
note_box = FancyBboxPatch((7, cm_y), 6.5, 1.5,
                          boxstyle="round,pad=0.1",
                          edgecolor='#f39c12', facecolor='#fff3cd', linewidth=2)
ax.add_patch(note_box)

ax.text(10.25, cm_y + 1.2, 'Per-Class Computation', fontsize=11, fontweight='bold', ha='center', color='#f39c12')
ax.text(7.5, cm_y + 0.7, '1. Compute confusion matrix for each class', fontsize=9)
ax.text(7.5, cm_y + 0.3, '2. Calculate metric per class', fontsize=9)
ax.text(7.5, cm_y - 0.1, '3. Average across 8 UAVid classes', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'evaluation_metrics_reference.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'evaluation_metrics_reference.pdf', bbox_inches='tight')
print(f"   [OK] Saved evaluation metrics reference")
plt.close()

# ============================================================================
# Figure 10: MoCo v2 Hyperparameters Breakdown
# ============================================================================
print("\n[10/10] Creating MoCo v2 Hyperparameters Breakdown...")

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# Title
ax.text(7, 8.5, 'MoCo v2 Self-Supervised Learning: Hyperparameter Configuration',
        fontsize=16, fontweight='bold', ha='center')

# Main components
components = [
    {
        'title': 'Queue Size (K)',
        'value': '4,096',
        'original': '65,536',
        'rationale': 'Reduced due to limited compute. Still maintains diverse negative samples.',
        'position': (1, 6),
        'color': '#3498db'
    },
    {
        'title': 'Momentum (m)',
        'value': '0.999',
        'original': '0.999',
        'rationale': 'Slow-moving average for stable key encoder updates.',
        'position': (7.5, 6),
        'color': '#2ecc71'
    },
    {
        'title': 'Temperature (T)',
        'value': '0.07',
        'original': '0.07',
        'rationale': 'Controls softmax sharpness in contrastive loss.',
        'position': (1, 3.5),
        'color': '#e74c3c'
    },
    {
        'title': 'Projection Dim',
        'value': '128',
        'original': '128',
        'rationale': 'MLP projection head output dimensionality.',
        'position': (7.5, 3.5),
        'color': '#9b59b6'
    },
]

for comp in components:
    x, y = comp['position']

    # Component box
    box = FancyBboxPatch((x, y), 5.5, 2,
                         boxstyle="round,pad=0.1",
                         edgecolor=comp['color'],
                         facecolor=f"{comp['color']}20",
                         linewidth=2.5)
    ax.add_patch(box)

    # Title
    ax.text(x + 2.75, y + 1.7, comp['title'],
            fontsize=12, fontweight='bold', ha='center', color=comp['color'])

    # Value
    ax.text(x + 0.3, y + 1.2, f"Our Value: {comp['value']}", fontsize=10, fontweight='bold')
    ax.text(x + 0.3, y + 0.85, f"Original Paper: {comp['original']}", fontsize=9, style='italic', color='#7f8c8d')

    # Rationale
    ax.text(x + 0.3, y + 0.4, comp['rationale'], fontsize=8.5, style='italic', wrap=True)

# Training configuration
train_box = FancyBboxPatch((1, 0.5), 12, 2,
                           boxstyle="round,pad=0.15",
                           edgecolor='#f39c12', facecolor='#fff3cd', linewidth=2.5)
ax.add_patch(train_box)

ax.text(7, 2.1, 'Training Configuration', fontsize=13, fontweight='bold', ha='center', color='#f39c12')

train_params = [
    ('Epochs', '200', 'Extended training for domain-specific features'),
    ('Batch Size', '256', 'Large batches critical for contrastive learning'),
    ('Optimizer', 'SGD', 'lr=0.001, momentum=0.9, weight decay=1e-4'),
    ('Scheduler', 'Cosine Annealing', 'Smooth LR decay to 0 over 200 epochs'),
    ('Backbone', 'ResNet-50', 'Initialized with ImageNet weights'),
    ('Augmentation', 'MoCo v2 standard', 'Random crop (256×256), color jitter, Gaussian blur'),
]

y = 1.6
for param, value, detail in train_params:
    ax.text(1.5, y, f'{param}:', fontsize=9, fontweight='bold')
    ax.text(3, y, value, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax.text(5, y, f'({detail})', fontsize=8, style='italic', color='#7f8c8d')
    y -= 0.23

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'moco_hyperparameters.png', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'moco_hyperparameters.pdf', bbox_inches='tight')
print(f"   [OK] Saved MoCo v2 hyperparameters breakdown")
plt.close()

print("\n" + "="*80)
print("[OK] ALL COMPREHENSIVE VISUALIZATIONS GENERATED SUCCESSFULLY!")
print(f"[OK] Saved to: {OUTPUT_DIR}")
print("="*80)
print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.glob('*.png')):
    print(f"  - {file.name}")
