import pandas as pd
from pathlib import Path

UAVID_CLS = [
    'Background clutter', # 0
    'Building',           # 1
    'Road',               # 2
    'Tree',               # 3
    'Low vegetation',     # 4
    'Moving car',         # 5
    'Static car',         # 6
    'Human',              # 7
]

CLS_RENAME = {
    'Background clutter': 'Clutter',
    'Building': 'Buildings',
    'Road': 'Road',
    'Tree': 'Tree',
    'Low vegetation': 'Vegetation',
    'Moving car': 'Moving Car',
    'Static car': 'Static Car',
    'Human': 'Human',
}

ROOT = Path(__file__).resolve().parent.parent.parent    
RESULTS_DIR = ROOT / 'results'

rows = []
for metrics_file in RESULTS_DIR.glob('*/test_metrics.csv'):
    model_name = metrics_file.parent.name
    df = pd.read_csv(metrics_file)

    df_classes = df[df['class_id'] >= 0].copy()
    df_mean = df[df['class_id'] == -1].copy()

    mean_row = df_mean.iloc[0]
   
    row = {}
    row['Model'] = model_name
    df_classes = df_classes.set_index('class_name')
    
    for cls_name in UAVID_CLS:
        if cls_name not in df_classes.index:
            raise ValueError(f'Class {cls_name} not found in metrics for model {model_name}')
        row[f'{CLS_RENAME[cls_name]}'] = df_classes.loc[cls_name, 'IoU'] * 100.0
    row['mIoU'] = mean_row['IoU'] * 100.0
    row['F1'] = mean_row['F1'] * 100.0
    row['OA'] = mean_row['OA'] * 100.0

    rows.append(row)

df_summary = pd.DataFrame(rows)
df_summary = df_summary.sort_values('mIoU', ascending=False)

out_csv = RESULTS_DIR / 'summary_metrics.csv'
df_summary.to_csv(out_csv, index=False)
print(f'Saved summary metrics to {out_csv}')