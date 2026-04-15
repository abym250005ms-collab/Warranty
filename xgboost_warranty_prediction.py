#!/usr/bin/env python3
# =============================================================================
# WARRANTY CLAIM PREDICTION — Full E2E Pipeline (v5 — Final)
# =============================================================================
# Algorithm  : XGBoost Binary Classifier
# Target     : Will vehicle file a warranty claim in next 90 days? (0 / 1)
# Dataset    : warranty_claim_dataset_v2_fixed.xlsx
#
# Outputs (all in ./outputs/)
# ────────────────────────────
# 01_roc_pr_curves.png
# 02_confusion_matrix_highrecall.png
# 03_confusion_matrix_balanced.png
# 04_threshold_tradeoff.png
# 05_feature_importance.png
# 06_probability_distribution.png
# 07_shap_beeswarm.png
# 08_shap_bar.png
# 09_shap_waterfall_high_risk.png
# 10_shap_dependence_top_feature.png
# 11_risk_distribution.png
# 12_risk_by_model.png
# 13_cost_saving_analysis.png
# vehicle_risk_scores.csv
# high_risk_vehicles.csv
# feature_importance.csv
# business_summary_report.txt
# xgboost_warranty_model.pkl
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
import pickle
from datetime import timedelta
import sys

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# Color Palette
C_BLUE   = '#1F4E79'
C_LIGHT  = '#4BACC6'
C_RED    = '#C00000'
C_GREEN  = '#375623'
C_GOLD   = '#FFC000'
C_ORANGE = '#ED7D31'
C_PURPLE = '#7B2D8B'
BG       = '#F7F9FC'
GRID_CLR = '#E0E0E0'

plt.rcParams.update({
    'axes.facecolor'    : BG,
    'figure.facecolor'  : BG,
    'axes.grid'         : True,
    'grid.color'        : GRID_CLR,
    'grid.linewidth'    : 0.5,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
})

# =============================================================================
# STEP 1 - LOAD DATA
# =============================================================================
print("=" * 70)
print("STEP 1 - Loading data from warranty_claim_dataset_v2_fixed.xlsx")
print("=" * 70)

FILE = 'warranty_claim_dataset_v2_fixed.xlsx'

try:
    claims_raw = pd.read_excel(FILE, sheet_name='Warranty Claims (Full)')
    vehicles   = pd.read_excel(FILE, sheet_name='Vehicles')
except FileNotFoundError:
    print(f"ERROR: File '{FILE}' not found!")
    sys.exit(1)
except Exception as e:
    print(f"ERROR loading file: {str(e)}")
    sys.exit(1)

def make_date(df, day_col, mon_col, yr_col):
    """Parse date from separate day/month/year columns"""
    try:
        return pd.to_datetime(
            dict(year=df[yr_col], month=df[mon_col], day=df[day_col]),
            errors='coerce'
        )
    except:
        return pd.Series([pd.NaT] * len(df))

# Parse dates
claims_raw['claim_date']   = make_date(claims_raw, 'Claim Date Day', 'Claim Date Month', 'Claim Date Year')
claims_raw['sale_date']    = make_date(claims_raw, 'Sale Date Day', 'Sale Date Month', 'Sale Date Year')
claims_raw['failure_date'] = make_date(claims_raw, 'Failure Date Day', 'Failure Date Month', 'Failure Date Year')
claims_raw['warranty_end'] = make_date(claims_raw, 'Warranty End Date Day', 'Warranty End Date Month', 'Warranty End Date Year')

vehicles['sale_date']         = pd.to_datetime(vehicles['Sale Date'], errors='coerce')
vehicles['warranty_end_date'] = pd.to_datetime(vehicles['Warranty End Date'], errors='coerce')
vehicles['manufacture_date']  = pd.to_datetime(vehicles['Manufacture Date'], errors='coerce')

print(f"  Claims loaded   : {len(claims_raw):,} rows")
print(f"  Vehicles loaded : {len(vehicles):,} rows")
print(f"  Claim date range: {claims_raw['claim_date'].min().date()} -> {claims_raw['claim_date'].max().date()}")
print(f"  Vehicles with claims : {claims_raw['Vehicle Id'].nunique():,}")
print(f"  Vehicles without any : {len(vehicles) - claims_raw['Vehicle Id'].nunique():,}")

# Prediction anchor
REFERENCE_DATE = pd.Timestamp('2024-10-01')
HORIZON_DAYS   = 90
WINDOW_END     = REFERENCE_DATE + timedelta(days=HORIZON_DAYS)

AVG_CLAIM_COST_INR = claims_raw['Total Claim Cost Inr'].mean()
print(f"\n  Average claim cost (from data): INR {AVG_CLAIM_COST_INR:,.0f}")

# =============================================================================
# STEP 2 - LEAKAGE-FREE FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2 - Feature engineering (leakage-free)")
print("=" * 70)

claims_past = claims_raw[claims_raw['claim_date'] < REFERENCE_DATE].copy()
print(f"  Past claims used for features : {len(claims_past):,}")
print(f"  Future claims (target only)   : {len(claims_raw) - len(claims_past):,}")

crit_map = {'High': 3, 'Medium': 2, 'Low': 1}
claims_past['criticality_score'] = claims_past['Criticality'].map(crit_map).fillna(0)

# Full history aggregation
agg = claims_past.groupby('Vehicle Id').agg(
    total_claims             = ('Claim Id', 'count'),
    repeat_claim_flag        = ('Repeat Claim Flag', 'max'),
    first_claim_date         = ('claim_date', 'min'),
    last_claim_date          = ('claim_date', 'max'),
    avg_months_in_service    = ('Months In Service', 'mean'),
    total_claim_cost         = ('Total Claim Cost Inr', 'sum'),
    avg_claim_cost           = ('Total Claim Cost Inr', 'mean'),
    max_claim_cost           = ('Total Claim Cost Inr', 'max'),
    total_parts_cost         = ('Parts Cost Inr', 'sum'),
    total_labour_cost        = ('Labour Cost Inr', 'sum'),
    max_odometer             = ('Odometer At Failure Km', 'max'),
    avg_odometer_per_claim   = ('Odometer At Failure Km', 'mean'),
    avg_repair_days          = ('Repair Duration Days', 'mean'),
    max_repair_days          = ('Repair Duration Days', 'max'),
    total_replace_count      = ('Repair Type', lambda x: (x == 'Replace').sum()),
    unique_components_failed = ('Component Id', 'nunique'),
    unique_subsystems_failed = ('Subsystem', 'nunique'),
    max_criticality_score    = ('criticality_score', 'max'),
    avg_criticality_score    = ('criticality_score', 'mean'),
    powertrain_failures      = ('Subsystem', lambda x: (x == 'Powertrain').sum()),
    thermal_failures         = ('Subsystem', lambda x: (x == 'Thermal').sum()),
    electrical_failures      = ('Subsystem', lambda x: (x == 'Electrical').sum()),
    avg_ambient_temp         = ('Ambient Temp Celsius', 'mean'),
    max_ambient_temp         = ('Ambient Temp Celsius', 'max'),
    claim_dealer_capacity    = ('Claim Dealer Capacity Score', 'mean'),
    unique_dealers_used      = ('Dealer Id', 'nunique'),
    avg_daily_km             = ('Avg Daily Km', 'first'),
    battery_capacity_kwh     = ('Battery Capacity Kwh', 'first'),
    model_variant            = ('Model Variant', 'first'),
    motor_type               = ('Motor Type', 'first'),
    vehicle_zone             = ('Vehicle Zone', 'first'),
    use_case                 = ('Use Case', 'first'),
    sale_date                = ('sale_date', 'first'),
    warranty_end             = ('warranty_end', 'first'),
    expected_life_km         = ('Expected Life Km', 'mean'),
).reset_index()

# Rolling window aggregations
def rolling_agg(months):
    cutoff = REFERENCE_DATE - pd.DateOffset(months=months)
    sub = claims_past[
        (claims_past['claim_date'] >= cutoff) &
        (claims_past['claim_date'] < REFERENCE_DATE - pd.Timedelta(days=30))
    ]
    return (sub.groupby('Vehicle Id')
               .agg(count=('Claim Id', 'count'), cost=('Total Claim Cost Inr', 'sum'))
               .rename(columns={'count': f'claims_{months}m', 'cost': f'cost_{months}m'}))

r3  = rolling_agg(3)
r6  = rolling_agg(6)
r12 = rolling_agg(12)

# Merge
all_vehicles = vehicles[['Vehicle Id', 'sale_date', 'warranty_end_date',
                         'Model Variant', 'Battery Capacity Kwh',
                         'Motor Type', 'Zone', 'Use Case', 'Avg Daily Km']].copy()
all_vehicles.columns = ['Vehicle Id', 'sale_date_v', 'warranty_end_v',
                        'model_variant_v', 'battery_kwh_v', 'motor_type_v',
                        'zone_v', 'use_case_v', 'avg_daily_km_v']

dataset = (all_vehicles
          .merge(agg, on='Vehicle Id', how='left')
          .merge(r3, on='Vehicle Id', how='left')
          .merge(r6, on='Vehicle Id', how='left')
          .merge(r12, on='Vehicle Id', how='left'))

# Fill zeros
zero_cols = [
    'total_claims', 'repeat_claim_flag', 'total_claim_cost', 'avg_claim_cost',
    'max_claim_cost', 'total_parts_cost', 'total_labour_cost',
    'max_odometer', 'avg_odometer_per_claim', 'avg_repair_days', 'max_repair_days',
    'total_replace_count', 'unique_components_failed', 'unique_subsystems_failed',
    'max_criticality_score', 'avg_criticality_score',
    'powertrain_failures', 'thermal_failures', 'electrical_failures',
    'avg_ambient_temp', 'max_ambient_temp', 'avg_months_in_service',
    'claims_3m', 'cost_3m', 'claims_6m', 'cost_6m', 'claims_12m', 'cost_12m',
]
dataset[zero_cols] = dataset[zero_cols].fillna(0)

for src, dst in [
    ('avg_daily_km_v', 'avg_daily_km'), ('battery_kwh_v', 'battery_capacity_kwh'),
    ('model_variant_v', 'model_variant'), ('motor_type_v', 'motor_type'),
    ('zone_v', 'vehicle_zone'), ('use_case_v', 'use_case'),
    ('sale_date_v', 'sale_date'), ('warranty_end_v', 'warranty_end'),
]:
    dataset[dst] = dataset[dst].fillna(dataset[src])

# Derived features
dataset['vehicle_age_days']        = (REFERENCE_DATE - dataset['sale_date']).dt.days.clip(lower=0)
dataset['vehicle_age_years']       = (dataset['vehicle_age_days'] / 365.25).round(3)
dataset['warranty_days_remaining'] = (dataset['warranty_end'] - REFERENCE_DATE).dt.days
dataset['is_in_warranty']          = (dataset['warranty_days_remaining'] > 0).astype(int)
dataset['warranty_expiring_soon']  = (
    (dataset['warranty_days_remaining'] > 0) &
    (dataset['warranty_days_remaining'] <= 90)
).astype(int)

dataset['odometer_utilisation'] = np.where(
    dataset['expected_life_km'] > 0,
    (dataset['max_odometer'] / dataset['expected_life_km']).clip(0, 1), 0
)
dataset['estimated_current_odo'] = dataset['avg_daily_km'] * dataset['vehicle_age_days']
dataset['claim_rate_per_year'] = np.where(
    dataset['vehicle_age_years'] > 0,
    dataset['total_claims'] / dataset['vehicle_age_years'], 0
)
dataset['cost_per_month'] = np.where(
    dataset['avg_months_in_service'] > 0,
    dataset['total_claim_cost'] / dataset['avg_months_in_service'], 0
)
dataset['days_since_last_claim'] = np.where(
    dataset['last_claim_date'].notna(),
    (REFERENCE_DATE - pd.to_datetime(dataset['last_claim_date'])).dt.days,
    dataset['vehicle_age_days']
)
dataset['claim_date_spread_days'] = np.where(
    dataset['first_claim_date'].notna() & dataset['last_claim_date'].notna(),
    (pd.to_datetime(dataset['last_claim_date']) -
     pd.to_datetime(dataset['first_claim_date'])).dt.days, 0
)
dataset['recent_claim_ratio']   = dataset['claims_6m'] / 6.0
dataset['is_high_usage']        = (dataset['avg_daily_km'] > dataset['avg_daily_km'].quantile(0.75)).astype(int)
dataset['has_repeat_component'] = (dataset['repeat_claim_flag'] > 0).astype(int)

print(f"  Feature matrix shape: {dataset.shape}")
print(f"  Vehicles with >=1 past claim: {(dataset['total_claims'] > 0).sum():,}")
print(f"  Vehicles with 0 past claims: {(dataset['total_claims'] == 0).sum():,}")

# =============================================================================
# STEP 3 - TARGET VARIABLE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3 - Target variable")
print("=" * 70)

future_claimers = set(
    claims_raw.loc[
        (claims_raw['claim_date'] >= REFERENCE_DATE) &
        (claims_raw['claim_date'] < WINDOW_END),
        'Vehicle Id'
    ].unique()
)
dataset['target'] = dataset['Vehicle Id'].isin(future_claimers).astype(int)

n_pos = dataset['target'].sum()
n_neg = (dataset['target'] == 0).sum()
print(f"  Prediction window: {REFERENCE_DATE.date()} -> {WINDOW_END.date()}")
print(f"  Will claim (1)   : {n_pos:,} ({n_pos/len(dataset)*100:.1f}%)")
print(f"  Won't claim (0)  : {n_neg:,} ({n_neg/len(dataset)*100:.1f}%)")

# =============================================================================
# STEP 4 - FEATURE PREPARATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4 - Preparing features")
print("=" * 70)

NUMERIC_FEATURES = [
    'total_claims', 'total_claim_cost', 'avg_claim_cost', 'max_claim_cost',
    'total_parts_cost', 'total_labour_cost', 'avg_repair_days', 'max_repair_days',
    'total_replace_count', 'unique_components_failed', 'unique_subsystems_failed',
    'max_criticality_score', 'avg_criticality_score',
    'powertrain_failures', 'thermal_failures', 'electrical_failures',
    'max_odometer', 'avg_odometer_per_claim', 'estimated_current_odo',
    'odometer_utilisation', 'avg_daily_km', 'battery_capacity_kwh',
    'vehicle_age_days', 'vehicle_age_years', 'avg_months_in_service',
    'warranty_days_remaining', 'is_in_warranty', 'warranty_expiring_soon',
    'is_high_usage', 'has_repeat_component',
    'claim_rate_per_year', 'cost_per_month',
    'days_since_last_claim', 'claim_date_spread_days',
    'avg_ambient_temp', 'max_ambient_temp',
    'claim_dealer_capacity', 'unique_dealers_used',
    'claims_3m', 'cost_3m', 'claims_6m', 'cost_6m', 'claims_12m', 'cost_12m',
    'recent_claim_ratio',
]

CATEGORICAL_FEATURES = ['model_variant', 'motor_type', 'vehicle_zone', 'use_case']

label_encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    dataset[col] = dataset[col].fillna('Unknown')
    dataset[col + '_enc'] = le.fit_transform(dataset[col].astype(str))
    label_encoders[col] = le

ENCODED_CAT   = [c + '_enc' for c in CATEGORICAL_FEATURES]
MODEL_FEATURES = NUMERIC_FEATURES + ENCODED_CAT

X = dataset[MODEL_FEATURES].fillna(0)
y = dataset['target']

scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"  Features: {len(MODEL_FEATURES)}")
print(f"  Total samples: {len(X):,}")
print(f"  Positive class: {y.mean()*100:.1f}%")
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

# =============================================================================
# STEP 5 - STRATIFIED SPLIT
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5 - Stratified train/val/test split")
print("=" * 70)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15 / 0.85, random_state=42, stratify=y_trainval
)

print(f"  Train: {len(X_train):,} samples ({y_train.mean()*100:.1f}% positive)")
print(f"  Val  : {len(X_val):,} samples ({y_val.mean()*100:.1f}% positive)")
print(f"  Test : {len(X_test):,} samples ({y_test.mean()*100:.1f}% positive)")

# =============================================================================
# STEP 6 - TRAIN XGBOOST
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6 - Training XGBoost classifier")
print("=" * 70)

model = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
    gamma=0.05, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc', early_stopping_rounds=15,
    random_state=42, n_jobs=-1, verbosity=0,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

best_iter = model.best_iteration
val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
print(f"  Best iteration: {best_iter}")
print(f"  Val AUC: {val_auc:.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    XGBClassifier(n_estimators=max(best_iter, 10), max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                  gamma=0.05, reg_alpha=0.1, reg_lambda=1.0,
                  scale_pos_weight=scale_pos_weight,
                  random_state=42, n_jobs=-1, verbosity=0),
    X, y, cv=cv, scoring='roc_auc', n_jobs=-1
)
print(f"  5-Fold CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# =============================================================================
# STEP 7 - THRESHOLD ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7 - Threshold analysis")
print("=" * 70)

y_pred_proba = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)
test_ap = average_precision_score(y_test, y_pred_proba)

prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_test, y_pred_proba)

f1_curve = np.where(
    (prec_curve[:-1] + rec_curve[:-1]) > 0,
    2 * prec_curve[:-1] * rec_curve[:-1] / (prec_curve[:-1] + rec_curve[:-1]),
    0
)

THRESH_HIGH_RECALL = 0.30
THRESH_BALANCED = thresh_curve[np.argmax(f1_curve)]

print(f"  Test AUC: {test_auc:.4f}")
print(f"  Test Avg Precision: {test_ap:.4f}")
print(f"  High-recall threshold: {THRESH_HIGH_RECALL:.2f}")
print(f"  Balanced threshold: {THRESH_BALANCED:.2f}")

def eval_threshold(threshold, label):
    preds = (y_pred_proba >= threshold).astype(int)
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f = f1_score(y_test, preds, zero_division=0)
    print(f"\n  [{label}] threshold={threshold:.2f}")
    print(f"    Precision={p:.2f}, Recall={r:.2f}, F1={f:.2f}")
    print(classification_report(y_test, preds, target_names=['No Claim', 'Will Claim']))
    return preds, p, r, f

preds_hr, p_hr, r_hr, f_hr = eval_threshold(THRESH_HIGH_RECALL, "HIGH RECALL")
preds_bal, p_bal, r_bal, f_bal = eval_threshold(THRESH_BALANCED, "BALANCED")

# =============================================================================
# STEP 8 - EVALUATION PLOTS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8 - Generating evaluation plots")
print("=" * 70)

# Plot 01: ROC + PR curves
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(fpr, tpr, color=C_BLUE, lw=2, label=f'AUC = {test_auc:.4f}')
ax1.plot([0, 1], [0, 1], '--', color='grey', lw=1)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve', fontweight='bold')
ax1.legend()

ax2.plot(rec_curve, prec_curve, color=C_RED, lw=2, label=f'AP = {test_ap:.4f}')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve', fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.savefig('outputs/01_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_roc_pr_curves.png")

# Plot 02: Confusion matrix - high recall
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, preds_hr),
                       display_labels=['No Claim', 'Will Claim']).plot(
    ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix - High Recall (t={THRESH_HIGH_RECALL:.2f})', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/02_confusion_matrix_highrecall.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_confusion_matrix_highrecall.png")

# Plot 03: Confusion matrix - balanced
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, preds_bal),
                       display_labels=['No Claim', 'Will Claim']).plot(
    ax=ax, colorbar=False, cmap='Oranges')
ax.set_title(f'Confusion Matrix - Balanced (t={THRESH_BALANCED:.2f})', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/03_confusion_matrix_balanced.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_confusion_matrix_balanced.png")

# Plot 04: Threshold tradeoff
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresh_curve, prec_curve[:-1], color=C_BLUE, lw=2, label='Precision')
ax.plot(thresh_curve, rec_curve[:-1], color=C_RED, lw=2, label='Recall')
ax.plot(thresh_curve, f1_curve, color=C_PURPLE, lw=2, label='F1 Score', linestyle='--')
ax.axvline(THRESH_HIGH_RECALL, color=C_RED, lw=1.2, linestyle=':', label=f'High-recall ({THRESH_HIGH_RECALL:.2f})')
ax.axvline(THRESH_BALANCED, color=C_PURPLE, lw=1.2, linestyle=':', label=f'Balanced ({THRESH_BALANCED:.2f})')
ax.set_xlabel('Decision threshold')
ax.set_ylabel('Score')
ax.set_title('Precision/Recall/F1 Tradeoff', fontweight='bold')
ax.legend(loc='center right')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('outputs/04_threshold_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_threshold_tradeoff.png")

# Plot 05: Feature importance
feat_imp = pd.Series(model.feature_importances_, index=MODEL_FEATURES)
top20 = feat_imp.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
colors = [C_RED if i < 5 else C_BLUE for i in range(len(top20))]
ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])
ax.set_xlabel('Feature importance (gain)')
ax.set_title('Top 20 Feature Importances', fontweight='bold')
red_patch = mpatches.Patch(color=C_RED, label='Top 5 drivers')
blue_patch = mpatches.Patch(color=C_BLUE, label='Other features')
ax.legend(handles=[red_patch, blue_patch])
plt.tight_layout()
plt.savefig('outputs/05_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_feature_importance.png")

# Plot 06: Probability distribution
all_proba = model.predict_proba(X)[:, 1]
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(all_proba[y == 0], bins=60, alpha=0.7, color=C_BLUE, label='No Claim')
ax.hist(all_proba[y == 1], bins=60, alpha=0.7, color=C_RED, label='Will Claim')
ax.axvline(THRESH_HIGH_RECALL, color=C_RED, linestyle='--', lw=1.2, label=f'High-recall ({THRESH_HIGH_RECALL:.2f})')
ax.axvline(THRESH_BALANCED, color=C_PURPLE, linestyle='--', lw=1.2, label=f'Balanced ({THRESH_BALANCED:.2f})')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('Count')
ax.set_title('Predicted Probability Distribution', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/06_probability_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_probability_distribution.png")

# =============================================================================
# STEP 9 - SHAP EXPLAINABILITY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9 - SHAP explainability")
print("=" * 70)

readable = [f.replace('_enc', '').replace('_', ' ').title() for f in MODEL_FEATURES]
X_sample = X.sample(n=min(500, len(X)), random_state=42).copy()
X_sample.columns = readable

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Plot 07: SHAP Beeswarm
fig, _ = plt.subplots(figsize=(10, 10))
shap.summary_plot(shap_values, X_sample, plot_type='dot', show=False, max_display=20)
plt.title('SHAP Beeswarm - Feature Impact', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/07_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 07_shap_beeswarm.png")

# Plot 08: SHAP Bar
fig, _ = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False, max_display=20)
plt.title('SHAP Feature Importance', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/08_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 08_shap_bar.png")

# Plot 09: SHAP Waterfall
high_idx = np.argmax(shap_values.sum(axis=1))
shap_expl = shap.Explanation(values=shap_values[high_idx], base_values=explainer.expected_value,
                             data=X_sample.iloc[high_idx], feature_names=readable)
fig, _ = plt.subplots(figsize=(12, 8))
shap.waterfall_plot(shap_expl, max_display=15, show=False)
plt.title('SHAP Waterfall - Highest Risk Vehicle', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/09_shap_waterfall_high_risk.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 09_shap_waterfall_high_risk.png")

# Plot 10: SHAP Dependence
top_idx = np.abs(shap_values).mean(axis=0).argmax()
fig, ax = plt.subplots(figsize=(9, 5))
shap.dependence_plot(top_idx, shap_values, X_sample, ax=ax, show=False, alpha=0.5)
ax.set_title(f'SHAP Dependence - {readable[top_idx]}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/10_shap_dependence_top_feature.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 10_shap_dependence_top_feature.png")

# =============================================================================
# STEP 10 - SCORE ALL VEHICLES
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10 - Scoring all vehicles")
print("=" * 70)

X_score = dataset[MODEL_FEATURES].fillna(0)
dataset['claim_probability'] = model.predict_proba(X_score)[:, 1]

dataset['risk_label'] = pd.cut(
    dataset['claim_probability'],
    bins=[0.00, 0.20, 0.50, 0.70, 1.00],
    labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
)
dataset['predicted_hr'] = (dataset['claim_probability'] >= THRESH_HIGH_RECALL).astype(int)
dataset['predicted_bal'] = (dataset['claim_probability'] >= THRESH_BALANCED).astype(int)

risk_counts = dataset['risk_label'].value_counts().sort_index()
print(f"\n  Risk distribution across {len(dataset):,} vehicles:")
for label, count in risk_counts.items():
    print(f"    {label}: {count:5,} ({count/len(dataset)*100:.1f}%)")

# Plot 11: Risk distribution
fig, ax = plt.subplots(figsize=(9, 5))
risk_order = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
risk_colors = [C_GREEN, C_GOLD, C_ORANGE, C_RED]
vals = [risk_counts.get(r, 0) for r in risk_order]
bars = ax.bar(risk_order, vals, color=risk_colors, edgecolor='white', width=0.6)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 15,
            f'{int(b.get_height()):,}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Number of vehicles')
ax.set_ylim(0, max(vals) * 1.15)
ax.set_title('Vehicle Risk Distribution - Next 90 Days', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/11_risk_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: 11_risk_distribution.png")

# Plot 12: Risk by model variant
risk_by_model = dataset.groupby('model_variant')['claim_probability'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(risk_by_model.index, risk_by_model.values,
       color=[C_RED if v > 0.5 else C_BLUE for v in risk_by_model.values],
       edgecolor='white', width=0.6)
ax.axhline(0.5, linestyle='--', color='grey', lw=1.2, label='50% threshold')
for i, (m, v) in enumerate(risk_by_model.items()):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Avg claim probability')
ax.set_title('Average Claim Risk by Model Variant', fontsize=12, fontweight='bold')
ax.legend()
ax.set_ylim(0, 0.7)
plt.tight_layout()
plt.savefig('outputs/12_risk_by_model.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 12_risk_by_model.png")

# =============================================================================
# STEP 11 - COST-SAVING ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 11 - Cost-saving analysis")
print("=" * 70)

PROACTIVE_COST_FRACTION = 0.30
INTERVENTION_SUCCESS_RATE = 0.60

scenarios = {
    f'High Recall\n(t={THRESH_HIGH_RECALL:.2f})': dataset['predicted_hr'],
    f'Balanced\n(t={THRESH_BALANCED:.2f})': dataset['predicted_bal'],
}

print(f"\n  Avg claim cost: INR {AVG_CLAIM_COST_INR:,.0f}")
print(f"  Proactive cost: {PROACTIVE_COST_FRACTION*100:.0f}% = INR {AVG_CLAIM_COST_INR * PROACTIVE_COST_FRACTION:,.0f}")
print(f"  Success rate: {INTERVENTION_SUCCESS_RATE*100:.0f}%\n")

cost_rows = []
for scenario_name, pred_col in scenarios.items():
    flagged = pred_col.sum()
    true_positives = ((pred_col == 1) & (dataset['target'] == 1)).sum()
    false_positives = ((pred_col == 1) & (dataset['target'] == 0)).sum()

    claims_prevented = true_positives * INTERVENTION_SUCCESS_RATE
    claims_avoided_cost = claims_prevented * AVG_CLAIM_COST_INR
    intervention_cost = flagged * AVG_CLAIM_COST_INR * PROACTIVE_COST_FRACTION
    net_saving = claims_avoided_cost - intervention_cost
    roi = (net_saving / intervention_cost * 100) if intervention_cost > 0 else 0

    cost_rows.append({
        'Scenario': scenario_name.replace('\n', ' '),
        'Flagged': flagged,
        'True Positives': true_positives,
        'False Positives': false_positives,
        'Claims Prevented': int(claims_prevented),
        'Gross Saving (INR)': int(claims_avoided_cost),
        'Intervention (INR)': int(intervention_cost),
        'Net Saving (INR)': int(net_saving),
        'ROI (%)': round(roi, 1),
    })

    print(f"  {scenario_name.replace(chr(10), ' ')}:")
    print(f"    Flagged: {flagged:,}")
    print(f"    True Positives: {true_positives:,}")
    print(f"    Claims Prevented: {int(claims_prevented):,}")
    print(f"    Net Saving: INR {net_saving:,.0f}")
    print(f"    ROI: {roi:.1f}%\n")

cost_df = pd.DataFrame(cost_rows)

# Plot 13: Cost-saving comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = ['Flagged', 'Net Saving (INR)', 'ROI (%)']
titles = ['Vehicles Flagged', 'Net Saving (INR)', 'ROI (%)']
bar_colors = [C_BLUE, C_GREEN, C_ORANGE]

for ax, metric, title, color in zip(axes, metrics, titles, bar_colors):
    bars = ax.bar([f"Mode {i+1}" for i in range(len(cost_df))], cost_df[metric].values,
                  color=color, edgecolor='white', width=0.5)
    for b in bars:
        val = b.get_height()
        if 'INR' in metric:
            label = f'INR {val:,.0f}'
        elif metric == 'ROI (%)':
            label = f'{val:.1f}%'
        else:
            label = f'{int(val):,}'
        ax.text(b.get_x() + b.get_width()/2, val * 1.02, label, ha='center', fontsize=9, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, max(cost_df[metric].values) * 1.25)

plt.suptitle('Cost-Saving Analysis by Operating Scenario', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/13_cost_saving_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 13_cost_saving_analysis.png")

# =============================================================================
# STEP 12 - EXPORT RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 12 - Exporting results")
print("=" * 70)

output_cols = [
    'Vehicle Id', 'model_variant', 'vehicle_zone', 'use_case',
    'avg_daily_km', 'battery_capacity_kwh', 'motor_type',
    'vehicle_age_years', 'total_claims', 'total_claim_cost',
    'claims_3m', 'claims_6m', 'claims_12m',
    'max_odometer', 'odometer_utilisation',
    'days_since_last_claim', 'claim_rate_per_year',
    'is_in_warranty', 'warranty_days_remaining', 'warranty_expiring_soon',
    'unique_components_failed', 'max_criticality_score',
    'claim_probability', 'risk_label', 'predicted_hr', 'predicted_bal', 'target'
]
results_df = dataset[output_cols].sort_values('claim_probability', ascending=False)
results_df.to_csv('outputs/vehicle_risk_scores.csv', index=False, encoding='utf-8')
print(f"  Saved: vehicle_risk_scores.csv ({len(results_df):,} rows)")

high_risk_df = results_df[results_df['risk_label'].isin(['Critical Risk', 'High Risk'])]
high_risk_df.to_csv('outputs/high_risk_vehicles.csv', index=False, encoding='utf-8')
print(f"  Saved: high_risk_vehicles.csv ({len(high_risk_df):,} vehicles)")

feat_df = pd.DataFrame({
    'Feature': readable,
    'Importance': model.feature_importances_,
}).sort_values('Importance', ascending=False)
feat_df.to_csv('outputs/feature_importance.csv', index=False, encoding='utf-8')
print(f"  Saved: feature_importance.csv ({len(feat_df)} features)")

cost_df.to_csv('outputs/cost_saving_scenarios.csv', index=False, encoding='utf-8')
print(f"  Saved: cost_saving_scenarios.csv")

with open('outputs/xgboost_warranty_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'features': MODEL_FEATURES,
        'label_encoders': label_encoders,
        'threshold_hr': THRESH_HIGH_RECALL,
        'threshold_bal': THRESH_BALANCED,
    }, f)
print(f"  Saved: xgboost_warranty_model.pkl")

# =============================================================================
# STEP 13 - BUSINESS SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 70)
print("STEP 13 - Business summary report")
print("=" * 70)

top5_features = feat_df['Feature'].head(5).tolist()

report_lines = [
    "=" * 75,
    "WARRANTY CLAIM PREDICTION - BUSINESS SUMMARY REPORT",
    "=" * 75,
    "",
    "1. OBJECTIVE",
    "   Predict vehicles that will file warranty claims in the next 90 days",
    "   to enable proactive maintenance and reduce claim costs.",
    "",
    "2. MODEL DETAILS",
    f"   Algorithm        : XGBoost Classifier",
    f"   Features         : {len(MODEL_FEATURES)}",
    f"   Training Samples : {len(X_train):,}",
    f"   Test Samples     : {len(X_test):,}",
    f"   Positive Class   : {y.mean()*100:.1f}%",
    "",
    "3. PERFORMANCE METRICS",
    f"   ROC-AUC (test)   : {test_auc:.4f}",
    f"   Avg Precision    : {test_ap:.4f}",
    f"   5-Fold CV AUC    : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}",
    "",
    "4. OPERATING SCENARIOS",
    f"",
    f"   Mode A - High Recall (threshold={THRESH_HIGH_RECALL:.2f})",
    f"   Precision={p_hr:.2f}, Recall={r_hr:.2f}, F1={f_hr:.2f}",
    f"",
    f"   Mode B - Balanced (threshold={THRESH_BALANCED:.2f})",
    f"   Precision={p_bal:.2f}, Recall={r_bal:.2f}, F1={f_bal:.2f}",
    "",
    "5. RISK DISTRIBUTION",
]
for label in ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']:
    c = risk_counts.get(label, 0)
    report_lines.append(f"   {label}: {c:5,} ({c/len(dataset)*100:.1f}%)")

report_lines += [
    "",
    "6. TOP 5 DRIVERS",
]
for i, f in enumerate(top5_features, 1):
    report_lines.append(f"   {i}. {f}")

report_lines += [
    "",
    "7. COST-SAVING ESTIMATES",
    f"   Average Claim Cost: INR {AVG_CLAIM_COST_INR:,.0f}",
    "",
]
for row in cost_rows:
    report_lines += [
        f"   {row['Scenario']}",
        f"     Flagged: {row['Flagged']:,}",
        f"     Net Saving: INR {row['Net Saving (INR)']:,}",
        f"     ROI: {row['ROI (%)']:.1f}%",
    ]

report_lines += [
    "",
    "=" * 75,
]

report_text = "\n".join(report_lines)
print(report_text)

with open('outputs/business_summary_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)
print("\n  Saved: business_summary_report.txt")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"  Version: v5 (Production Ready)")
print(f"  Test AUC: {test_auc:.4f}")
print(f"  Features: {len(MODEL_FEATURES)}")
print(f"  High/Critical Risk: {len(high_risk_df):,}")
print(f"  Outputs: 13 plots + 5 CSVs + report + model")
print("=" * 70)
