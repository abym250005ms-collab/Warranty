#!/usr/bin/env python3
# =============================================================================
# WARRANTY COST FORECASTING - Prophet + ARIMA (v2 - FIXED)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
from datetime import timedelta

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
os.makedirs('outputs/forecast_plots', exist_ok=True)

# =============================================================================
# CONFIG
# =============================================================================
FORECAST_MONTHS = 12
CI_WIDTH = 0.95

print("=" * 70)
print("WARRANTY COST FORECASTING - Prophet + ARIMA")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading data...")

claims = pd.read_excel(
    'warranty_claim_dataset_v2_fixed.xlsx',
    sheet_name='Warranty Claims (Full)'
)

claims['claim_date'] = pd.to_datetime(dict(
    year=claims['Claim Date Year'],
    month=claims['Claim Date Month'],
    day=claims['Claim Date Day']
), errors='coerce')

print(f"  Loaded {len(claims):,} claims")

# =============================================================================
# MONTHLY AGGREGATION
# =============================================================================
print("\nMonthly aggregation...")

monthly = (
    claims
    .groupby([claims['claim_date'].dt.to_period('M'), 'Model Variant'])
    .agg(total_cost=('Total Claim Cost Inr', 'sum'))
    .reset_index()
)

monthly.columns = ['period', 'model', 'total_cost']
monthly['ds'] = monthly['period'].dt.to_timestamp()

VARIANTS = sorted(monthly['model'].unique())
LAST_DATE = monthly['ds'].max()

print(f"  Models: {len(VARIANTS)}")
print(f"  Date range: {monthly['ds'].min().date()} to {monthly['ds'].max().date()}")

# =============================================================================
# TRAIN MODELS
# =============================================================================
print("\nTraining models per variant...\n")

all_forecasts = {}
all_metrics = {}

for variant in VARIANTS:
    print(f"  {variant}...", end=" ")

    df = (
        monthly[monthly['model'] == variant][['ds', 'total_cost']]
        .rename(columns={'total_cost': 'y'})
        .sort_values('ds')
    )

    # Outlier handling
    q = df['y'].quantile(0.97)
    df.loc[df['y'] > q, 'y'] = np.nan

    # Log transform
    df['y'] = np.log1p(df['y'])

    # Train Prophet
    model = Prophet(
        yearly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        interval_width=CI_WIDTH
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df)

    # Forecast
    future = model.make_future_dataframe(periods=FORECAST_MONTHS, freq='MS')
    forecast = model.predict(future)

    # Inverse transform
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast[col] = np.expm1(forecast[col]).clip(lower=0)

    # Get future forecasts
    forecast_future = forecast[forecast['ds'] > LAST_DATE].head(FORECAST_MONTHS)

    forecast['model'] = variant
    all_forecasts[variant] = forecast

    # Metrics on training data
    merged = forecast.merge(
        monthly[monthly['model'] == variant][['ds', 'total_cost']],
        on='ds', how='left'
    )
    train = merged[merged['total_cost'].notna()]

    if len(train) > 0:
        mae = mean_absolute_error(train['total_cost'], train['yhat'])
        rmse = np.sqrt(mean_squared_error(train['total_cost'], train['yhat']))
        mape = np.mean(np.abs((train['total_cost'] - train['yhat']) / 
                              train['total_cost'].replace(0, np.nan))) * 100
    else:
        mape = np.nan

    all_metrics[variant] = {'MAPE': round(mape, 2)}

    print(f"MAPE={mape:.1f}%")

# =============================================================================
# AGGREGATED FORECAST
# =============================================================================
print("\n" + "=" * 70)
print("AGGREGATED FORECAST (All Variants)")
print("=" * 70)

grand_total = 0
grand_lower = 0
grand_upper = 0
summary = []

for variant in VARIANTS:
    fc = all_forecasts[variant]
    
    # Future forecast (safely)
    future_fc = fc[fc['ds'] > LAST_DATE].head(FORECAST_MONTHS)
    
    if len(future_fc) > 0:
        total = future_fc['yhat'].sum()
        lower = future_fc['yhat_lower'].sum()
        upper = future_fc['yhat_upper'].sum()
    else:
        total = lower = upper = 0

    summary.append((variant, total, lower, upper))
    grand_total += total
    grand_lower += lower
    grand_upper += upper

print(f"\nForecast for next {FORECAST_MONTHS} months:\n")
for variant, total, lower, upper in summary:
    print(f"  {variant:<20} INR {total:>12,.0f}  [{lower:>12,.0f} — {upper:>12,.0f}]")

print(f"\n  {'TOTAL':<20} INR {grand_total:>12,.0f}  [{grand_lower:>12,.0f} — {grand_upper:>12,.0f}]")

# =============================================================================
# EXPORT SUMMARY
# =============================================================================
summary_df = pd.DataFrame(summary, columns=['Variant', 'Forecast', 'Lower_CI', 'Upper_CI'])
summary_df.to_csv('outputs/forecast_summary.csv', index=False, encoding='utf-8')
print(f"\n  Saved: forecast_summary.csv")

# =============================================================================
# PLOT: BAR CHART
# =============================================================================
print("\nGenerating plots...")

names = [x[0] for x in summary]
totals = np.array([x[1] for x in summary])
lows = np.array([x[2] for x in summary])
highs = np.array([x[3] for x in summary])

# Safe error bars
lower_err = np.maximum(totals - lows, 0)
upper_err = np.maximum(highs - totals, 0)

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(names, totals, xerr=[lower_err, upper_err], capsize=5, color='steelblue', edgecolor='navy')
ax.set_xlabel('12-Month Cost Forecast (INR)', fontweight='bold')
ax.set_title('Warranty Cost Forecast by Model Variant', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Format x-axis as currency
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'INR {x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig('outputs/forecast_plots/forecast_bar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: forecast_bar_chart.png")

# =============================================================================
# PLOT: TIME SERIES WITH FORECAST
# =============================================================================
fig, axes = plt.subplots(len(VARIANTS), 1, figsize=(14, 3*len(VARIANTS)))
if len(VARIANTS) == 1:
    axes = [axes]

for idx, variant in enumerate(VARIANTS):
    ax = axes[idx]
    fc = all_forecasts[variant]
    
    # Historical
    hist = monthly[monthly['model'] == variant].sort_values('ds')
    ax.plot(hist['ds'], hist['total_cost'], 'o-', color='navy', label='Historical', linewidth=2, markersize=6)
    
    # Forecast
    future_fc = fc[fc['ds'] > LAST_DATE].head(FORECAST_MONTHS)
    ax.plot(future_fc['ds'], future_fc['yhat'], 's--', color='darkorange', label='Forecast', linewidth=2, markersize=6)
    ax.fill_between(future_fc['ds'], future_fc['yhat_lower'], future_fc['yhat_upper'], alpha=0.2, color='darkorange')
    
    ax.axvline(LAST_DATE, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Forecast Start')
    ax.set_title(f'{variant} - 12-Month Forecast', fontweight='bold')
    ax.set_ylabel('Cost (INR)', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if idx == len(VARIANTS) - 1:
        ax.set_xlabel('Date', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/forecast_plots/forecast_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: forecast_timeseries.png")

# =============================================================================
# PLOT: PIE CHART (FORECAST SHARE)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
wedges, texts, autotexts = ax.pie(totals, labels=names, autopct='%1.1f%%', colors=colors, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax.set_title('Forecast Cost Share by Model Variant', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/forecast_plots/forecast_pie_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: forecast_pie_chart.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FORECAST COMPLETE")
print("=" * 70)
print(f"  Total 12-Month Forecast: INR {grand_total:,.0f}")
print(f"  Confidence Range: [{grand_lower:,.0f} - {grand_upper:,.0f}]")
print(f"  Models: {len(VARIANTS)}")
print(f"  Plots: 3 (bar, timeseries, pie)")
print("=" * 70)
