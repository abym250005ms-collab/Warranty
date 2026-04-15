# =============================================================================
# WARRANTY COST FORECASTING — FIXED VERSION (MINIMAL EDITS)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings, os
warnings.filterwarnings('ignore')

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.makedirs('outputs/forecast_plots', exist_ok=True)

# =============================================================================
# CONSTANTS
# =============================================================================
FORECAST_MONTHS = 12
CI_WIDTH = 0.95

# =============================================================================
# 🔥 HELPER FUNCTION (ADDED)
# =============================================================================
def get_future_df(fc):
    return (
        fc[fc['ds'] > LAST_DATE]
        .sort_values('ds')
        .head(FORECAST_MONTHS)
        .reset_index(drop=True)
    )

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")

claims = pd.read_excel(
    'warranty_claim_dataset_v2_fixed.xlsx',
    sheet_name='Warranty Claims (Full)'
)

claims['claim_date'] = pd.to_datetime(dict(
    year=claims['Claim Date Year'],
    month=claims['Claim Date Month'],
    day=claims['Claim Date Day']
), errors='coerce')

# =============================================================================
# MONTHLY AGGREGATION
# =============================================================================
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

# =============================================================================
# TRAIN MODELS
# =============================================================================
all_forecasts = {}
all_metrics = {}

for variant in VARIANTS:

    print(f"\nProcessing {variant}")

    df = (
        monthly[monthly['model'] == variant][['ds', 'total_cost']]
        .rename(columns={'total_cost': 'y'})
        .sort_values('ds')
    )

    # OUTLIER HANDLING
    q = df['y'].quantile(0.97)
    df.loc[df['y'] > q, 'y'] = np.nan

    # LOG TRANSFORM
    df['y'] = np.log1p(df['y'])

    # MODEL
    model = Prophet(
        yearly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        interval_width=CI_WIDTH
    )

    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df)

    # FORECAST
    future = model.make_future_dataframe(periods=FORECAST_MONTHS, freq='MS')
    forecast = model.predict(future)

    # INVERSE TRANSFORM
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast[col] = np.expm1(forecast[col]).clip(lower=0)

    forecast['model'] = variant

    # METRICS
    merged = forecast.merge(
        monthly[monthly['model'] == variant][['ds', 'total_cost']],
        on='ds', how='left'
    )

    train = merged[merged['total_cost'].notna()]

    mae = mean_absolute_error(train['total_cost'], train['yhat'])
    rmse = np.sqrt(mean_squared_error(train['total_cost'], train['yhat']))
    mape = np.mean(np.abs((train['total_cost'] - train['yhat']) /
                          train['total_cost'].replace(0, np.nan))) * 100

    all_metrics[variant] = {'MAPE': round(mape,2)}
    all_forecasts[variant] = forecast

    print(f"MAPE={mape:.2f}%")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\nFINAL FORECAST SUMMARY\n")

grand_total = 0
summary = []

for variant in VARIANTS:

    fc = all_forecasts[variant]

    # 🔥 FIX APPLIED HERE
    future_df = get_future_df(fc)

    total = future_df['yhat'].sum()
    low = future_df['yhat_lower'].sum()
    high = future_df['yhat_upper'].sum()

    summary.append((variant, total, low, high))
    grand_total += total

    print(f"{variant:<15} ₹{total:,.0f}  [{low:,.0f} — {high:,.0f}]")

print(f"\nTOTAL FORECAST: ₹{grand_total:,.0f}")

# =============================================================================
# BAR CHART (FIXED)
# =============================================================================
names = [x[0] for x in summary]
totals = np.array([x[1] for x in summary])
lows = np.array([x[2] for x in summary])
highs = np.array([x[3] for x in summary])

# 🔥 SAFE ERROR
lower_err = np.maximum(totals - lows, 0)
upper_err = np.maximum(highs - totals, 0)

plt.figure(figsize=(10,6))
plt.barh(names, totals, xerr=[lower_err, upper_err])
plt.title("12-Month Warranty Forecast")
plt.xlabel("Cost (₹)")
plt.savefig("outputs/forecast_plots/final_bar.png")
plt.close()

print("\nDone — all outputs generated successfully")
