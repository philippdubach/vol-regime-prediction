#!/usr/bin/env python3
"""Comprehensive sanity check for Alpha Vantage options data."""

import pandas as pd
import numpy as np

def main():
    df = pd.read_parquet('data/raw/alpha_vantage/SPY_options_history.parquet')

    print('='*70)
    print('SANITY CHECKS & LOGICAL ERROR DETECTION')
    print('='*70)

    # 1. IV Sanity Checks
    print('\n🔍 IMPLIED VOLATILITY SANITY CHECKS')
    iv_cols = ['AV_ATM_IV', 'AV_VW_IV', 'AV_CALL_IV_MEAN', 'AV_PUT_IV_MEAN']
    for col in iv_cols:
        s = df[col]
        neg = (s < 0).sum()
        extreme = (s > 1.5).sum()
        print(f'  {col}: Negative values: {neg}, >150%: {extreme}')
        if s.max() > 0.5:
            print(f'    ⚠️ Max IV = {s.max():.4f} ({s.max()*100:.1f}%) - possible high vol period')
            print(f'    Date: {df.index[s.argmax()]}')

    # 2. Volume Sanity Checks
    print('\n🔍 VOLUME SANITY CHECKS')
    vol_diff = abs(df['AV_CALL_VOLUME'] + df['AV_PUT_VOLUME'] - df['AV_TOTAL_VOLUME'])
    print(f'  Volume consistency (Call+Put == Total): Max diff = {vol_diff.max():.0f}')
    
    pcr = df['AV_PUT_CALL_RATIO_VOL']
    print(f'  Put/Call Ratio Vol: Min={pcr.min():.2f}, Max={pcr.max():.2f}')
    extreme_pcr = ((pcr < 0.3) | (pcr > 4)).sum()
    print(f'  Extreme P/C ratios (<0.3 or >4): {extreme_pcr}')

    # 3. Greeks Sanity Checks
    print('\n🔍 GREEKS SANITY CHECKS')
    delta = df['AV_NET_DELTA']
    print(f'  Net Delta: Mean={delta.mean():.4f}, Std={delta.std():.4f}')
    print(f'  Net Delta Range: [{delta.min():.4f}, {delta.max():.4f}]')
    
    gamma = df['AV_TOTAL_GAMMA']
    neg_gamma = (gamma < 0).sum()
    print(f'  Negative Gamma values: {neg_gamma}')

    # 4. IV Skew Checks
    print('\n🔍 IV SKEW SANITY CHECKS')
    skew25 = df['AV_IV_SKEW_25D']
    skew10 = df['AV_IV_SKEW_10D']
    neg_skew25 = (skew25 < 0).sum()
    neg_skew10 = (skew10 < 0).sum()
    print(f'  Negative 25-delta skew: {neg_skew25} ({neg_skew25/len(df)*100:.1f}%)')
    print(f'  Negative 10-delta skew: {neg_skew10} ({neg_skew10/len(df)*100:.1f}%)')
    skew_violation = (skew10 < skew25 - 0.05).sum()
    print(f'  10-delta < 25-delta (by >5pp): {skew_violation} ({skew_violation/len(df)*100:.1f}%)')

    # 5. Term Structure Checks
    print('\n🔍 TERM STRUCTURE SANITY CHECKS')
    term_slope = df['AV_IV_TERM_SLOPE']
    backwardation = (term_slope < -0.1).sum()
    contango = (term_slope > 0.1).sum()
    print(f'  Backwardation (slope < -0.1): {backwardation} ({backwardation/len(df)*100:.1f}%)')
    print(f'  Contango (slope > 0.1): {contango} ({contango/len(df)*100:.1f}%)')

    # 6. Strike Price Sanity
    print('\n🔍 ATM STRIKE SANITY CHECKS')
    strike = df['AV_ATM_STRIKE']
    low_strike = (strike < 40).sum()
    high_strike = (strike > 700).sum()
    print(f'  Unreasonably low strikes (<40): {low_strike}')
    print(f'  Unreasonably high strikes (>700): {high_strike}')
    print(f'  Strike range: ${strike.min():.0f} to ${strike.max():.0f}')

    # 7. Date Continuity
    print('\n🔍 DATE CONTINUITY CHECKS')
    dates = pd.to_datetime(df.index)
    date_diffs = dates.diff().dropna().days
    unusual_gaps = ((date_diffs != 7) & (date_diffs != 14)).sum()
    max_gap = date_diffs.max()
    print(f'  Unusual gaps (not 7 or 14 days): {unusual_gaps}')
    print(f'  Max gap: {max_gap} days')

    # 8. Historic Event Verification
    print('\n🔍 HISTORIC EVENT VERIFICATION')
    covid_period = df.loc['2020-03-01':'2020-04-30']
    if len(covid_period) > 0:
        covid_max_iv = covid_period['AV_ATM_IV'].max()
        print(f'  COVID crash (Mar-Apr 2020) max ATM IV: {covid_max_iv:.1%}')
        if covid_max_iv > 0.5:
            print(f'    ✅ Expected: >50%, Got: {covid_max_iv:.1%}')
        else:
            print(f'    ⚠️ Unexpectedly low')

    volmageddon = df.loc['2018-02-01':'2018-02-28']
    if len(volmageddon) > 0:
        volmageddon_max_iv = volmageddon['AV_ATM_IV'].max()
        print(f'  Volmageddon (Feb 2018) max ATM IV: {volmageddon_max_iv:.1%}')

    gfc = df.loc['2008-09-01':'2008-12-31']
    if len(gfc) > 0:
        gfc_max_iv = gfc['AV_ATM_IV'].max()
        print(f'  Global Financial Crisis (Sep-Dec 2008) max ATM IV: {gfc_max_iv:.1%}')

    # 9. Correlation Checks
    print('\n🔍 CORRELATION SANITY CHECKS')
    # ATM IV and VW IV should be highly correlated
    iv_corr = df['AV_ATM_IV'].corr(df['AV_VW_IV'])
    print(f'  ATM IV vs VW IV correlation: {iv_corr:.4f}')
    if iv_corr < 0.8:
        print(f'    ⚠️ Unexpectedly low correlation')
    else:
        print(f'    ✅ Expected high correlation')
    
    # Put and Call IV should be correlated
    pc_iv_corr = df['AV_PUT_IV_MEAN'].corr(df['AV_CALL_IV_MEAN'])
    print(f'  Put IV vs Call IV correlation: {pc_iv_corr:.4f}')

    # 10. Data Completeness Summary
    print('\n📊 DATA QUALITY SUMMARY')
    total_features = len(df.columns)
    complete_features = (df.isnull().sum() == 0).sum()
    print(f'  Complete features: {complete_features}/{total_features}')
    print(f'  Total data points: {len(df)}')
    print(f'  Data quality score: {complete_features/total_features*100:.1f}%')
    
    print('\n' + '='*70)
    print('SANITY CHECK COMPLETE')
    print('='*70)

if __name__ == '__main__':
    main()
