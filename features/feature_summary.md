# Feature Summary
────────────────────────────────────────────

## Identifiers
- product
- source_model
- entry_dt
- direction
- entry_type
- entry_price
- entry_atr

## Regime Features
- vr_16
- vr_30
- short_vr
- long_vr
- vr_score
- regime5_strong_mom
- regime5_mom
- regime5_neutral
- regime5_rev
- regime5_strong_rev

## Volatility Features
- atr
- atr_norm
- bbw
- bbw_percentile
- is_expansion
- walk_up
- walk_down
- rolling_std_20
- rolling_std_100

## Momentum & Price Position Features
- bb_position
- dist_ma_norm
- mom_5_norm
- mom_10_norm
- mom_20_norm

## Time Features
- hour_of_day
- day_of_week

Target Label
- y (1 = profitable trade, 0 = non-profitable trade)
