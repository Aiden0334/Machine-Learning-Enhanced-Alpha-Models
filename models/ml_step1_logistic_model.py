"""
═══════════════════════════════════════════════════════════════════
  ML Step 1 — 통합 + Logistic Regression
═══════════════════════════════════════════════════════════════════
[목적]
  v1 + v4 통합 데이터셋에 Logistic 적용
  ML의 선형 효과 측정 (sanity check)

[입력]
  ./ml/features.csv (feature_engineering.py 출력)

[학습 구조]
  IS (2018-2023): Train 80% + Val 20% (시간순)
    Train: 모델 학습
    Val: 임계값 선택 (IS Sharpe 가장 좋은 거)
  OOS (2024-2026): 최종 평가 (단 한 번)

[지표]
  ML: Accuracy, Precision, Recall, F1, AUC
  트레이딩: 필터링 후 OOS Sharpe, 거래 수, 성공률
  
[비교]
  베이스라인 (필터 없음) vs Logistic 필터 (선택된 임계값)
═══════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

INPUT_FILE = "./ml/features.csv"

# 식별자 (학습 안 함)
ID_COLS = ["entry_dt", "exit_dt", "entry_price", "exit_price",
           "trade_return", "y"]

# Categorical (one-hot encoding 필요)
CATEGORICAL_COLS = ["product", "source_model", "entry_type"]

# IS/OOS 경계
OOS_START = "2024-01-01"

# 임계값 후보
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]


def compute_sharpe(returns, years):
    """연환산 Sharpe"""
    if len(returns) == 0 or years <= 0: return 0
    rets = np.array(returns)
    if rets.std() == 0: return 0
    return rets.mean() / rets.std() * np.sqrt(len(rets) / years)


def trading_stats(df_trades, label=""):
    """거래 그룹의 트레이딩 통계"""
    if len(df_trades) == 0:
        return dict(n=0, win_rate=0, mean_ret=0, sharpe=0, total=0, mdd=0)
    
    rets = df_trades["trade_return"].values
    n = len(rets)
    years = (df_trades["exit_dt"].max() - df_trades["exit_dt"].min()).days / 365.25
    years = max(years, 0.01)
    
    win_rate = (rets > 0).mean() * 100
    mean_ret = rets.mean() * 100
    sharpe = compute_sharpe(rets, years)
    total = (np.prod(1 + rets) - 1) * 100
    
    # MDD
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min() * 100
    
    return dict(n=n, win_rate=win_rate, mean_ret=mean_ret,
                sharpe=sharpe, total=total, mdd=mdd, years=years)


def main():
    print("=" * 80)
    print(" ML Step 1 — 통합 (v1+v4) + Logistic Regression")
    print("=" * 80)
    
    # 데이터 로드
    df = pd.read_csv(INPUT_FILE, parse_dates=["entry_dt", "exit_dt"])
    print(f"\n[데이터 로드] {len(df):,} 거래")
    
    # Feature 준비
    # 1. categorical → one-hot
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    
    # 2. Feature 컬럼만 추출 (식별자 + raw categorical 제외)
    feature_cols = [c for c in df_encoded.columns 
                    if c not in ID_COLS and c not in CATEGORICAL_COLS]
    
    print(f"  Feature 컬럼: {len(feature_cols)}개")
    
    X = df_encoded[feature_cols].astype(float).values
    y = df_encoded["y"].astype(int).values
    
    # IS/OOS 분할
    mask_oos = df_encoded["entry_dt"] >= OOS_START
    X_is = X[~mask_oos]
    y_is = y[~mask_oos]
    df_is = df_encoded[~mask_oos].copy()
    
    X_oos = X[mask_oos]
    y_oos = y[mask_oos]
    df_oos = df_encoded[mask_oos].copy()
    
    print(f"\n[IS/OOS 분할]")
    print(f"  IS  (2018-2023): {len(X_is):>5} 거래 (성공률 {y_is.mean()*100:.1f}%)")
    print(f"  OOS (2024-2026): {len(X_oos):>5} 거래 (성공률 {y_oos.mean()*100:.1f}%)")
    
    # IS 내부 Train/Val 분할 (시간순, 80/20)
    n_is = len(X_is)
    split_idx = int(n_is * 0.8)
    
    X_train = X_is[:split_idx]
    y_train = y_is[:split_idx]
    X_val = X_is[split_idx:]
    y_val = y_is[split_idx:]
    df_val = df_is.iloc[split_idx:].copy()
    
    print(f"\n[IS 내부 분할 (시간순)]")
    print(f"  Train: {len(X_train):>5} 거래")
    print(f"  Val:   {len(X_val):>5} 거래")
    
    # 정규화 (Logistic 필수)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_oos_s = scaler.transform(X_oos)
    
    # Logistic Regression 학습
    print(f"\n[학습 — Logistic Regression]")
    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    model.fit(X_train_s, y_train)
    
    # Train 성능
    train_pred = model.predict(X_train_s)
    train_proba = model.predict_proba(X_train_s)[:, 1]
    train_acc = accuracy_score(y_train, train_pred)
    train_auc = roc_auc_score(y_train, train_proba)
    
    # Val 성능
    val_pred = model.predict(X_val_s)
    val_proba = model.predict_proba(X_val_s)[:, 1]
    val_acc = accuracy_score(y_val, val_pred)
    val_prec = precision_score(y_val, val_pred)
    val_rec = recall_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val, val_proba)
    
    print(f"  Train: Acc {train_acc:.3f}, AUC {train_auc:.3f}")
    print(f"  Val:   Acc {val_acc:.3f}, AUC {val_auc:.3f}, F1 {val_f1:.3f}")
    print(f"         Precision {val_prec:.3f}, Recall {val_rec:.3f}")
    
    # ─── 임계값 선택 (Val에서 IS 가장 좋은 거) ───
    print(f"\n[임계값 선택 — Validation에서 Sharpe 가장 좋은 거]")
    print(f"  {'임계값':<8} {'거래':>5} {'성공률':>7} {'평균수익':>9} {'Sharpe':>7}")
    print("  " + "─" * 50)
    
    df_val["proba"] = val_proba
    
    best_threshold = 0.5
    best_val_sharpe = -np.inf
    
    for th in THRESHOLDS:
        filtered = df_val[df_val["proba"] >= th]
        if len(filtered) < 10:
            print(f"  {th:<8.2f}  거래 너무 적음 ({len(filtered)})")
            continue
        s = trading_stats(filtered)
        marker = "  "
        if s["sharpe"] > best_val_sharpe:
            best_val_sharpe = s["sharpe"]
            best_threshold = th
            marker = "★"
        print(f"  {th:<8.2f} {s['n']:>5} {s['win_rate']:>6.1f}% "
              f"{s['mean_ret']:>+8.3f}% {s['sharpe']:>6.3f} {marker}")
    
    print(f"\n  ▶ 선택된 임계값: {best_threshold} (Val Sharpe {best_val_sharpe:.3f})")
    
    # ─── OOS 평가 ───
    print(f"\n" + "=" * 80)
    print(f" OOS 평가 — 최종 ")
    print("=" * 80)
    
    df_oos["proba"] = model.predict_proba(X_oos_s)[:, 1]
    
    # 베이스라인 (필터 없음)
    baseline = trading_stats(df_oos)
    
    # ML 필터링 후
    filtered_oos = df_oos[df_oos["proba"] >= best_threshold]
    filtered = trading_stats(filtered_oos)
    
    print(f"\n  {'구성':<25} {'거래':>5} {'성공률':>7} {'평균수익':>10} {'Sharpe':>8} {'MDD':>8}")
    print("  " + "─" * 75)
    print(f"  {'베이스라인 (필터 X)':<25} {baseline['n']:>5} {baseline['win_rate']:>6.1f}% "
          f"{baseline['mean_ret']:>+9.3f}% {baseline['sharpe']:>7.3f} {baseline['mdd']:>7.1f}%")
    print(f"  {'Logistic 필터 (th=' + str(best_threshold) + ')':<25} "
          f"{filtered['n']:>5} {filtered['win_rate']:>6.1f}% "
          f"{filtered['mean_ret']:>+9.3f}% {filtered['sharpe']:>7.3f} {filtered['mdd']:>7.1f}%")
    
    # 변화량
    sharpe_delta = filtered["sharpe"] - baseline["sharpe"]
    rejected = baseline["n"] - filtered["n"]
    rejected_pct = rejected / baseline["n"] * 100 if baseline["n"] > 0 else 0
    
    print(f"\n  변화:")
    print(f"    Sharpe:     {baseline['sharpe']:.3f} → {filtered['sharpe']:.3f} ({sharpe_delta:+.3f})")
    print(f"    평균수익:    {baseline['mean_ret']:+.3f}% → {filtered['mean_ret']:+.3f}%")
    print(f"    거래 거부:   {rejected} ({rejected_pct:.1f}%)")
    
    # ─── 종목별 분석 ───
    print(f"\n[종목별 OOS 효과]")
    print(f"  {'종목':<5} {'베이스라인':<20} {'필터링후':<20}")
    print(f"  {'':5} {'n  성공률  Sharpe':<20} {'n  성공률  Sharpe':<20}")
    print("  " + "─" * 50)
    
    # 원본 df_oos는 product 컬럼 있음
    df_oos_orig = df[df["entry_dt"] >= OOS_START].copy()
    df_oos_orig["proba"] = df_oos["proba"].values
    
    for p in ["ES", "NQ", "YM", "RTY"]:
        sub_base = df_oos_orig[df_oos_orig["product"] == p]
        sub_filt = sub_base[sub_base["proba"] >= best_threshold]
        s_base = trading_stats(sub_base)
        s_filt = trading_stats(sub_filt)
        print(f"  {p:<5} "
              f"{s_base['n']:>3}  {s_base['win_rate']:>5.1f}%  {s_base['sharpe']:>+6.3f}    "
              f"{s_filt['n']:>3}  {s_filt['win_rate']:>5.1f}%  {s_filt['sharpe']:>+6.3f}")
    
    # ─── 모델별 분석 ───
    print(f"\n[모델별 OOS 효과 (v1 vs v4)]")
    print(f"  {'모델':<5} {'베이스라인':<20} {'필터링후':<20}")
    print(f"  {'':5} {'n  성공률  Sharpe':<20} {'n  성공률  Sharpe':<20}")
    print("  " + "─" * 50)
    
    for m in ["v1", "v4"]:
        sub_base = df_oos_orig[df_oos_orig["source_model"] == m]
        sub_filt = sub_base[sub_base["proba"] >= best_threshold]
        s_base = trading_stats(sub_base)
        s_filt = trading_stats(sub_filt)
        print(f"  {m:<5} "
              f"{s_base['n']:>3}  {s_base['win_rate']:>5.1f}%  {s_base['sharpe']:>+6.3f}    "
              f"{s_filt['n']:>3}  {s_filt['win_rate']:>5.1f}%  {s_filt['sharpe']:>+6.3f}")
    
    # ─── Feature 중요도 (Logistic은 계수 절댓값) ───
    print(f"\n[Feature 중요도 (Logistic 계수, 절댓값 기준 Top 15)]")
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coef": model.coef_[0],
        "abs_coef": np.abs(model.coef_[0])
    }).sort_values("abs_coef", ascending=False)
    
    for _, row in coef_df.head(15).iterrows():
        sign = "+" if row["coef"] > 0 else "-"
        print(f"  {row['feature']:<25} coef = {sign}{abs(row['coef']):.3f}")
    
    # ─── 결과 및 판단 ───
    print(f"\n" + "=" * 80)
    print("결과 및 판단")
    print("=" * 80)
    
    if sharpe_delta > 0.1:
        verdict = f"Logistic 필터 효과 명확 (Sharpe +{sharpe_delta:.3f})"
    elif sharpe_delta > 0:
        verdict = f"Logistic 살짝 도움 (Sharpe +{sharpe_delta:.3f})"
    elif sharpe_delta > -0.05:
        verdict = "Logistic 효과 미미 (선형 정보 부족)"
    else:
        verdict = f"Logistic 부정적 (Sharpe {sharpe_delta:.3f}) - 과적합 가능성 체크"
    
    print(f"\n  ▶ {verdict}")
    print(f"\n  다음 단계 — XGBoost로 비선형 효과 보기")


if __name__ == "__main__":
    main()
