"""
═══════════════════════════════════════════════════════════════════
  ML Step 3 — v1/v4 분리 모델 (분리 모델 > 통합 모델)
═══════════════════════════════════════════════════════════════════
[가설]
  v1과 v4는 진입 메커니즘 다름 (회귀 vs 5중 분류 + 선별)
  분리해서 학습하면 각 모델 특성 더 잘 학습

[비교 8가지]
  v1 + Logistic   (vs Step 1 통합)
  v1 + XGBoost    (vs Step 2 통합)
  v4 + Logistic   (vs Step 1 통합)
  v4 + XGBoost    (vs Step 2 통합)
  
[Walk-Forward 4폴드 (2026 제외)]
  고정 임계 0.60
═══════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGB = False

INPUT_FILE = "./ml/features.csv"

ID_COLS = ["entry_dt", "exit_dt", "entry_price", "exit_price",
           "trade_return", "y"]
CATEGORICAL_COLS = ["product", "source_model", "entry_type"]

DATA_END = pd.Timestamp("2025-12-31 23:59:59")
FIXED_THRESHOLD = 0.60

FOLDS = [
    {"name": "Fold 1", "train_end": "2021-12-31", "test_year": 2022},
    {"name": "Fold 2", "train_end": "2022-12-31", "test_year": 2023},
    {"name": "Fold 3", "train_end": "2023-12-31", "test_year": 2024},
    {"name": "Fold 4", "train_end": "2024-12-31", "test_year": 2025},
]


def get_logistic():
    return LogisticRegression(max_iter=1000, random_state=42, C=1.0)


def get_xgboost():
    if USE_XGB:
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric="logloss", verbosity=0
        )
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )


def compute_sharpe(returns, years):
    if len(returns) == 0 or years <= 0: return 0
    rets = np.array(returns)
    if rets.std() == 0: return 0
    return rets.mean() / rets.std() * np.sqrt(len(rets) / years)


def trading_stats(df_trades):
    if len(df_trades) == 0:
        return dict(n=0, win_rate=0, mean_ret=0, sharpe=0, mdd=0, total=0)
    rets = df_trades["trade_return"].values
    n = len(rets)
    years = (df_trades["exit_dt"].max() - df_trades["exit_dt"].min()).days / 365.25
    years = max(years, 0.01)
    sharpe = compute_sharpe(rets, years)
    total = (np.prod(1 + rets) - 1) * 100
    eq = np.cumprod(1 + rets); peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min() * 100
    return dict(n=n, win_rate=(rets > 0).mean() * 100, mean_ret=rets.mean() * 100,
                sharpe=sharpe, mdd=mdd, total=total)


def run_walk_forward(df_data, feature_cols, model_factory, label):
    """단일 모델 종류로 4폴드 WF 실행"""
    results = []
    all_base = []
    all_filt = []
    
    for fold in FOLDS:
        train_end = pd.Timestamp(fold["train_end"] + " 23:59:59")
        test_end = pd.Timestamp(f"{fold['test_year']}-12-31 23:59:59")
        
        df_train = df_data[df_data["entry_dt"] <= train_end].copy()
        df_test = df_data[(df_data["entry_dt"] > train_end) & 
                          (df_data["entry_dt"] <= test_end)].copy()
        
        if len(df_train) < 100 or len(df_test) < 15:
            results.append(None); continue
        
        X_train = df_train[feature_cols].astype(float).values
        y_train = df_train["y"].astype(int).values
        X_test = df_test[feature_cols].astype(float).values
        y_test = df_test["y"].astype(int).values
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = model_factory()
        model.fit(X_train_s, y_train)
        
        test_proba = model.predict_proba(X_test_s)[:, 1]
        df_test["proba"] = test_proba
        
        try: auc = roc_auc_score(y_test, test_proba)
        except: auc = 0
        
        baseline = trading_stats(df_test)
        filtered = trading_stats(df_test[df_test["proba"] >= FIXED_THRESHOLD])
        
        results.append({
            "fold": fold["name"], "test_year": fold["test_year"],
            "auc": auc,
            "baseline": baseline, "filtered": filtered,
            "delta": filtered["sharpe"] - baseline["sharpe"]
        })
        all_base.extend(df_test.to_dict("records"))
        all_filt.extend(df_test[df_test["proba"] >= FIXED_THRESHOLD].to_dict("records"))
    
    return results, pd.DataFrame(all_base), pd.DataFrame(all_filt)


def print_fold_table(results, label):
    print(f"\n  [{label}]")
    print(f"  {'Fold':<8} {'Test':>6} {'AUC':>6} "
          f"{'베이스라인':>22} {'필터링후':>22} {'개선':>9}")
    print(f"  {'':<8} {'':>6} {'':>6} "
          f"{'n  성공률  Sharpe':<22} {'n  성공률  Sharpe':<22}")
    print("  " + "─" * 80)
    
    for r in results:
        if r is None:
            print(f"  표본 부족"); continue
        b = r["baseline"]; f = r["filtered"]
        print(f"  {r['fold']:<8} {r['test_year']:>6} {r['auc']:>6.3f} "
              f"{b['n']:>3} {b['win_rate']:>5.1f}% {b['sharpe']:>+6.3f}      "
              f"{f['n']:>3} {f['win_rate']:>5.1f}% {f['sharpe']:>+6.3f}      "
              f"{r['delta']:>+6.3f}")
    
    valid = [r for r in results if r]
    if valid:
        deltas = np.array([r["delta"] for r in valid])
        print(f"\n  → 평균 개선 {deltas.mean():+.3f}, "
              f"std {deltas.std():.3f}, "
              f"개선 {(deltas > 0).sum()}/{len(deltas)}")


def main():
    print("=" * 100)
    print(" ML Step 3 — v1/v4 분리 모델")
    print("=" * 100)
    
    df = pd.read_csv(INPUT_FILE, parse_dates=["entry_dt", "exit_dt"])
    df = df[df["entry_dt"] <= DATA_END].copy().reset_index(drop=True)
    
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    
    # source_model_v1, source_model_v4 컬럼 제거 (분리하니 의미 없음)
    feature_cols = [c for c in df_enc.columns 
                    if c not in ID_COLS and c not in CATEGORICAL_COLS
                    and not c.startswith("source_model_")]
    
    df_enc["product"] = df["product"]
    df_enc["source_model"] = df["source_model"]
    
    print(f"\n[데이터] {len(df_enc):,} 거래 (2018-2025)")
    print(f"  Feature: {len(feature_cols)}개 (source_model 컬럼 제거)")
    
    # v1, v4 분리
    df_v1 = df_enc[df_enc["source_model"] == "v1"].copy().reset_index(drop=True)
    df_v4 = df_enc[df_enc["source_model"] == "v4"].copy().reset_index(drop=True)
    
    print(f"  v1: {len(df_v1):,} 거래")
    print(f"  v4: {len(df_v4):,} 거래")
    
    # ═══════════════════════════════════════
    # [1] v1 모델 — Logistic
    # ═══════════════════════════════════════
    print("\n" + "=" * 100)
    print("[1] v1 분리 모델")
    print("=" * 100)
    
    v1_log_results, v1_log_base, v1_log_filt = run_walk_forward(
        df_v1, feature_cols, get_logistic, "v1 + Logistic"
    )
    print_fold_table(v1_log_results, "v1 + Logistic")
    
    v1_xgb_results, v1_xgb_base, v1_xgb_filt = run_walk_forward(
        df_v1, feature_cols, get_xgboost, "v1 + XGBoost"
    )
    print_fold_table(v1_xgb_results, "v1 + XGBoost")
    
    # v1 합산
    s_v1_base = trading_stats(v1_log_base)  # 베이스라인은 동일
    s_v1_log_filt = trading_stats(v1_log_filt)
    s_v1_xgb_filt = trading_stats(v1_xgb_filt)
    
    print(f"\n  [v1 4폴드 합산]")
    print(f"  {'구성':<30} {'거래':>5} {'성공률':>7} {'Sharpe':>8} {'MDD':>8}")
    print("  " + "─" * 65)
    print(f"  {'v1 베이스라인':<30} {s_v1_base['n']:>5} {s_v1_base['win_rate']:>6.1f}% "
          f"{s_v1_base['sharpe']:>+7.3f} {s_v1_base['mdd']:>+7.1f}%")
    print(f"  {'v1 + Logistic 필터':<30} {s_v1_log_filt['n']:>5} {s_v1_log_filt['win_rate']:>6.1f}% "
          f"{s_v1_log_filt['sharpe']:>+7.3f} {s_v1_log_filt['mdd']:>+7.1f}%")
    print(f"  {'v1 + XGBoost 필터':<30} {s_v1_xgb_filt['n']:>5} {s_v1_xgb_filt['win_rate']:>6.1f}% "
          f"{s_v1_xgb_filt['sharpe']:>+7.3f} {s_v1_xgb_filt['mdd']:>+7.1f}%")
    
    # ═══════════════════════════════════════
    # [2] v4 모델
    # ═══════════════════════════════════════
    print("\n" + "=" * 100)
    print("[2] v4 분리 모델")
    print("=" * 100) 
    
    v4_log_results, v4_log_base, v4_log_filt = run_walk_forward(
        df_v4, feature_cols, get_logistic, "v4 + Logistic"
    )
    print_fold_table(v4_log_results, "v4 + Logistic")
    
    v4_xgb_results, v4_xgb_base, v4_xgb_filt = run_walk_forward(
        df_v4, feature_cols, get_xgboost, "v4 + XGBoost"
    )
    print_fold_table(v4_xgb_results, "v4 + XGBoost")
    
    # v4 합산
    s_v4_base = trading_stats(v4_log_base)
    s_v4_log_filt = trading_stats(v4_log_filt)
    s_v4_xgb_filt = trading_stats(v4_xgb_filt)
    
    print(f"\n  [v4 4폴드 합산]")
    print(f"  {'구성':<30} {'거래':>5} {'성공률':>7} {'Sharpe':>8} {'MDD':>8}")
    print("  " + "─" * 65)
    print(f"  {'v4 베이스라인':<30} {s_v4_base['n']:>5} {s_v4_base['win_rate']:>6.1f}% "
          f"{s_v4_base['sharpe']:>+7.3f} {s_v4_base['mdd']:>+7.1f}%")
    print(f"  {'v4 + Logistic 필터':<30} {s_v4_log_filt['n']:>5} {s_v4_log_filt['win_rate']:>6.1f}% "
          f"{s_v4_log_filt['sharpe']:>+7.3f} {s_v4_log_filt['mdd']:>+7.1f}%")
    print(f"  {'v4 + XGBoost 필터':<30} {s_v4_xgb_filt['n']:>5} {s_v4_xgb_filt['win_rate']:>6.1f}% "
          f"{s_v4_xgb_filt['sharpe']:>+7.3f} {s_v4_xgb_filt['mdd']:>+7.1f}%")
    
    # ═══════════════════════════════════════
    # [3] 최적 조합 (v1 best + v4 best)
    # ═══════════════════════════════════════
    print("\n" + "=" * 100)
    print("[3] 최적 조합 — v1 best + v4 best")
    print("=" * 100)
    
    # v1 best 선택
    v1_log_sharpe = s_v1_log_filt["sharpe"]
    v1_xgb_sharpe = s_v1_xgb_filt["sharpe"]
    if v1_log_sharpe >= v1_xgb_sharpe:
        v1_best_name = "Logistic"
        v1_best_filt = v1_log_filt
    else:
        v1_best_name = "XGBoost"
        v1_best_filt = v1_xgb_filt
    
    # v4 best 선택
    v4_log_sharpe = s_v4_log_filt["sharpe"]
    v4_xgb_sharpe = s_v4_xgb_filt["sharpe"]
    if v4_log_sharpe >= v4_xgb_sharpe:
        v4_best_name = "Logistic"
        v4_best_filt = v4_log_filt
    else:
        v4_best_name = "XGBoost"
        v4_best_filt = v4_xgb_filt
    
    print(f"\n  v1 최고: {v1_best_name} (Sharpe {max(v1_log_sharpe, v1_xgb_sharpe):+.3f})")
    print(f"  v4 최고: {v4_best_name} (Sharpe {max(v4_log_sharpe, v4_xgb_sharpe):+.3f})")
    
    # 최적 조합 합산
    df_optimal = pd.concat([v1_best_filt, v4_best_filt], ignore_index=True)
    s_optimal = trading_stats(df_optimal)
    
    # 전체 베이스라인
    df_all_base = pd.concat([v1_log_base, v4_log_base], ignore_index=True)
    s_all_base = trading_stats(df_all_base)
    
    print(f"\n  [최적 조합 (v1 {v1_best_name} + v4 {v4_best_name})]")
    print(f"  {'구성':<30} {'거래':>5} {'성공률':>7} {'Sharpe':>8} {'MDD':>8} {'총수익':>10}")
    print("  " + "─" * 75)
    print(f"  {'전체 베이스라인':<30} {s_all_base['n']:>5} {s_all_base['win_rate']:>6.1f}% "
          f"{s_all_base['sharpe']:>+7.3f} {s_all_base['mdd']:>+7.1f}% {s_all_base['total']:>+9.2f}%")
    print(f"  {'분리 ML 최적':<30} {s_optimal['n']:>5} {s_optimal['win_rate']:>6.1f}% "
          f"{s_optimal['sharpe']:>+7.3f} {s_optimal['mdd']:>+7.1f}% {s_optimal['total']:>+9.2f}%")
    
    # ═══════════════════════════════════════
    # [4] 통합 vs 분리 비교
    # ═══════════════════════════════════════
    print("\n" + "=" * 100)
    print("[4] 통합 (Step 1, 2) vs 분리 (Step 3) 비교")
    print("=" * 100)
    
    print(f"\n  {'구성':<35} {'Sharpe':>8} {'MDD':>8} {'총수익':>10}")
    print("  " + "─" * 65)
    print(f"  {'베이스라인':<35} {s_all_base['sharpe']:>+7.3f} {s_all_base['mdd']:>+7.1f}% {s_all_base['total']:>+9.2f}%")
    print(f"  {'Step 1 통합 Logistic':<35} {'+1.583':>8} {'-34.9%':>8} {'+352.41%':>10}")
    print(f"  {'Step 2 통합 XGBoost':<35} {'+1.435':>8} {'-52.9%':>8} {'+334.56%':>10}")
    print(f"  {'Step 3 분리 ' + v1_best_name + '+' + v4_best_name:<35} "
          f"{s_optimal['sharpe']:>+7.3f} {s_optimal['mdd']:>+7.1f}% {s_optimal['total']:>+9.2f}%")
    
    vs_step1 = s_optimal["sharpe"] - 1.583
    print(f"\n  Step 3 vs Step 1: Sharpe {vs_step1:+.3f}")
    
    # ═══════════════════════════════════════
    # 최종 판단
    # ═══════════════════════════════════════
    print("\n" + "=" * 100)
    print("[최종 판단]")
    print("=" * 100)
    
    if vs_step1 > 0.15:
        verdict = f"분리 모델이 통합보다 명확히 좋음 (+{vs_step1:.3f}) → v1/v4 분리 가치 명확"
    elif vs_step1 > 0.05:
        verdict = f"분리 살짝 좋음 (+{vs_step1:.3f})"
    elif vs_step1 > -0.05:
        verdict = f"분리 ≈ 통합 ({vs_step1:+.3f}) → 단순한 Step 1 통합 추천"
    else:
        verdict = f"분리 나쁨 ({vs_step1:+.3f}) → 표본 부족 영향"
    
    print(f"\n  ▶ {verdict}")
    print(f"\n  v1 ML 효과 (v1 베이스 {s_v1_base['sharpe']:+.3f}):")
    print(f"    Logistic: {s_v1_log_filt['sharpe']:+.3f} ({s_v1_log_filt['sharpe']-s_v1_base['sharpe']:+.3f})")
    print(f"    XGBoost:  {s_v1_xgb_filt['sharpe']:+.3f} ({s_v1_xgb_filt['sharpe']-s_v1_base['sharpe']:+.3f})")
    print(f"\n  v4 ML 효과 (v4 베이스 {s_v4_base['sharpe']:+.3f}):")
    print(f"    Logistic: {s_v4_log_filt['sharpe']:+.3f} ({s_v4_log_filt['sharpe']-s_v4_base['sharpe']:+.3f})")
    print(f"    XGBoost:  {s_v4_xgb_filt['sharpe']:+.3f} ({s_v4_xgb_filt['sharpe']-s_v4_base['sharpe']:+.3f})")


if __name__ == "__main__":
    main()

