import pandas as pd
import numpy as np

base = (
    r"batch_tier_phase_runs"
    r"\tiers_tier1_tier2_full_ablated_20260331_220819_tier1_tier2_full_ablated_3seed_ann"
    r"\seed7\tier2_seed7\logs"
)

for ep in [6, 7]:
    dbg_path = base + rf"\tier2_debug_ep{ep}.csv"
    port_path = base + rf"\tier2_portfolio_ep{ep}.csv"
    rew_path = base + rf"\tier2_rewards_ep{ep}.csv"

    try:
        dbg = pd.read_csv(dbg_path)
        port = pd.read_csv(port_path)
        rew = pd.read_csv(rew_path)
    except FileNotFoundError as e:
        print(f"ep{ep}: missing file — {e}")
        continue

    dec = dbg[dbg["decision_step"] == 1].copy()
    en = dec[dec["tier2_overlay_enabled"] == 1].copy()

    lg = dec["tier2_overlay_learned_gate"]
    rgs = en["tier2_overlay_realized_gain_signal"]
    delta = en["tier2_overlay_delta"]
    gs = dec["training_global_step"]
    nav = port["fund_nav_usd"]
    ir = rew["investor_reward"]

    print("=" * 62)
    print(f"EPISODE {ep}  |  steps={len(dec)}  |  global_step={int(gs.min())}-{int(gs.max())}")
    print()

    print("[GATE]")
    pct = round(len(en) / len(dec) * 100, 1)
    print(f"  enabled:     {len(en)} / {len(dec)} = {pct}%")
    print(f"  learned_gate mean={round(lg.mean(),4)}  std={round(lg.std(),4)}  >0.5={round((lg>0.5).mean()*100,1)}%")
    print(f"  value_gate   mean={round(dec['tier2_overlay_value_gate'].mean(),4)}  >0.5={round((dec['tier2_overlay_value_gate']>0.5).mean()*100,1)}%")
    print(f"  nav_gate     mean={round(dec['tier2_overlay_nav_gate'].mean(),4)}  >0.5={round((dec['tier2_overlay_nav_gate']>0.5).mean()*100,1)}%")
    print(f"  floor_gate   mean={round(dec['tier2_overlay_return_floor_gate'].mean(),4)}  >0.5={round((dec['tier2_overlay_return_floor_gate']>0.5).mean()*100,1)}%")
    print()

    print("[REALIZED GAIN - when enabled]")
    rg_pos = round((rgs > 0).mean() * 100, 1)
    print(f"  pos%={rg_pos}%  mean={round(rgs.mean(),7)}  median={round(rgs.median(),7)}")
    print(f"  std={round(rgs.std(),7)}  min={round(rgs.min(),7)}  max={round(rgs.max(),7)}")
    print()

    print("[DELTA - when enabled]")
    print(f"  mean={round(delta.mean(),5)}  pos%={round((delta>0).mean()*100,1)}%  |delta| mean={round(delta.abs().mean(),5)}")
    print()

    print("[SHARPE - when enabled]")
    sb = en["tier2_overlay_sharpe_before"]
    sa = en["tier2_overlay_sharpe_after"]
    sd = en["tier2_overlay_sharpe_delta"]
    print(f"  sharpe_before mean={round(sb.mean(),4)}")
    print(f"  sharpe_after  mean={round(sa.mean(),4)}")
    print(f"  sharpe_delta  mean={round(sd.mean(),4)}  >0: {round((sd>0).mean()*100,1)}%")
    print()

    print("[PREDICTIONS - when enabled]")
    print(f"  value_pred  mean={round(en['tier2_overlay_value_prediction'].mean(),6)}")
    print(f"  value_lcb   mean={round(en['tier2_overlay_value_lcb'].mean(),6)}")
    print(f"  nav_pred    mean={round(en['tier2_overlay_nav_prediction'].mean(),6)}")
    print(f"  floor_pred  mean={round(en['tier2_overlay_return_floor_prediction'].mean(),6)}")
    print(f"  tail_risk   mean={round(en['tier2_overlay_tail_risk_prediction'].mean(),4)}")
    print(f"  reliability mean={round(en['tier2_overlay_reliability'].mean(),4)}")
    print()

    print("[EXPERT MIX - when enabled]")
    for col, name in [
        ("tier2_overlay_internal_base_weight", "base"),
        ("tier2_overlay_internal_trend_weight", "trend"),
        ("tier2_overlay_internal_reversion_weight", "reversion"),
        ("tier2_overlay_internal_defensive_weight", "defensive"),
    ]:
        if col in en.columns:
            print(f"  {name}: mean={round(en[col].mean(),4)}")
    print()

    print("[PORTFOLIO & REWARDS]")
    nav_ret = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
    print(f"  NAV: {round(nav.iloc[0]/1e6,1)}M -> {round(nav.iloc[-1]/1e6,1)}M  ({round(nav_ret,3)}%)")
    print(f"  inv_reward: sum={round(ir.sum(),1)}  mean={round(ir.mean(),5)}  pos%={round((ir>0).mean()*100,1)}%")
    print()
