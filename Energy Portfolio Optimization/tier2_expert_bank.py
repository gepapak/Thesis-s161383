import numpy as np

from config import TIER2_CV_MEMORY_CHANNELS
from forecast_price_experts import PRICE_SHORT_EXPERT_METHODS


TIER2_ROUTE_EXPERT_NAMES = ("abstain",) + tuple(PRICE_SHORT_EXPERT_METHODS)
TIER2_ACTIVE_EXPERT_NAMES = TIER2_ROUTE_EXPERT_NAMES[1:]

_NUM_EXPERTS = len(TIER2_ACTIVE_EXPERT_NAMES)
_SIGNAL_SLICE = slice(0, _NUM_EXPERTS)
_QUALITY_SLICE = slice(_NUM_EXPERTS, 2 * _NUM_EXPERTS)
_RISK_SLICE = slice(2 * _NUM_EXPERTS, 3 * _NUM_EXPERTS)
_CONSENSUS_IDX = 3 * _NUM_EXPERTS
_DISAGREEMENT_IDX = _CONSENSUS_IDX + 1
_IMBALANCE_IDX = _CONSENSUS_IDX + 2


def tier2_expert_quality_blend(
    direction_accuracy: float,
    mape_quality: float,
    metadata_skill: float,
    economic_skill: float,
) -> float:
    return float(
        np.clip(
            0.30 * float(direction_accuracy)
            + 0.15 * float(mape_quality)
            + 0.20 * float(metadata_skill)
            + 0.35 * float(economic_skill),
            0.0,
            1.0,
        )
    )


def tier2_quality_gain(quality: float) -> float:
    return float(np.clip(0.60 + 0.40 * float(quality), 0.60, 1.0))


def tier2_risk_gain(
    risk: float,
    conformal_risk_scale: float = 1.0,
    runtime_conformal_margin: float = 0.05,
) -> float:
    scaled_excess = max(
        0.0,
        float(conformal_risk_scale) * float(np.clip(risk, 0.0, 1.0)) - float(runtime_conformal_margin),
    )
    return float(np.clip(1.0 - 0.35 * scaled_excess, 0.0, 1.0))


def tier2_route_probability_gain(route_probability: float) -> float:
    return float(np.clip(0.40 + 0.60 * float(route_probability), 0.0, 1.0))


def compute_short_horizon_window_stats(
    immediate_returns: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    returns_np = np.asarray(immediate_returns, dtype=np.float32).flatten()
    n = int(returns_np.size)
    if n <= 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z

    horizon = max(1, int(horizon))
    gross_returns = np.zeros(n, dtype=np.float32)
    downside_returns = np.zeros(n, dtype=np.float32)
    vol_returns = np.zeros(n, dtype=np.float32)
    for idx in range(n):
        end = min(n, idx + horizon)
        window = returns_np[idx:end]
        if window.size <= 0:
            continue
        gross_returns[idx] = float(np.sum(window, dtype=np.float32))
        downside_returns[idx] = float(
            np.sqrt(np.mean(np.square(np.minimum(window, 0.0)).astype(np.float32)))
        )
        vol_returns[idx] = float(np.std(window, dtype=np.float32) if window.size > 1 else abs(float(window[0])))
    return gross_returns, downside_returns, vol_returns


def decision_utility_numpy(
    exposure: np.ndarray,
    base_exposure: np.ndarray,
    gross_return: np.ndarray,
    downside_return: np.ndarray,
    vol_return: np.ndarray,
    downside_weight: float,
    vol_weight: float,
    stability_penalty: float,
) -> np.ndarray:
    exposure_np = np.asarray(exposure, dtype=np.float32)
    base_np = np.asarray(base_exposure, dtype=np.float32)
    gross_np = np.asarray(gross_return, dtype=np.float32)
    downside_np = np.asarray(downside_return, dtype=np.float32)
    vol_np = np.asarray(vol_return, dtype=np.float32)
    return (
        exposure_np * gross_np
        - float(downside_weight) * np.abs(exposure_np) * downside_np
        - float(vol_weight) * np.abs(exposure_np) * vol_np
        - float(stability_penalty) * np.square(exposure_np - base_np)
    ).astype(np.float32)


def tier2_short_expert_realized_utility_score(
    forecast_return: float,
    actual_return: float,
    utility_scale: float,
    delta_limit: float,
    runtime_gain: float,
    downside_weight: float,
    vol_weight: float,
    stability_penalty: float,
) -> float:
    """
    Map realized short-horizon expert usefulness into a bounded skill score.

    This is intentionally sign-aware: a bearish expert should be rewarded when
    realized returns are negative, rather than being penalized just because the
    raw market move itself was a downside event.
    """
    try:
        f_ret = float(forecast_return)
        a_ret = float(actual_return)
    except Exception:
        return 0.5
    if not (np.isfinite(f_ret) and np.isfinite(a_ret)):
        return 0.5

    utility_scale = max(float(utility_scale), 1e-6)
    reference_delta = max(float(delta_limit), 1e-6) * float(np.clip(runtime_gain, 0.0, 1.0))
    direction = float(np.sign(f_ret))
    if abs(direction) <= 1e-12 or reference_delta <= 1e-12:
        return 0.5

    signal_strength = float(np.clip(np.tanh(abs(f_ret) / utility_scale), 0.0, 1.0))
    exposure = direction * reference_delta * signal_strength
    if abs(exposure) <= 1e-12:
        return 0.5

    signed_realized_return = direction * a_ret
    gross_component = float(abs(exposure) * signed_realized_return)
    downside_component = float(abs(exposure) * abs(min(signed_realized_return, 0.0)))
    vol_component = float(abs(exposure) * abs(signed_realized_return))
    stability_component = float(stability_penalty) * float(exposure * exposure)
    utility = (
        gross_component
        - float(downside_weight) * downside_component
        - float(vol_weight) * vol_component
        - stability_component
    )
    return float(np.clip(0.5 + 0.5 * np.tanh(utility / utility_scale), 0.0, 1.0))


def extract_latest_tier2_memory_row(
    forecast_features: np.ndarray,
    core_dim: int = 4,
) -> np.ndarray:
    try:
        ff = np.asarray(forecast_features, dtype=np.float32).flatten()
    except Exception:
        return np.zeros(TIER2_CV_MEMORY_CHANNELS, dtype=np.float32)
    if ff.size < core_dim + TIER2_CV_MEMORY_CHANNELS:
        return np.zeros(TIER2_CV_MEMORY_CHANNELS, dtype=np.float32)
    memory_flat = ff[core_dim:]
    latest = np.asarray(memory_flat[-TIER2_CV_MEMORY_CHANNELS:], dtype=np.float32).flatten()
    if latest.size < TIER2_CV_MEMORY_CHANNELS:
        return np.zeros(TIER2_CV_MEMORY_CHANNELS, dtype=np.float32)
    return np.nan_to_num(latest, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def extract_tier2_route_state_from_features(
    forecast_features: np.ndarray,
    core_dim: int = 4,
) -> dict:
    latest = extract_latest_tier2_memory_row(forecast_features, core_dim=core_dim)
    has_active_memory = bool(np.max(np.abs(latest)) > 1e-8)
    if latest.size < _IMBALANCE_IDX + 1:
        latest = np.pad(latest, (0, max(0, _IMBALANCE_IDX + 1 - latest.size)))

    expert_signals = np.clip(latest[_SIGNAL_SLICE], -1.0, 1.0).astype(np.float32)
    route_quality = np.clip(latest[_QUALITY_SLICE], 0.0, 1.0).astype(np.float32)
    route_risk = np.clip(latest[_RISK_SLICE], 0.0, 1.0).astype(np.float32)
    consensus_signal = float(np.clip(latest[_CONSENSUS_IDX], -1.0, 1.0))
    disagreement_signal = float(np.clip(latest[_DISAGREEMENT_IDX], 0.0, 1.0))
    short_imbalance = float(np.clip(latest[_IMBALANCE_IDX], -1.0, 1.0))

    if not has_active_memory:
        route_quality = np.full(_NUM_EXPERTS, 0.5, dtype=np.float32)
        route_risk = np.full(_NUM_EXPERTS, 0.5, dtype=np.float32)

    quality_context = float(
        np.clip(
            float(np.mean(route_quality)) * (1.0 - 0.25 * float(np.mean(route_risk))) * (1.0 - 0.15 * disagreement_signal),
            0.0,
            1.0,
        )
    )
    forecast_trust = float(
        np.clip(
            0.60 * float(np.max(route_quality))
            + 0.25 * quality_context
            + 0.15 * (1.0 - float(np.mean(route_risk))),
            0.0,
            1.0,
        )
    )
    context_strength = float(
        np.clip(
            0.50 * float(np.max(np.abs(expert_signals))) +
            0.25 * abs(consensus_signal) +
            0.15 * disagreement_signal +
            0.10 * abs(short_imbalance),
            0.0,
            1.0,
        )
    )

    state = {
        "has_active_memory": float(has_active_memory),
        "expert_signals": expert_signals,
        "route_quality": route_quality,
        "route_risk": route_risk,
        "quality_context": quality_context,
        "forecast_trust": forecast_trust,
        "consensus_signal": consensus_signal,
        "disagreement_signal": disagreement_signal,
        "short_imbalance_signal": short_imbalance,
        "context_strength": context_strength,
    }
    for idx, method in enumerate(TIER2_ACTIVE_EXPERT_NAMES):
        state[f"{method}_signal"] = float(expert_signals[idx])
        state[f"{method}_quality"] = float(route_quality[idx])
        state[f"{method}_risk"] = float(route_risk[idx])
    return state


def build_routed_decision_training_bundle(
    immediate_returns: np.ndarray,
    base_exposures: np.ndarray,
    executed_exposures: np.ndarray,
    forecast_features: np.ndarray,
    horizon: int,
    delta_limit: float,
    utility_scale: float,
    downside_weight: float = 0.75,
    vol_weight: float = 0.25,
    stability_penalty: float = 0.12,
    magnitude_grid_points: int = 11,
    conformal_risk_scale: float = 1.0,
    runtime_conformal_margin: float = 0.05,
    runtime_gain: float = 1.0,
) -> dict:
    returns_np = np.asarray(immediate_returns, dtype=np.float32).flatten()
    base_np = np.asarray(base_exposures, dtype=np.float32).flatten()
    executed_np = np.asarray(executed_exposures, dtype=np.float32).flatten()
    ff_np = np.asarray(forecast_features, dtype=np.float32)
    n = int(returns_np.size)
    if n <= 0:
        z = np.zeros(0, dtype=np.float32)
        return {
            "delta_targets": z,
            "confidence_targets": z,
            "sample_weights": z,
            "gross_returns": z,
            "downside_returns": z,
            "vol_returns": z,
            "oracle_improvement": z,
            "executed_advantage": z,
            "base_exposures": z,
            "executed_exposures": z,
            "quality_targets": z,
            "route_targets": np.zeros(0, dtype=np.int32),
            "route_margins": z,
            "expert_signal_targets": z,
            "certified_improvement": z,
        }

    if ff_np.ndim == 1:
        ff_np = ff_np.reshape(n, -1)
    if base_np.size < n:
        fixed = np.zeros(n, dtype=np.float32)
        fixed[:base_np.size] = base_np[:base_np.size]
        base_np = fixed
    else:
        base_np = base_np[:n]
    if executed_np.size < n:
        fixed = base_np.copy()
        fixed[:executed_np.size] = executed_np[:executed_np.size]
        executed_np = fixed
    else:
        executed_np = executed_np[:n]
    if ff_np.shape[0] < n:
        pad = np.zeros((n, ff_np.shape[1]), dtype=np.float32)
        pad[:ff_np.shape[0], :] = ff_np[:ff_np.shape[0], :]
        ff_np = pad
    else:
        ff_np = ff_np[:n, :]

    delta_limit = float(max(delta_limit, 1e-6))
    utility_scale = float(max(utility_scale, 1e-6))
    magnitude_grid_points = max(3, int(magnitude_grid_points))
    runtime_gain = float(np.clip(runtime_gain, 0.0, 1.0))

    gross_returns, downside_returns, vol_returns = compute_short_horizon_window_stats(returns_np, horizon)
    base_utility = decision_utility_numpy(
        base_np,
        base_np,
        gross_returns,
        downside_returns,
        vol_returns,
        downside_weight,
        vol_weight,
        stability_penalty,
    )
    executed_utility = decision_utility_numpy(
        executed_np,
        base_np,
        gross_returns,
        downside_returns,
        vol_returns,
        downside_weight,
        vol_weight,
        stability_penalty,
    )
    executed_advantage = (executed_utility - base_utility).astype(np.float32)

    quality_targets = np.zeros(n, dtype=np.float32)
    route_targets = np.zeros(n, dtype=np.int32)
    route_margins = np.zeros(n, dtype=np.float32)
    delta_targets = np.zeros(n, dtype=np.float32)
    confidence_targets = np.zeros(n, dtype=np.float32)
    sample_weights = np.zeros(n, dtype=np.float32)
    oracle_improvements = np.zeros(n, dtype=np.float32)
    certified_improvements = np.zeros(n, dtype=np.float32)
    expert_signal_targets = np.zeros(n, dtype=np.float32)

    magnitude_grid = np.linspace(0.0, 1.0, num=magnitude_grid_points, dtype=np.float32)
    for idx in range(n):
        route_state = extract_tier2_route_state_from_features(ff_np[idx])
        expert_signals = np.asarray(route_state["expert_signals"], dtype=np.float32)
        route_quality = np.asarray(route_state["route_quality"], dtype=np.float32)
        route_risk = np.asarray(route_state["route_risk"], dtype=np.float32)
        quality = float(route_state["quality_context"])
        quality_targets[idx] = quality

        base_exposure = float(np.clip(base_np[idx], -1.0, 1.0))
        best_utility = float(base_utility[idx])
        best_certified = 0.0
        second_best = 0.0
        best_route = 0
        best_effective_delta = 0.0
        best_deterministic_gate = 1.0
        best_signal = 0.0
        best_risk = 1.0

        for local_idx, signal in enumerate(expert_signals, start=1):
            signal = float(np.clip(signal, -1.0, 1.0))
            if abs(signal) <= 1e-8:
                continue
            local_risk = float(np.clip(route_risk[local_idx - 1], 0.0, 1.0))
            local_quality = float(np.clip(route_quality[local_idx - 1], 0.0, 1.0))
            quality_gain = tier2_quality_gain(local_quality)
            risk_gain = tier2_risk_gain(
                local_risk,
                conformal_risk_scale=conformal_risk_scale,
                runtime_conformal_margin=runtime_conformal_margin,
            )
            deterministic_gate = float(np.clip(quality_gain * risk_gain, 0.0, 1.0))
            candidate_deltas = delta_limit * runtime_gain * signal * magnitude_grid * deterministic_gate
            candidate_exposures = np.clip(base_exposure + candidate_deltas, -1.0, 1.0)
            candidate_utilities = decision_utility_numpy(
                candidate_exposures,
                np.full(candidate_exposures.shape, base_exposure, dtype=np.float32),
                np.full(candidate_exposures.shape, gross_returns[idx], dtype=np.float32),
                np.full(candidate_exposures.shape, downside_returns[idx], dtype=np.float32),
                np.full(candidate_exposures.shape, vol_returns[idx], dtype=np.float32),
                downside_weight,
                vol_weight,
                stability_penalty,
            )
            local_best_idx = int(np.argmax(candidate_utilities))
            local_best_utility = float(candidate_utilities[local_best_idx])
            local_best_delta = float(np.clip(candidate_deltas[local_best_idx], -delta_limit, delta_limit))
            # Only treat a route as certified when it delivers real utility
            # improvement over the baseline exposure. Deterministic gating still
            # affects the searched delta and normalized target, but it should not
            # manufacture a positive route label by itself.
            certified = float(local_best_utility - base_utility[idx])
            if certified > best_certified:
                second_best = best_certified
                best_certified = certified
                best_utility = local_best_utility
                best_route = local_idx
                best_effective_delta = local_best_delta
                best_deterministic_gate = deterministic_gate
                best_signal = signal
                best_risk = local_risk
            else:
                second_best = max(second_best, certified)

        oracle_improvement = float(best_utility - base_utility[idx])
        oracle_improvements[idx] = oracle_improvement
        certified_improvement = float(max(0.0, best_certified))
        certified_improvements[idx] = certified_improvement
        route_margin = float(max(0.0, best_certified - second_best))
        route_margins[idx] = route_margin
        executed_improvement = float(max(0.0, executed_advantage[idx]))
        oracle_strength = float(np.clip(np.tanh(max(oracle_improvement, 0.0) / utility_scale), 0.0, 1.0))
        execution_gap = float(max(0.0, oracle_improvement - executed_improvement))
        execution_gap_strength = float(np.clip(np.tanh(execution_gap / utility_scale), 0.0, 1.0))

        effective_delta_strength = float(np.clip(abs(best_effective_delta) / delta_limit, 0.0, 1.0))
        if best_route > 0 and certified_improvement > 1e-8 and abs(best_effective_delta) > 1e-8:
            raw_delta_scale = max(delta_limit * runtime_gain * max(best_deterministic_gate, 1e-6), 1e-6)
            normalized_delta = float(np.clip(best_effective_delta / raw_delta_scale, -1.0, 1.0))
            margin_strength = float(np.clip(np.tanh(route_margin / utility_scale), 0.0, 1.0))
            route_targets[idx] = int(best_route)
            delta_targets[idx] = normalized_delta
            expert_signal_targets[idx] = float(best_signal)
            confidence_targets[idx] = float(
                np.clip(
                    0.25 * quality
                    + 0.25 * oracle_strength
                    + 0.20 * execution_gap_strength
                    + 0.15 * margin_strength
                    + 0.15 * (1.0 - best_risk),
                    0.0,
                    1.0,
                )
            )
        else:
            route_targets[idx] = 0
            delta_targets[idx] = 0.0
            expert_signal_targets[idx] = 0.0
            confidence_targets[idx] = 0.0

        signal_strength = max(effective_delta_strength, float(confidence_targets[idx]))
        sample_weights[idx] = float(
            np.clip(
                0.50
                + 0.15 * quality
                + 0.10 * signal_strength
                + 0.10 * (1.0 - best_risk)
                + 0.20 * oracle_strength
                + 0.15 * execution_gap_strength,
                0.55,
                1.60,
            )
        )

    return {
        "delta_targets": delta_targets.astype(np.float32),
        "confidence_targets": confidence_targets.astype(np.float32),
        "sample_weights": sample_weights.astype(np.float32),
        "gross_returns": gross_returns.astype(np.float32),
        "downside_returns": downside_returns.astype(np.float32),
        "vol_returns": vol_returns.astype(np.float32),
        "oracle_improvement": oracle_improvements.astype(np.float32),
        "executed_advantage": executed_advantage.astype(np.float32),
        "base_exposures": base_np.astype(np.float32),
        "executed_exposures": executed_np.astype(np.float32),
        "quality_targets": quality_targets.astype(np.float32),
        "route_targets": route_targets.astype(np.int32),
        "route_margins": route_margins.astype(np.float32),
        "expert_signal_targets": expert_signal_targets.astype(np.float32),
        "certified_improvement": certified_improvements.astype(np.float32),
    }
