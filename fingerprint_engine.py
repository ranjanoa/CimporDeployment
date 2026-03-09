import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime, timedelta

# ADVANCED MATH IMPORTS
try:
    from scipy.spatial.distance import mahalanobis
    from scipy.linalg import pinv
except ImportError:
    mahalanobis = None
    pinv = None

# LOCAL IMPORTS
try:
    import config
except ImportError:
    config = None


# ==============================================================================
# 0. ENHANCED LOGGING SETUP
# ==============================================================================
def setup_logging():
    """Sets up a specific logger for the Fingerprint Engine."""
    _logger = logging.getLogger("FingerprintEngine")
    _logger.setLevel(logging.INFO)

    if not _logger.handlers:
        if not os.path.exists("logs"):
            os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler("logs/fingerprint_debug.log", encoding='utf-8')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        _logger.addHandler(fh)
        _logger.addHandler(ch)
    return _logger


engine_logger = setup_logging()

# GLOBAL PROCESS MODEL INIT
process_model = None
try:
    import process_model
    HAS_PROCESS_MODEL = True
except ImportError:
    HAS_PROCESS_MODEL = False

# ==============================================================================
# 1. GLOBAL CACHE & CONFIG
# ==============================================================================
CACHE_DF = None
CACHE_COV = None
CACHE_MTIME = 0.0

STATE_FILE = "files/json/engine_state.json"


def get_config_path():
    if config:
        return getattr(config, 'HISTORICAL_DATA_CSV_PATH', "files/data/fingerprint4.csv")
    return "files/data/fingerprint4.csv"


def get_timestamp_col():
    if config:
        return getattr(config, 'TIMESTAMP_COLUMN', "1_timestamp")
    return "1_timestamp"


def load_engine_state():
    if not os.path.exists(STATE_FILE): return {}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_engine_state(state):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        engine_logger.error(f"State Save Error: {e}")


def get_model_config_safe():
    if HAS_PROCESS_MODEL and process_model:
        try:
            return process_model.load_model_config()
        except Exception:
            pass
    return {}


# ==============================================================================
# 2. LOW-LEVEL HELPERS
# ==============================================================================
def robust_read_csv(file_path):
    parquet_path = file_path.replace('.csv', '.parquet')
    try:
        if os.path.exists(parquet_path):
            csv_mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
            parquet_mtime = os.path.getmtime(parquet_path)
            if csv_mtime > parquet_mtime:
                engine_logger.info("Detected newly updated CSV file! Recompiling Parquet cache...")
            else:
                df = pd.read_parquet(parquet_path)
                engine_logger.info(f"Loaded Parquet file instantly with {len(df)} rows.")
                return df

        if not os.path.exists(file_path):
            engine_logger.warning(f"Data file not found at: {file_path} or {parquet_path}")
            return pd.DataFrame()

        engine_logger.info("Reading raw CSV. This will take a moment before optimizing...")
        df = pd.read_csv(file_path)
        df.columns = [str(c).strip() for c in df.columns]

        try:
            df.to_parquet(parquet_path, engine='pyarrow')
            engine_logger.info(f"Optimized history and saved Parquet cache to {parquet_path}")
        except Exception as pe:
            engine_logger.warning(f"Could not save Parquet file: {pe} (Install pyarrow to enable caching).")

        return df
    except Exception as e:
        engine_logger.error(f"Data Read Error: {e}")
        return pd.DataFrame()


def map_csv_headers(hist_df, controls_cfg, indicators_cfg):
    """Restored mapping logic using 'tag_name' from JSON."""
    if hist_df.empty: return hist_df
    df = hist_df.copy()
    rename_map = {}
    all_vars = {}
    if controls_cfg: all_vars.update(controls_cfg)
    if indicators_cfg: all_vars.update(indicators_cfg)

    for friendly, cfg in all_vars.items():
        opc = cfg.get('tag_name')
        if opc and opc in df.columns:
            rename_map[opc] = friendly

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def map_tags_to_friendly_names(current_state_map, controls_cfg, indicators_cfg):
    mapped_state = current_state_map.copy()
    all_vars = {}
    if controls_cfg: all_vars.update(controls_cfg)
    if indicators_cfg: all_vars.update(indicators_cfg)
    opc_lookup = {}
    for friendly_name, cfg in all_vars.items():
        if 'tag_name' in cfg: opc_lookup[cfg['tag_name']] = friendly_name
    for key, value in current_state_map.items():
        if key in opc_lookup: mapped_state[opc_lookup[key]] = value
    return mapped_state


def align_magnitude(target_val, current_val):
    try:
        if target_val == 0 or current_val == 0: return target_val
        ratio = abs(current_val / target_val)
        if 800 < ratio < 1200: return target_val * 1000.0
        if 0.0008 < ratio < 0.0012: return target_val / 1000.0
        if 80 < ratio < 120: return target_val * 100.0
        return target_val
    except Exception:
        return target_val


def pre_calculate_slopes(df, controls_cfg):
    df_slopes = df.copy()
    if controls_cfg:
        for tag_key in controls_cfg.keys():
            if tag_key in df.columns:
                df_slopes[f"{tag_key}_slope"] = df[tag_key].diff().fillna(0)
    return df_slopes


def get_heat_input(fuel_tag, flow_value, current_state, conf):
    """
    Computes specific heat input (GJ/h) = flow (t/h) * calorific_value (GJ/t).
    Pairing is defined in model_config.json under 'fuel_calorific_pairing'.
    Falls back to raw flow value if CV tag not available.
    """
    pairing = conf.get('fuel_calorific_pairing', {})
    cv_tag = pairing.get(fuel_tag)
    if cv_tag:
        cv_value = float(current_state.get(cv_tag, 0))
        if cv_value > 0:
            heat_input = flow_value * cv_value
            return heat_input
    return flow_value  # Fallback: use raw flow if CV not available


def check_future_stability(historical_df, candidate_ts):
    """
    Checks if the process remains stable after the candidate timestamp.
    All stability tags defined in model_config.json 'logic_tags.stability_tags' must pass.
    Each tag has its own threshold_pct for standard deviation / mean ratio.
    """
    ts_col = get_timestamp_col()
    if ts_col not in historical_df.columns: return False
    try:
        conf = get_model_config_safe().get('logic_tags', {})
        lookahead = conf.get('stability_lookahead', 30)

        # Read the full list of stability tags from config — no hardcoded tags
        stability_tag_list = conf.get('stability_tags', [])

        # Legacy fallback: if old single-tag config is used
        if not stability_tag_list:
            legacy_tag = conf.get('primary_stability_tag')
            if legacy_tag:
                threshold_pct = conf.get('stability_threshold_pct', 0.05)
                stability_tag_list = [{"tag": legacy_tag, "threshold_pct": threshold_pct}]

        if not stability_tag_list: return True

        match_idx = historical_df.index[historical_df[ts_col] == candidate_ts].tolist()
        if not match_idx: return False
        idx = match_idx[0]
        if idx + 1 + lookahead >= len(historical_df): return False

        future_slice = historical_df.iloc[idx + 1: idx + 1 + lookahead]
        if future_slice.empty: return False

        # All tags must pass their individual threshold
        for entry in stability_tag_list:
            tag_name = entry.get('tag') if isinstance(entry, dict) else entry
            threshold_pct = entry.get('threshold_pct', 0.05) if isinstance(entry, dict) else 0.05

            if tag_name in future_slice.columns:
                std_dev = future_slice[tag_name].std()
                mean_val = future_slice[tag_name].mean()
                if mean_val != 0 and (std_dev / abs(mean_val)) > threshold_pct:
                    engine_logger.debug(
                        f"Stability FAIL on [{tag_name}]: std/mean={std_dev/abs(mean_val):.3f} > {threshold_pct}")
                    return False

        return True
    except Exception:
        return True


def get_cached_dataframe(controls_cfg, indicators_cfg):
    global CACHE_DF, CACHE_MTIME, CACHE_COV
    csv_path = get_config_path()
    try:
        if not os.path.exists(csv_path): return pd.DataFrame()
        current_mtime = float(os.path.getmtime(csv_path))
        if CACHE_DF is not None and CACHE_MTIME == current_mtime: return CACHE_DF

        engine_logger.info("Reloading dataframe from disk (cache miss or update)...")
        hist_df = robust_read_csv(csv_path)
        hist_df = map_csv_headers(hist_df, controls_cfg, indicators_cfg)
        ts_col = get_timestamp_col()
        if ts_col in hist_df.columns:
            hist_df[ts_col] = pd.to_datetime(hist_df[ts_col], format="%Y-%m-%d %H:%M:%S", errors='coerce')
        CACHE_DF = hist_df
        CACHE_MTIME = current_mtime
        CACHE_COV = None
        return hist_df
    except Exception as e:
        engine_logger.error(f"Cache Error: {e}")
        return pd.DataFrame()


# ==============================================================================
# 3. DYNAMIC WEIGHT BIAS (OPERATIONAL MATRIX)
# ==============================================================================
def calculate_dynamic_weights(current_state, base_weights):
    """
    All rules, thresholds, tags, and weights are read from model_config.json.
    Nothing is hardcoded. Rules can be enabled/disabled via 'matrix_rules' in config.
    Hot kiln: fuel-first priority enforced by 'hot_kiln_rule.priority_action'.
    """
    if not HAS_PROCESS_MODEL or not process_model: return base_weights

    new_weights = base_weights.copy()
    try:
        full_conf = process_model.load_model_config()
        matrix = full_conf.get('operational_matrix_settings', {})
        if not matrix.get('enabled', False): return base_weights

        tags = matrix.get('tags', {})
        lim = matrix.get('limits', {})
        bias = matrix.get('matrix_bias', {})
        actuators = matrix.get('actuators', {})
        rules_enabled = matrix.get('matrix_rules', {})
        hot_kiln_rule = matrix.get('hot_kiln_rule', {})

        bzt = float(current_state.get(tags.get('bzt'), 0))
        o2 = float(current_state.get(tags.get('o2_inlet'), 0))
        c4_temp = float(current_state.get(tags.get('c4_temp'), 0))

        fuel_tag = actuators.get('fuel_main')
        feed_tag = actuators.get('feed')
        fan_tag = actuators.get('id_fan')
        calc_tag = actuators.get('fuel_calciner')

        # --- RULE 1: HOT KILN ---
        if rules_enabled.get('hot_kiln_enabled', True):
            if bzt > lim.get('bzt_hot', 9999):
                priority = hot_kiln_rule.get('priority_action', 'fuel_first')
                fuel_w = hot_kiln_rule.get('fuel_weight', bias.get('hot_kiln_fuel_weight', -15.0))
                feed_w = hot_kiln_rule.get('feed_weight', bias.get('hot_kiln_feed_weight', 10.0))
                fuel_first_only = hot_kiln_rule.get('apply_feed_only_if_fuel_already_reduced', True)

                if fuel_tag:
                    new_weights[fuel_tag] = fuel_w

                # Feed increase is secondary — only applied if fuel is already trending down
                # and priority_action allows it
                if feed_tag and priority == 'fuel_first' and not fuel_first_only:
                    new_weights[feed_tag] = feed_w
                elif feed_tag and priority != 'fuel_first':
                    new_weights[feed_tag] = feed_w

                engine_logger.info(
                    f"KILN HOT ({bzt:.0f}°C): Fuel weight={fuel_w} applied first. "
                    f"Feed weight applied: {not fuel_first_only}")

        # --- RULE 2: COLD KILN ---
        if rules_enabled.get('cold_kiln_enabled', True):
            if lim.get('bzt_cold', 0) > bzt > 500:
                if fuel_tag:
                    new_weights[fuel_tag] = bias.get('cold_kiln_fuel_weight', 15.0)
                engine_logger.info(f"KILN COLD ({bzt:.0f}°C): Adjusted weights for High Fuel")

        # --- RULE 3: LOW O2 → ID FAN (disabled by default per operator instruction) ---
        if rules_enabled.get('low_o2_id_fan_enabled', False):
            if o2 < lim.get('o2_min', 9999) and o2 > 0.1:
                if fan_tag:
                    new_weights[fan_tag] = bias.get('low_o2_fan_weight', 0)
                engine_logger.info(f"LOW O2 ({o2:.1f}%): Adjusted weights for High Draft")

        # --- RULE 4: LOW C4 TEMP ---
        if rules_enabled.get('low_c4_temp_enabled', True):
            if c4_temp < lim.get('c4_temp_min', 9999) and c4_temp > 400:
                if calc_tag:
                    new_weights[calc_tag] = bias.get('low_c4_calciner_weight', 8.0)
                engine_logger.info(f"LOW C4 TEMP ({c4_temp:.0f}°C): Adjusted weights for High Calciner Fuel")

    except Exception as e:
        engine_logger.error(f"Weight Bias Error: {e}")
        return base_weights

    return new_weights


# ==============================================================================
# 4. CORE SCORING ENGINE & MATCH CALCULATION
# ==============================================================================
def calculate_match_percentage(current_state, row, controls_cfg):
    """Calculates a numerical 0-100% similarity score."""
    if not isinstance(current_state, dict) or not controls_cfg: return 0.0
    conf = get_model_config_safe()
    fuel_pairing = conf.get('fuel_calorific_pairing', {})

    dist_sum, count = 0.0, 0
    for tag, props in controls_cfg.items():
        if tag in row:
            curr_val = float(current_state.get(tag, 0))
            hist_val = align_magnitude(row.get(tag, 0), curr_val)

            # Use heat input (flow × CV) for fuel tags if pairing is available
            if tag in fuel_pairing:
                curr_val = get_heat_input(tag, curr_val, current_state, conf)
                hist_val_raw = float(row.get(tag, 0))
                hist_cv_tag = fuel_pairing[tag]
                hist_cv = float(row.get(hist_cv_tag, 0))
                if hist_cv > 0:
                    hist_val = hist_val_raw * hist_cv

            if curr_val != 0:
                dist_sum += ((abs(curr_val - hist_val) / curr_val) ** 2)
                count += 1
    if count == 0: return 0.0
    return max(0, min(100, 100 * (1 - np.sqrt(dist_sum / count))))


def _calculate_core_score(row, current_state, controls_cfg, weights=None, active_constraints=None, inv_cov=None,
                          live_slopes=None, penalty_weight=1000.0, is_advanced=False):
    score = 0.0
    now = pd.Timestamp.now()
    ts_col = get_timestamp_col()
    conf = get_model_config_safe()

    if weights:
        for tag, w in weights.items(): score += (row.get(tag, 0) * w)

    if isinstance(current_state, dict):
        dist_sum = 0.0

        for tag, props in active_constraints.items() if active_constraints else controls_cfg.items():
            if is_advanced:
                try:
                    curr_row_val = float(row.get(tag, 0))
                    user_min = float(props.get('min', props.get('Min', props.get('default_min', -9e9))))
                    user_max = float(props.get('max', props.get('Max', props.get('default_max', 9e9))))
                    if curr_row_val < user_min or curr_row_val > user_max:
                        return -999999.9
                except:
                    pass

            prio = int(props.get('priority', 3))
            if not is_advanced and prio != 1: continue

            curr_val = float(current_state.get(tag, 0))
            hist_val = align_magnitude(row.get(tag, 0), curr_val)

            # Use specific heat input for fuel tags
            if tag in conf.get('fuel_calorific_pairing', {}):
                curr_val = get_heat_input(tag, curr_val, current_state, conf)
                hist_val_raw = float(row.get(tag, 0))
                cv_tag = conf['fuel_calorific_pairing'][tag]
                hist_cv = float(row.get(cv_tag, 0))
                if hist_cv > 0:
                    hist_val = hist_val_raw * hist_cv

            if curr_val != 0:
                weight = {1: 10.0, 2: 5.0}.get(prio, 1.0) if is_advanced else 1.0
                dist_sum += ((abs(curr_val - hist_val) / curr_val) ** 2) * weight

        score -= (dist_sum * penalty_weight)

    if ts_col in row and pd.notnull(row[ts_col]):
        try:
            age_days = (now - row[ts_col]).total_seconds() / 86400.0
            score -= (age_days * 0.05)
        except:
            pass
    return score


# ==============================================================================
# 5. SEARCH & OPTIMIZATION
# ==============================================================================
def apply_golden_filter(hist_df):
    if hist_df.empty: return hist_df
    conf = get_model_config_safe().get('logic_tags', {})

    filter_tag = conf.get('golden_filter_tag')
    filter_limit = conf.get('golden_filter_max', 850.0)

    if filter_tag and filter_tag in hist_df.columns:
        return hist_df[hist_df[filter_tag] <= filter_limit]
    return hist_df


def get_mahalanobis_matrix(hist_df, active_cols):
    global CACHE_COV
    if mahalanobis is None or pinv is None: return None
    try:
        if CACHE_COV is not None and isinstance(CACHE_COV, np.ndarray) and CACHE_COV.shape[0] == len(active_cols):
            return CACHE_COV
        sub_df = hist_df[active_cols].dropna()
        if sub_df.empty: return None
        cov_matrix = np.cov(sub_df.values.T)
        inv_cov = pinv(cov_matrix)
        CACHE_COV = inv_cov
        return inv_cov
    except Exception:
        return None


def find_best_fingerprint_advanced(current_real_df_window, historical_df, frontend_strategy, current_state,
                                   weights=None):
    if historical_df.empty or not frontend_strategy: return []

    initial_count = len(historical_df)
    engine_logger.info(f"[SEARCH] Starting optimization on total dataset of {initial_count} rows.")

    valid_history = apply_golden_filter(historical_df.copy())
    after_golden = len(valid_history)
    if after_golden < initial_count:
        engine_logger.info(f"[SEARCH] Golden Filter applied: {after_golden} rows remaining.")

    ts_col = get_timestamp_col()
    active_constraints = {}
    active_tags = []

    for tag, strategy in frontend_strategy.items():
        if tag not in valid_history.columns: continue
        try:
            prev_count = len(valid_history)

            abs_min = float(strategy.get('min', -9e9))
            abs_max = float(strategy.get('max', 9e9))

            tol_pct = float(strategy.get('tolerance_pct', 25.0)) / 100.0
            cur_val = float(current_state.get(tag, 0))

            if cur_val != 0:
                tol_min = cur_val * (1 - tol_pct)
                tol_max = cur_val * (1 + tol_pct)
                eff_min = max(abs_min, tol_min)
                eff_max = min(abs_max, tol_max)
            else:
                eff_min, eff_max = abs_min, abs_max

            if eff_min > eff_max:
                engine_logger.warning(f"Impossible range for {tag}: {eff_min} to {eff_max}. Adjusting limits.")
                eff_min, eff_max = min(eff_min, eff_max), max(eff_min, eff_max)

            valid_history = valid_history[valid_history[tag].between(eff_min, eff_max)]

            dropped = prev_count - len(valid_history)
            if dropped > 0:
                engine_logger.info(
                    f"Filter {tag} [{eff_min:.1f}-{eff_max:.1f}]: Removed {dropped} rows. Remaining: {len(valid_history)}")

            active_constraints[tag] = strategy
            active_tags.append(tag)

            if valid_history.empty:
                engine_logger.warning(f"Filter {tag} dropped ALL rows. Aborting further filtering.")
                break
        except:
            continue

    if valid_history.empty:
        engine_logger.warning("[SEARCH] Strict filters returned 0 matches. Reverting to tail(500) fallback.")
        valid_history = historical_df.tail(500).copy()
        for tag, strategy in frontend_strategy.items():
            if tag in valid_history.columns:
                try:
                    min_l = float(strategy.get('min', strategy.get('Min', -9e9)))
                    max_l = float(strategy.get('max', strategy.get('Max', 9e9)))
                    valid_history = valid_history[valid_history[tag].between(min_l, max_l)]
                except:
                    continue

    inv_cov = get_mahalanobis_matrix(valid_history, active_tags)

    if ts_col in valid_history.columns:
        valid_history[ts_col] = pd.to_datetime(valid_history[ts_col], errors='coerce')

    def _adv_score_wrapper(row):
        return _calculate_core_score(
            row, current_state, None, weights,
            active_constraints=active_constraints,
            inv_cov=inv_cov,
            is_advanced=True
        )

    valid_history['score'] = valid_history.apply(_adv_score_wrapper, axis=1)
    df_sorted = valid_history.sort_values(by='score', ascending=False)
    df_sorted = df_sorted[df_sorted['score'] > -900000]

    stable_rows = []
    engine_logger.info(f"OPTIMIZATION: Found {len(df_sorted)} matches.")

    for _, r in df_sorted.iterrows():
        if check_future_stability(historical_df, r.get(ts_col)):
            stable_rows.append(r)
        if len(stable_rows) >= 5: break

    return stable_rows


# ==============================================================================
# 6. MAIN CONTROLLER
# ==============================================================================

# --- GLOBAL CACHE FOR AUTO MODE ---
LAST_AUTO_SCAN_TIME = None
CACHED_AUTO_RESULT = None


def get_scan_interval():
    return getattr(config, 'SCAN_INTERVAL_SECONDS', 300)


def calculate_kpis(current_state):
    """
    Calculates Key Performance Indicators for the UI.
    Tags are read from model_config.json — no hardcoded tag names.
    Now includes primary optimisation target (motor current) alongside BZT and O2.
    """
    defaults = {"BZT": 0.0, "Feed": 0.0, "O2": 0.0, "MotorCurrent": 0.0}
    try:
        conf = get_model_config_safe()
        matrix_tags = conf.get('operational_matrix_settings', {}).get('tags', {})
        matrix_acts = conf.get('operational_matrix_settings', {}).get('actuators', {})
        opt_target = conf.get('optimisation_target', {})

        bzt_tag = matrix_tags.get('bzt')
        o2_tag = matrix_tags.get('o2_inlet')
        feed_tag = matrix_acts.get('feed')
        motor_tag = opt_target.get('primary_tag')

        return {
            "BZT":          round(float(current_state.get(bzt_tag, 0)), 1) if bzt_tag else 0,
            "Feed":         round(float(current_state.get(feed_tag, 0)), 1) if feed_tag else 0,
            "O2":           round(float(current_state.get(o2_tag, 0)), 2) if o2_tag else 0,
            "MotorCurrent": round(float(current_state.get(motor_tag, 0)), 1) if motor_tag else 0
        }
    except:
        return defaults


def check_disturbance_rules(current_state):
    """
    Checks safety rules from model_config.json 'safety_rules'.
    All thresholds and actions are config-driven.
    Enforces gradual ramp via 'ramp_rate' — no step changes ever produced.
    """
    if not HAS_PROCESS_MODEL or not process_model: return None
    try:
        conf = process_model.load_model_config()
        # Also read nudge_settings for a default rate cap
        nudge = conf.get('nudge_settings', {})
        default_ramp_rate = nudge.get('min_step_fraction', 0.005)

        for rule in conf.get('safety_rules', []):
            live = float(current_state.get(rule['condition_var'], 9999))
            op = rule.get('operator')
            thresh = rule.get('threshold')

            if (op == '>' and live > thresh) or (op == '<' and live < thresh):
                tgt = rule['action_var']
                action_type = rule.get('action_type', 'offset')
                raw_value = rule['action_value']

                # Enforce ramp_rate — the action_value per cycle is capped
                ramp_rate = rule.get('ramp_rate', None)
                curr = float(current_state.get(tgt, 0))

                if action_type == 'offset':
                    # Cap the offset to ramp_rate to ensure gradual change
                    if ramp_rate is not None:
                        capped_value = max(-abs(ramp_rate), min(abs(ramp_rate), raw_value))
                    else:
                        # Fallback: cap to default_ramp_rate fraction of current value
                        capped_value = raw_value * default_ramp_rate if curr != 0 else raw_value
                    new_v = curr + capped_value
                elif action_type == 'min_clamp':
                    # Convert legacy min_clamp to a gradual offset toward the clamp value
                    target_clamped = float(raw_value)
                    diff = target_clamped - curr
                    if ramp_rate is not None:
                        step = max(-abs(ramp_rate), min(abs(ramp_rate), diff))
                    else:
                        step = diff * default_ramp_rate
                    new_v = curr + step
                else:
                    new_v = curr + raw_value

                engine_logger.warning(
                    f"SAFETY RULE '{rule['name']}': {rule['condition_var']}={live:.1f} {op} {thresh}. "
                    f"Nudging {tgt}: {curr:.3f} → {new_v:.3f} (ramp_rate={ramp_rate})")
                return {
                    "match_score": "SAFETY-CLAMP",
                    "timestamp": str(pd.Timestamp.now()),
                    "actions": [{"var_name": tgt, "fingerprint_set_point": new_v,
                                 "reason": f"SAFETY: {rule['name']} (gradual)"}]
                }
    except Exception as e:
        engine_logger.error(f"Disturbance Rule Error: {e}")
    return None


def get_live_fingerprint_action(current_real_df_window, frontend_strategy=None):
    """
    Main Loop. All nudge step parameters are read from model_config.json 'nudge_settings'.
    No hardcoded step fractions or full-jump fallbacks.
    """
    global LAST_AUTO_SCAN_TIME, CACHED_AUTO_RESULT

    if current_real_df_window.empty: return None
    try:
        raw_state = current_real_df_window.iloc[-1].to_dict()
        now = pd.Timestamp.now()

        mode = getattr(config, 'FINGERPRINT_MODE_TYPE', 'AUTO') if config else "AUTO"

        # 1. Configuration & Mapping
        if HAS_PROCESS_MODEL and process_model:
            controls_cfg = process_model.get_control_variables()
            indicators_cfg = process_model.get_indicator_variables()
            base_weights = process_model.get_optimization_weights()
            if not frontend_strategy:
                frontend_strategy = {
                    k: {"priority": int(v.get('priority', 3)),
                        "min": float(v.get('default_min', -9e9)),
                        "max": float(v.get('default_max', 9e9)),
                        "tolerance_pct": 25}
                    for k, v in controls_cfg.items()
                }
        else:
            controls_cfg = getattr(config, 'control_variables', {}) if config else {}
            indicators_cfg = getattr(config, 'indicator_variables', {}) if config else {}
            base_weights = {}
            frontend_strategy = frontend_strategy or {}

        # Read nudge settings from config — no hardcoded step values
        full_conf = get_model_config_safe()
        nudge_cfg = full_conf.get('nudge_settings', {})
        step_fraction = nudge_cfg.get('step_fraction', 0.15)
        min_step_fraction = nudge_cfg.get('min_step_fraction', 0.005)
        allow_full_jump = nudge_cfg.get('allow_full_jump', False)

        current_state = map_tags_to_friendly_names(raw_state, controls_cfg, indicators_cfg)

        if (d := check_disturbance_rules(current_state)): return d

        dynamic_weights = calculate_dynamic_weights(current_state, base_weights)

        target_vals, target_disp, reason = {}, "Searching...", "Optimized"
        future_data, top_matches = [], []

        # =========================================================
        # DECISION BLOCK: MANUAL vs AUTO (TIMED)
        # =========================================================

        if mode == 'MANUAL':
            engine_logger.info("=== CYCLE START | Mode: MANUAL ===")
            try:
                with open(os.path.join(config.JSON_DIR, "current_target.json"), 'r') as f:
                    data = json.load(f)
                    target_disp = data.get("fingerprint_timestamp", "Manual")
                    for a in data.get('actions', []): target_vals[a['var_name']] = float(a['fingerprint_set_point'])
                    reason = "Manual Target"
            except:
                mode = 'AUTO'  # Fallback

        if mode != 'MANUAL':
            time_since_last = (now - LAST_AUTO_SCAN_TIME).total_seconds() if LAST_AUTO_SCAN_TIME else 99999

            if time_since_last >= get_scan_interval() or CACHED_AUTO_RESULT is None:
                engine_logger.info(f"=== CYCLE START | Mode: AUTO [SCANNING NEW TARGET] ===")

                hist_df = get_cached_dataframe(controls_cfg, indicators_cfg)
                best_rows = find_best_fingerprint_advanced(
                    current_real_df_window, hist_df, frontend_strategy, current_state, weights=dynamic_weights
                )

                if best_rows:
                    best = best_rows[0]
                    ts_col = get_timestamp_col()

                    CACHED_AUTO_RESULT = {
                        "target_vals": best.to_dict(),
                        "target_disp": str(best.get(ts_col)),
                        "top_matches": [str(r.get(ts_col)) for r in best_rows]
                    }
                    LAST_AUTO_SCAN_TIME = now
                    engine_logger.info(f"[AUTO] New Target Found: {CACHED_AUTO_RESULT['target_disp']}")
                else:
                    engine_logger.warning("[AUTO] No matches found. Keeping previous target.")
                    LAST_AUTO_SCAN_TIME = now

            else:
                engine_logger.info(f"=== CYCLE START | Mode: AUTO [USING CACHED TARGET] ===")
                engine_logger.info(f"Next scan in {int(get_scan_interval() - time_since_last)} seconds.")

            if CACHED_AUTO_RESULT:
                target_vals = CACHED_AUTO_RESULT["target_vals"]
                target_disp = CACHED_AUTO_RESULT["target_disp"]
                top_matches = CACHED_AUTO_RESULT.get("top_matches", [])
                reason = "Best Match (Cached)"

        # =========================================================
        # CONTROL LOOP — Nudge calculation, fully config-driven
        # =========================================================
        ui_actions = []

        for tag, cfg_var in controls_cfg.items():
            # Skip if explicitly excluded from AI control (aipc: false)
            if not cfg_var.get('aipc', True): continue
            if not cfg_var.get('is_setpoint', True): continue

            curr = float(current_state.get(tag, 0))
            tgt = align_magnitude(float(target_vals.get(tag, curr)), curr)

            gap = tgt - curr
            step = gap * step_fraction

            # Enforce minimum step — but NEVER allow a full jump (allow_full_jump=False)
            if abs(step) < abs(curr) * min_step_fraction:
                if allow_full_jump:
                    step = gap  # Only if explicitly permitted by config
                else:
                    step = gap * min_step_fraction  # Gradual minimum nudge

            ui_actions.append({
                "var_name": tag,
                "fingerprint_set_point": curr + step,
                "current_setpoint": str(curr),
                "reason": f"{reason} (Nudging)"
            })

        return {
            "match_score": f"ACTIVE-{mode}", "timestamp": str(now),
            "target_timestamp": target_disp, "top_matches": top_matches,
            "fingerprint_future": future_data,
            "calculated_metrics": calculate_kpis(current_state),
            "actions": ui_actions
        }
    except Exception as e:
        engine_logger.error(f"Runtime Error: {e}", exc_info=True)
        return None


# ==============================================================================
# 7. LEGACY API SUPPORT (RESTORED)
# ==============================================================================
def calculate_deviation_ranges(real_data_series, user_deviation_json):
    deviation_ranges = {}
    deviation_data = user_deviation_json.get("deviation", {})
    engine_logger.info("--- [SCAN] Calculating Deviation Ranges ---")
    for key, values in deviation_data.items():
        if key not in real_data_series: continue
        try:
            current_value = float(real_data_series.get(key, 0))
            if current_value == 0: continue
            abs_min = values.get("Min")
            abs_max = values.get("Max")
            lower_pct = float(values.get("Lower", 80)) / 100.0
            higher_pct = float(values.get("Higher", 120)) / 100.0
            calc_min = current_value * lower_pct
            calc_max = current_value * higher_pct
            final_min = float(abs_min) if abs_min is not None else calc_min
            if abs_min is not None and calc_min < float(abs_min): final_min = float(abs_min)
            final_max = float(abs_max) if abs_max is not None else calc_max
            if abs_max is not None and calc_max > float(abs_max): final_max = float(abs_max)
            deviation_ranges[key] = (final_min, final_max)
        except Exception:
            continue
    return deviation_ranges, {}, {}


def filter_historical_by_deviation(historical_df, deviation_ranges):
    if historical_df.empty: return pd.DataFrame()
    initial_count = len(historical_df)
    engine_logger.info(f"--- [SCAN] Filtering History (Initial: {initial_count}) ---")
    df_filtered = historical_df.copy()
    try:
        for col, (min_val, max_val) in deviation_ranges.items():
            if col in df_filtered.columns:
                prev_len = len(df_filtered)
                df_filtered = df_filtered[df_filtered[col].between(min_val, max_val)]
                new_len = len(df_filtered)
                if prev_len - new_len > 0:
                    engine_logger.info(
                        f"Filter {col} [{min_val:.1f}-{max_val:.1f}]: Removed {prev_len - new_len} rows. Remaining: {new_len}")
        return df_filtered
    except Exception as e:
        engine_logger.error(f"Filtering Error: {e}")
        return pd.DataFrame()


def rank_and_select_recommendations(historical_df, candidates, weights=None, current_state=None, controls_cfg=None,
                                    **kwargs):
    engine_logger.info("--- [SCAN] Ranking & Selection Started ---")
    ts_col = get_timestamp_col()
    if isinstance(candidates, list):
        df = historical_df[historical_df[ts_col].isin(candidates)].copy()
    elif hasattr(candidates, 'empty'):
        df = candidates.copy() if not candidates.empty else pd.DataFrame()
    else:
        return []
    if df.empty: return []

    conf = get_model_config_safe()
    df['score'] = 0.0
    if weights:
        for tag, w in weights.items():
            if tag in df.columns:
                df['score'] += df[tag].fillna(0) * w

    if isinstance(current_state, dict) and controls_cfg:
        dist_sum = pd.Series(0.0, index=df.index)

        for tag, props in controls_cfg.items():
            if tag not in df.columns: continue
            try:
                user_min = float(props.get('min', props.get('Min', props.get('default_min', -9e9))))
                user_max = float(props.get('max', props.get('Max', props.get('default_max', 9e9))))

                out_of_bounds = (df[tag] < user_min) | (df[tag] > user_max)
                df.loc[out_of_bounds, 'score'] = -999999.9

                prio = int(props.get('priority', 3))
                curr_val = float(current_state.get(tag, 0))

                # Use specific heat input for fuel tags
                if tag in conf.get('fuel_calorific_pairing', {}):
                    curr_val = get_heat_input(tag, curr_val, current_state, conf)

                if curr_val != 0:
                    weight = {1: 10.0, 2: 5.0}.get(prio, 1.0)

                    hist_vals = df[tag].copy()
                    ratios = np.abs(curr_val / hist_vals.replace(0, np.nan))
                    hist_vals = np.where((ratios > 800) & (ratios < 1200), hist_vals * 1000.0, hist_vals)
                    hist_vals = np.where((ratios > 0.0008) & (ratios < 0.0012), hist_vals / 1000.0, hist_vals)
                    hist_vals = np.where((ratios > 80) & (ratios < 120), hist_vals * 100.0, hist_vals)

                    dist_sum += ((np.abs(curr_val - hist_vals) / curr_val) ** 2) * weight

            except Exception:
                continue

        df['score'] -= (dist_sum * 1000.0)

    if ts_col in df.columns:
        now = pd.Timestamp.now()
        age_days = (now - df[ts_col]).dt.total_seconds() / 86400.0
        df['score'] -= (age_days.fillna(0) * 0.05)

    df = df.sort_values(by=['score'], ascending=False)
    stable_candidates = []
    unstable_candidates = []
    for _, row in df.iterrows():
        ts = row[ts_col]
        score = row['score']
        if check_future_stability(historical_df, ts):
            stable_candidates.append(ts)
            if len(stable_candidates) <= 5:
                engine_logger.info(f"MATCH #{len(stable_candidates)}: {ts} (Score: {score:.1f}) - Stable: YES")
        else:
            unstable_candidates.append(ts)
        if len(stable_candidates) >= 5: break
    if len(stable_candidates) < 5:
        needed = 5 - len(stable_candidates)
        stable_candidates.extend(unstable_candidates[:needed])
    return stable_candidates


def pre_filter_by_constraints(historical_df, current_state, controls_cfg):
    if not controls_cfg or not isinstance(current_state, dict): return historical_df
    df_filtered = historical_df.copy()
    conf = get_model_config_safe()
    for tag, cfg in controls_cfg.items():
        try:
            if int(cfg.get('priority', 100)) == 1:
                val = float(current_state.get(tag, 0))
                if val == 0: continue

                # Use heat input for fuel tags
                if tag in conf.get('fuel_calorific_pairing', {}):
                    val = get_heat_input(tag, val, current_state, conf)

                min_v, max_v = val * 0.75, val * 1.25
                if tag in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered[tag].between(min_v, max_v)]
                if len(df_filtered) < 5: return historical_df
        except:
            continue
    return df_filtered


def find_candidates_hierarchical(hist_df, current_state, controls_cfg, indicators_cfg):
    engine_logger.info("--- [SCAN] Starting Hierarchical Candidate Search ---")
    all_vars = {}
    if controls_cfg: all_vars.update(controls_cfg)
    if indicators_cfg: all_vars.update(indicators_cfg)
    p1_vars = {k: v for k, v in all_vars.items() if int(v.get('priority', 99)) == 1}

    def get_auto_ranges(vars_dict, multiplier=1.0):
        ranges = {}
        for tag, cfg in vars_dict.items():
            if tag in current_state:
                try:
                    val = float(current_state[tag])
                    if val != 0:
                        ranges[tag] = (val * (1.0 - (0.10 * multiplier)), val * (1.0 + (0.10 * multiplier)))
                except:
                    pass
        return ranges

    engine_logger.info("[SCAN] Attempting Pass 1: Strict +/- 10% on Priority 1 tags")
    candidates = filter_historical_by_deviation(hist_df, get_auto_ranges(p1_vars, 1.0))
    if candidates.empty:
        engine_logger.info("[SCAN] Pass 1 yielded 0 results. Attempting Pass 2: Loose +/- 30%")
        candidates = filter_historical_by_deviation(hist_df, get_auto_ranges(p1_vars, 3.0))
    return candidates if not candidates.empty else pd.DataFrame()