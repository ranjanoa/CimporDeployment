import json
import os
import config
import pandas as pd
import numpy as np


# ==============================================================================
# 1. CONFIGURATION MANAGEMENT
# ==============================================================================
def load_model_config():
    """Loads the central configuration for variables and limits."""
    try:
        if os.path.exists(config.MODEL_CONFIG_PATH):
            with open(config.MODEL_CONFIG_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Config Load Error: {e}")

    # Fallback Default Config
    return {
        "model_name": "NEXUS-V4 Default",
        "control_variables": {},
        "indicator_variables": {}
    }


def save_model_config(new_config):
    """Saves updated configuration to disk."""
    try:
        with open(config.MODEL_CONFIG_PATH, 'w') as f:
            json.dump(new_config, f, indent=2)
        return True, "Saved"
    except Exception as e:
        return False, str(e)


# ==============================================================================
# 2. VARIABLE HELPERS
# ==============================================================================
def get_control_variables():
    conf = load_model_config()
    return conf.get('control_variables', {})


def get_indicator_variables():
    conf = load_model_config()
    return conf.get('indicator_variables', {})


def get_tag_to_name_map():
    """Maps DB Column Names -> Human Friendly Names"""
    conf = load_model_config()
    mapping = {}
    for name, data in conf.get('control_variables', {}).items():
        if 'tag_name' in data: mapping[data['tag_name']] = name
    for name, data in conf.get('indicator_variables', {}).items():
        if 'tag_name' in data: mapping[data['tag_name']] = name
    return mapping


def get_name_to_tag_map():
    """Maps Human Friendly Names -> DB Column Names"""
    tag_map = get_tag_to_name_map()
    return {v: k for k, v in tag_map.items()}


# ==============================================================================
# 3. ALGORITHM HELPERS
# ==============================================================================
def get_optimization_weights():
    """Returns weights for fingerprint ranking."""
    conf = load_model_config()
    weights = {}
    for name, data in conf.get('control_variables', {}).items():
        weights[name] = data.get('weight', 1.0)
    for name, data in conf.get('indicator_variables', {}).items():
        weights[name] = data.get('weight', 1.0)
    return weights


def get_setpoint_scale_factors():
    """Returns scaling factors for setpoint adjustments."""
    return {
        "calcinerHeadTemp": 1.0,
        "coalMainBurner": 1.0,
        "kilnFeed": 1.0,
        "primaryAirFan": 1.0,
        "systemFanSpeed": 1.0
    }


def get_setpoint_tag_map():
    """Returns mapping for Setpoint Tags (Write) vs PV Tags (Read)."""
    conf = load_model_config()
    mapping = {}
    for name, data in conf.get('control_variables', {}).items():
        # Use 'setpoint_tag' if available, else fallback to 'tag_name'
        tag = data.get('setpoint_tag', data.get('tag_name', name))
        mapping[name] = tag
    return mapping

def get_safety_rules():
    """
    Returns safety rules from the configuration.
    Used by fingerprint_engine to override setpoints if dangerous conditions exist.
    """
    conf = load_model_config()
    # Return empty list if no rules are defined in the JSON
    return conf.get('safety_rules', [])


# ==============================================================================
# 4. DATA FORMATTERS (API RESPONSES)
# ==============================================================================
def build_api_response(real_df, match_row, future_df, score, confidence, mode):
    """
    Formats response.
    INCLUDES FIX: Recalculates Score if it comes in as 0%.
    """
    controls = get_control_variables()

    # --- FIX: SYNTHETIC SCORE CALCULATION ---
    # If the incoming score is 0 or missing, calculate it manually based on % difference.
    if (score is None or score <= 0.1) and not real_df.empty:
        total_error = 0
        valid_vars = 0
        for var, data in controls.items():
            col = data.get('tag_name', var)
            try:
                curr = float(real_df.iloc[-1][col]) if col in real_df.columns else 0
                hist = float(match_row.get(col, 0))
                if curr != 0:
                    # Calculate % difference
                    total_error += abs((hist - curr) / curr)
                    valid_vars += 1
            except:
                pass

        if valid_vars > 0:
            avg_error = total_error / valid_vars
            # Formula: 100% - (Average % Error). e.g. 2% error = 98% match.
            score = max(0, 100 - (avg_error * 100))
            score = round(score, 1)

    # --- BUILD ACTION LIST ---
    actions = []
    for var, data in controls.items():
        col = data.get('tag_name', var)

        try:
            current_val = float(real_df.iloc[-1][col]) if not real_df.empty and col in real_df.columns else 0.0
            target_val = float(match_row.get(col, 0.0))
        except:
            continue

        diff = target_val - current_val

        # Determine Label (Optimizing vs Ramping)
        if abs(diff) > 0.1:
            pct_change = abs(diff / current_val) if current_val != 0 else 0

            if pct_change < 0.02:
                reason = "Optimizing"
            else:
                reason = "Ramping Up" if diff > 0 else "Ramping Down"

            actions.append({
                "var_name": var,
                "current_setpoint": current_val,
                "fingerprint_set_point": target_val,
                "diff": diff,
                "reason": reason,
                "type": "Control"
            })

    # --- BUILD CHARTS ---
    live_history = {}
    fingerprint_pred = {}

    clean_real = real_df.copy()
    clean_real.columns = [str(c).strip() for c in clean_real.columns]

    clean_future = future_df.copy()
    clean_future.columns = [str(c).strip() for c in clean_future.columns]

    top_vars = list(controls.keys())[:5]
    for v in top_vars:
        col = controls[v].get('tag_name', v)
        if col in clean_real.columns:
            live_history[v] = clean_real[col].fillna(0).tolist()
        if col in clean_future.columns:
            fingerprint_pred[v] = clean_future[col].fillna(0).tolist()

    return {
        "match_score": score,
        "confidence": confidence,
        "fingerprint_timestamp": str(match_row.get(config.TIMESTAMP_COLUMN, "N/A")),
        "actions": actions,
        "live_history": live_history,
        "fingerprint_prediction": fingerprint_pred,
        "top_variables": top_vars
    }


def build_no_fingerprint_response(current_state):
    return {
        "fingerprint_Found": "False",
        "match_score": 0,
        "actions": [],
        "debug_message": "No valid historical match found within constraints."
    }