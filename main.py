import os
import pandas as pd
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
from datetime import datetime, timedelta
import threading
import traceback
import logging
# Fix for PyInstaller: Do not attempt to activate venv if compiled
if not getattr(sys, 'frozen', False):
    venv_script = os.path.join(os.getcwd(), '.venv', 'Scripts', 'activate_this.py')
    if os.path.exists(venv_script):
        exec(open(venv_script).read(), {'__file__': venv_script})
    else:
        print("Warning: Virtual environment not found. Running with system Python.")

# 🚀 SILENCE OPC LOGS (Critical for Performance)
# This prevents the console from flooding with "received header"
logging.getLogger("opcua").setLevel(logging.WARNING)
logging.getLogger("asyncua").setLevel(logging.WARNING)
logging.getLogger("uaclient").setLevel(logging.WARNING)

# --- PATH SETUP (CRITICAL) ---
# Ensure we are using the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'modules'))

# --- DYNAMIC EXTERNAL LOADER ---
import importlib.util

if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = BASE_DIR

def load_external_module(module_name, file_name):
    """Dynamically load a python file overriding any bundled versions."""
    file_path = os.path.join(APP_DIR, file_name)
    if os.path.exists(file_path):
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            print(f"✅ Loaded external override for module: {module_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to load external module {module_name}: {e}")
    return False

# Attempt to load external configs BEFORE they are imported by anything else
load_external_module("config", "config.py")
load_external_module("control_service", "control_service.py")

# --- IMPORTS ---
import config
import database
import process_model
import fingerprint_engine

# Wrap AI import in try/except so it doesn't crash if you are only testing Fingerprint
try:
    from modules.ai_core import mbrl_manager
except ImportError:
    print("⚠️ AI Module (mbrl_manager) not found. AI Strategy will be disabled.")
    mbrl_manager = None

import control_service  # <--- REQUIRED for PLC Control

from api import api_routes
from previousInfo import previous_info_routes
from authentication import auth_routes
from Interactive_plot_duna import create_dash_app

# --- LOGGING ---
from logging.handlers import RotatingFileHandler

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 1. Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# 2. File Handler (Rotating)
# Max 5MB per file, keep 5 backups
log_file_path = os.path.join(config.LOG_DIR, 'app.log')
file_handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Ensure the APP_DIR variables exist via config
template_dir = os.path.join(config.APP_DIR, 'templates')
static_dir = os.path.join(config.APP_DIR, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")
app.config.from_object('config')


# Note: We use config.CONTROL_MODE instead of a local variable now.

def initialize_system():
    logger.info("System Initializing...")

    # 1. Force Config Load with Absolute Path
    # Use config.APP_DIR to ensure it references the executable path, not the temp _MEIPASS folder.
    target_config_path = os.path.join(config.APP_DIR, 'files', 'json', 'model_config.json')
    logger.info(f"Target Config Path: {target_config_path}")

    if not os.path.exists(target_config_path):
        logger.error(f"CRITICAL: Config file DOES NOT EXIST at {target_config_path}")
    else:
        # Force the config module to use this specific path
        config.MODEL_CONFIG_PATH = target_config_path
        process_model.load_model_config()

    # 2. Verify Variables Count
    ctrls = process_model.get_control_variables()
    inds = process_model.get_indicator_variables()
    total_vars = len(ctrls) + len(inds)

    logger.info(f"Variables Loaded: {len(ctrls)} Controls + {len(inds)} Indicators = {total_vars} Total")

    # 3. Load History
    try:
        csv_path = os.path.join(config.APP_DIR, 'files', 'data', 'fingerprint4.csv')
        config.HISTORICAL_DATA_CSV_PATH = csv_path

        # Changed: Use the Parquet auto-optimizer instead of hard-loading CSV
        df = fingerprint_engine.robust_read_csv(csv_path)

        # === DATE FIX APPLIED HERE ===
        if config.TIMESTAMP_COLUMN in df.columns:
            # Let Pandas dynamically infer the datetime format to prevent strict assertion errors
            df[config.TIMESTAMP_COLUMN] = pd.to_datetime(df[config.TIMESTAMP_COLUMN], format='mixed', errors='coerce')

        app.config['df_fingerprint'] = df
        logger.info(f"History Loaded: {len(df)} rows")
    except Exception as e:
        logger.error(f"History Load Failed: {e}")
        app.config['df_fingerprint'] = pd.DataFrame()


# Run Init Logic
initialize_system()

# Register Blueprints
app.register_blueprint(api_routes, url_prefix="/api")
app.register_blueprint(previous_info_routes, url_prefix="/previous")
app.register_blueprint(auth_routes, url_prefix="/auth")
dash_app = create_dash_app(app)


# --- ROUTE: Serve Frontend ---
@app.route('/')
def index():
    """Serves the main HTML interface."""
    return render_template('index.html')


# --- BACKGROUND TASK: Data Stream ---
def background_data_emitter():
    """Reads real-time data from InfluxDB and pushes it to the UI every 2 seconds."""
    socketio.sleep(2)
    while True:
        try:
            tag_map = process_model.get_tag_to_name_map()
            tag_list = list(tag_map.keys())
            if tag_list:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(minutes=1)

                # Fetch live data
                df = database.get_realtime_data_window(start_time, end_time, tag_list, tag_map)

                if not df.empty:
                    latest = df.iloc[-1].to_dict()
                    # Convert timestamps for JSON serialization
                    if config.TIMESTAMP_COLUMN in latest:
                        latest[config.TIMESTAMP_COLUMN] = str(latest[config.TIMESTAMP_COLUMN])

                    socketio.emit('live_values', latest)
        except Exception as e:
            logger.error(f"Stream Error: {e}")
        socketio.sleep(2)


# --- BACKGROUND TASK: Autopilot Logic ---
def automated_control_loop():
    """
    The Core Intelligence Loop.
    Features:
    - Fast Cycle (2s): Sends Heartbeat (Watchdog) to PLC.
    - Slow Cycle (10s): Runs AI/Fingerprint calculations and executes Writes.
    """
    logger.info("🧠 Autopilot Thread Started")
    socketio.sleep(5)

    loop_counter = 0
    watchdog_val = 0

    # Timers pulled from config with safe fallbacks
    AI_INTERVAL_SECONDS = getattr(config, 'AI_INTERVAL_SECONDS', 30)
    FAST_CYCLE_SECONDS = getattr(config, 'FAST_CYCLE_SECONDS', 2)

    # SAFETY THRESHOLD (Adjust as needed)
    MAX_DATA_DELAY_SECONDS = 120

    while True:
        try:
            # 1. READ GLOBAL STATE
            current_mode = getattr(config, 'CONTROL_MODE', 0)

            # --- CRITICAL: DATA STALL CHECK ---
            tag_map = process_model.get_tag_to_name_map()
            all_tags = list(tag_map.keys())

            check_end = datetime.utcnow()
            check_start = check_end - timedelta(minutes=1)

            # Quick fetch to check freshness
            fresh_df = database.get_realtime_data_window(check_start, check_end, all_tags, tag_map)

            is_stalled = False

            if fresh_df.empty:
                is_stalled = True
            else:
                last_ts = fresh_df.iloc[-1].get(config.TIMESTAMP_COLUMN)
                if last_ts:
                    if isinstance(last_ts, str):
                        last_ts = pd.to_datetime(last_ts)

                    # Timezone fix
                    if hasattr(last_ts, 'tzinfo') and last_ts.tzinfo is not None:
                        last_ts = last_ts.tz_convert(None)

                    delay = (datetime.utcnow() - last_ts).total_seconds()

                    if delay > MAX_DATA_DELAY_SECONDS:
                        is_stalled = True
                        if loop_counter == 0:
                            logger.warning(
                                f"⚠️ DATA STALL: Last data was {delay:.1f}s ago (Limit: {MAX_DATA_DELAY_SECONDS}s)")

            # --- SAFETY ENFORCEMENT (PAUSE LOGIC) ---
            if is_stalled:
                # 1. Tell PLC we are OFF (Safety), even if internally engaged
                plc_status_code = 0

                # 2. Update UI to show "PAUSED" state, but do NOT disengage config.CONTROL_MODE
                if current_mode > 0 and loop_counter == 0:
                    logger.warning("⏸️ SYSTEM PAUSED: Waiting for data connectivity...")
                    socketio.emit('autopilot_update', {
                        "match_score": "SYSTEM-PAUSED",
                        "actions": [],
                        "reason": "Data Connection Lost - Retrying..."
                    })
            else:
                # Data is fresh, send actual mode to PLC (1 or 2)
                plc_status_code = current_mode

            # --- FAST CYCLE (Heartbeat) ---
            # Send Watchdog and Status Code to PLC
            control_service.service.send_handshake(watchdog_val, plc_status_code)

            watchdog_val = (watchdog_val + 1) % 100
            loop_counter += FAST_CYCLE_SECONDS

            # --- SLOW CYCLE (Control Optimization) ---
            if loop_counter >= AI_INTERVAL_SECONDS:
                loop_counter = 0

                # If stalled, skip calculation but keep loop alive
                if is_stalled:
                    socketio.sleep(FAST_CYCLE_SECONDS)
                    continue

                if current_mode > 0:
                    logger.info(f"Starting Optimization Cycle. Mode: {current_mode}")

                end_time = datetime.utcnow()
                start_time = end_time - timedelta(minutes=30)
                real_df = database.get_realtime_data_window(start_time, end_time, all_tags, tag_map)

                # Simulation Fallback (Strictly for Test Mode)
                is_test = getattr(config, 'TEST_MODE', False)
                if real_df.empty and is_test and app.config.get('df_fingerprint') is not None:
                    real_df = app.config['df_fingerprint'].iloc[-30:].copy()
                elif real_df.empty:
                    socketio.sleep(FAST_CYCLE_SECONDS)
                    continue

                if not real_df.empty:
                    recommendation = None

                    if current_mode == 2:  # FINGERPRINT
                        recommendation = fingerprint_engine.get_live_fingerprint_action(real_df)
                    elif current_mode == 1:  # AI
                        if mbrl_manager:
                            recommendation = mbrl_manager.get_optimal_action(real_df)
                    elif current_mode == 0:  # MONITOR
                        if mbrl_manager:
                            recommendation = mbrl_manager.get_optimal_action(real_df)
                            recommendation['match_score'] = "MONITOR"

                    if current_mode > 0 and recommendation and isinstance(recommendation, dict):
                        score = recommendation.get('match_score', '0')
                        if score == "SAFETY-CLAMP":
                            logger.warning("🛡️ Guardian Blocked Control")
                        else:
                            setpoints = {}
                            for action in recommendation.get('actions', []):
                                val = action.get('fingerprint_set_point')
                                if val is not None:
                                    setpoints[action['var_name']] = val

                            if setpoints:
                                setpoint_map = process_model.get_setpoint_tag_map()
                                scale_factors = process_model.get_setpoint_scale_factors()
                                database.write_setpoints(datetime.utcnow(), setpoints, setpoint_map, scale_factors)
                                control_service.service.execute_recommendation(recommendation)

                    if recommendation:
                        socketio.emit('autopilot_update', recommendation)

        except Exception as e:
            logger.error(f"Autopilot Cycle Error: {e}")
            traceback.print_exc()

        socketio.sleep(FAST_CYCLE_SECONDS)


# --- THREAD MANAGEMENT ---
thread = None
thread_lock = threading.Lock()


@socketio.on("connect")
def on_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = True  # Mark thread as started to prevent explosion
            socketio.start_background_task(background_data_emitter)
            socketio.start_background_task(automated_control_loop)


if __name__ == "__main__":
    # Ensure templates folder exists for the new route
    if not os.path.exists(os.path.join(config.APP_DIR, 'templates')):
        os.makedirs(os.path.join(config.APP_DIR, 'templates'), exist_ok=True)
        # print("⚠️ WARNING: Created 'templates' folder. Please move 'index.html' there.")

    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)