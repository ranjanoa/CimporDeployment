# config.py
# System-level configuration for InfluxDB 2.0+ & Hybrid Control

import os
import sys
from datetime import datetime

# --- PATH FIX: Get the absolute path of the project root ---
# If running as an executable created by PyInstaller (sys.frozen),
# get the directory of the executable itself to allow 'files/' to live next to it.
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
    # PyInstaller unpacks internal files (templates, static) to _MEIPASS
    BASE_DIR = sys._MEIPASS
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = APP_DIR

# ==============================================================================
# 1. NEW HYBRID CONTROL SETTINGS
# ==============================================================================
# Global State Machine: 0=Monitor, 1=AI, 2=Fingerprint
CONTROL_MODE = 0
TEST_MODE = False

# PLC Connection Requirement (True to require PLC for setpoint generation, False to generate without PLC)
REQUIRE_PLC = False

# Fingerprint Sub-Mode: 'AUTO' (Search CSV) or 'MANUAL' (Locked on Target)
FINGERPRINT_MODE_TYPE = 'AUTO'

# OPC UA Connection (Simulated or Real PLC)
# Use 'localhost' if running the simulator on this PC.
OPC_URL = "opc.tcp://localhost:53530/OPCUA/SimulationServer"

# ==============================================================================
# 2. INFLUXDB SETTINGS
# ==============================================================================
DB_URL = "http://localhost:8086"
DB_TOKEN = "3RLVXp9GFpNSjpmIs00a1or2F1NgtYK2YITt0pk6N8XDE1KJhLH4-c50QJxowYJk0mQGf4MGThB-RilNDnHXxA=="
DB_ORG = "Innomotics"
DB_BUCKET = "test"

# Measurements (Tables)
DB_MEASUREMENT = "kiln1"
DB_MEASUREMENT_SETPOINTS = "kiln2"
DB_MEASUREMENT_AUTH = "auth"

# ==============================================================================
# 3. TIME & DATA SETTINGS
# ==============================================================================
RESAMPLE_INTERVAL = '1s'
FILL_METHOD = 'bfill'
TIME_VAR_OFFSET_MINUTES = 60
DEMO_END_DATETIME = datetime(2023, 11, 14, 15, 30, 30)

# Defaults
DEFAULT_PREVIOUS_TIME = 30
DEFAULT_FUTURE_TIME = 30

# CRITICAL: Matches your CSV column name
TIMESTAMP_COLUMN = "1_timeStamp"

# Algorithm Tuning
SIMILARITY_PLUS_THRESHOLD_PERCENT = 85
SIMILARITY_MINUS_LOWER_PCT = 0.90
SIMILARITY_MINUS_UPPER_PCT = 1.10

# ==============================================================================
# 4. TIMERS & DELAYS
# ==============================================================================
# The baseline interval for sending logic actions to the PLC (NN and Fingerprint nudges)
AI_INTERVAL_SECONDS = 30  

# The fast loop tick interval for updating the heartbeat/watchdog
FAST_CYCLE_SECONDS = 2    

# The delay between deep historical dataset scans in Fingerprint AUTO mode
SCAN_INTERVAL_SECONDS = 300

# ==============================================================================
# 5. FILE PATHS
# ==============================================================================
LOG_DIR = os.path.join(APP_DIR, "files", "logs")
JSON_DIR = os.path.join(APP_DIR, "files", "json")
DATA_DIR = os.path.join(APP_DIR, "files", "data")
MODELS_DIR = os.path.join(APP_DIR, "files", "models")

# Ensure dirs exist
for d in [LOG_DIR, JSON_DIR, DATA_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Specific Files
SOCKET_STATE_PATH = os.path.join(JSON_DIR, "socket.json")
PREVIOUS_FINGERPRINT_PATH = os.path.join(JSON_DIR, "previous_Fingerprint.json")
MIN_MAX_PATH = os.path.join(JSON_DIR, "min_max.json")
PREVIOUS_JSON_PATH = os.path.join(JSON_DIR, "previous.json")
TREND_FINGERPRINT_PATH = os.path.join(JSON_DIR, "trend_Fingerprint.json")
HISTORICAL_DATA_CSV_PATH = os.path.join(DATA_DIR, "fingerprint4.csv")
MODEL_CONFIG_PATH = os.path.join(JSON_DIR, "model_config.json")