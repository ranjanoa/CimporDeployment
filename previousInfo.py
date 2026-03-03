# Copyright © 2025 INNOMOTICS
# previousInfo.py

import json
import warnings
import pandas as pd
from flask import Flask, jsonify, request, Blueprint
from datetime import datetime  # <--- Vital Import
import config
import process_model
import database
import control_service  # <--- NEW IMPORT

warnings.filterwarnings('ignore')
previous_info_routes = Blueprint('previous', __name__)


@previous_info_routes.route('/previous', methods=['GET'])
def previous():
    try:
        with open(config.PREVIOUS_JSON_PATH, "r") as read:
            return jsonify(json.load(read))
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({"previous_Time": 30, "future_Time": 30, "deviation": {}})


@previous_info_routes.route('/store_Previous', methods=['POST'])
def store_previous():
    try:
        # 1. Save the selection to a local file (Persistence)
        with open(config.SOCKET_STATE_PATH, 'w') as f:
            json.dump({"socket_stop": False}, f, indent=4)

        previous_data = request.get_json()
        with open(config.PREVIOUS_FINGERPRINT_PATH, "w") as data:
            json.dump(previous_data, data, indent=4)

        # 2. Prepare Data for InfluxDB Write-Back
        setpoint_tag_map = process_model.get_setpoint_tag_map()
        scale_factors = process_model.get_setpoint_scale_factors()

        if "data" not in previous_data or not previous_data["data"]:
            return {"result": "failure", "error": "No data received"}

        # Extract actions from the first batch
        actions = previous_data['data'][0]['actions']

        # Parse Setpoints for Database
        setpoints_dict = {}
        for action in actions:
            # Handle both naming conventions
            var_name = action.get('var_name')
            val = action.get('fingerprint_set_point')
            if val is None:
                val = action.get('setpoint')

            # Ensure valid float
            if var_name and val is not None:
                try:
                    setpoints_dict[var_name] = float(val)
                except ValueError:
                    continue

        if not setpoints_dict:
            return {"result": "success", "message": "No writable setpoints found"}

        # 3. Write to InfluxDB using UTC Time
        utc_timestamp = datetime.utcnow()
        db_success = database.write_setpoints(utc_timestamp, setpoints_dict, setpoint_tag_map, scale_factors)

        # 4. CRITICAL: Write to PLC Immediately (Manual Override)
        # Bypasses the Autopilot "Enabled" check
        plc_success = control_service.service.write_immediate(actions)

        if plc_success:
            print(f"✅ Command Written to InfluxDB & PLC at {utc_timestamp} UTC")
            return {"result": "success"}
        else:
            return {"result": "partial_success", "message": "Written to DB, but PLC Write Failed"}

    except Exception as e:
        print(f"❌ Store Previous Error: {e}")
        return {"result": "failure", "error": str(e)}