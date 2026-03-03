# Copyright © 2025 INNOMOTICS
# database.py
# Handles InfluxDB 2.0 Communication using Flux queries.

import config
import pandas as pd
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS


def get_db_client():
    """Initializes the InfluxDB 2.0 Client."""
    try:
        client = InfluxDBClient(
            url=config.DB_URL,
            token=config.DB_TOKEN,
            org=config.DB_ORG
        )
        return client
    except Exception as e:
        print(f"Error connecting to InfluxDB: {e}")
        return None


def _rename_and_format_df(df, tag_map):
    """Formats the raw InfluxDB DataFrame into the platform format."""
    # InfluxDB 2.0 returns columns like '_time', '_value', '_field'.
    # We need to pivot this so fields become columns.

    if df.empty: return df

    # Drop internal Influx columns we don't need
    cols_to_drop = ['result', 'table', '_start', '_stop', '_measurement']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Rename _time to timestamp
    df = df.rename(columns={'_time': config.TIMESTAMP_COLUMN})

    # If the data isn't pivoted yet (long format), pivot it
    if '_field' in df.columns and '_value' in df.columns:
        df = df.pivot_table(index=config.TIMESTAMP_COLUMN, columns='_field', values='_value')
        df = df.reset_index()

    # Rename columns using the tag map
    df = df.rename(columns=tag_map)

    # Ensure timestamp is datetime
    df[config.TIMESTAMP_COLUMN] = pd.to_datetime(df[config.TIMESTAMP_COLUMN])

    # Resample to ensure regular intervals
    df = df.set_index(config.TIMESTAMP_COLUMN).sort_index()
    df = df.resample(config.RESAMPLE_INTERVAL).first().fillna(method=config.FILL_METHOD).reset_index()

    return df


def get_realtime_data_window(start_time, end_time, process_tags, tag_map):
    """
    Reads data using a Flux query.
    """
    client = get_db_client()
    if not client: return pd.DataFrame()

    # Construct Flux Query
    # 1. Define Bucket & Time Range
    # 2. Filter by Measurement
    # 3. Filter by Specific Fields (Tags)
    # 4. Pivot to make it a wide table (like CSV)

    # Convert list of tags to Flux filter string: r["_field"] == "TagA" or r["_field"] == "TagB"
    field_filters = ' or '.join([f'r["_field"] == "{tag}"' for tag in process_tags])

    query = f'''
    from(bucket: "{config.DB_BUCKET}")
      |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
      |> filter(fn: (r) => r["_measurement"] == "{config.DB_MEASUREMENT}")
      |> filter(fn: (r) => {field_filters})
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''

    try:
        # Query directly into DataFrame
        df = client.query_api().query_data_frame(org=config.DB_ORG, query=query)

        # Handle case where list of DFs is returned
        if isinstance(df, list):
            df = pd.concat(df) if df else pd.DataFrame()

        if not df.empty:
            return _rename_and_format_df(df, tag_map)

        return pd.DataFrame()

    except Exception as e:
        print(f"Error executing Flux query: {e}")
        return pd.DataFrame()


def write_setpoints(timestamp, setpoints_dict, setpoint_tag_map, scale_factors):
    """
    Writes setpoints to InfluxDB 2.0.
    """
    client = get_db_client()
    if not client: return False

    write_api = client.write_api(write_options=SYNCHRONOUS)

    try:
        point = Point(config.DB_MEASUREMENT_SETPOINTS).time(timestamp)

        fields_added = False
        for name, value in setpoints_dict.items():
            tag = setpoint_tag_map.get(name)
            if tag:
                scale = scale_factors.get(name, 1)
                point.field(tag, float(value * scale))
                fields_added = True

        if fields_added:
            write_api.write(bucket=config.DB_BUCKET, org=config.DB_ORG, record=point)
            return True
        return False

    except Exception as e:
        print(f"Error writing setpoints: {e}")
        return False
    finally:
        write_api.close()
        client.close()