"""
src/check_data_drift.py
Uses Evidently AI to check if our INPUT DATA's statistical properties
have changed between January (reference) and March (current).

Run from the project root as: python src/check_data_drift.py
"""

import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

import config


def main():
    reference = pd.read_csv(config.JANUARY_DATA_PATH)
    current = pd.read_csv(config.MARCH_DATA_PATH)

    report = Report([DataDriftPreset()])
    result = report.run(current_data=current, reference_data=reference)

    # Save a full interactive HTML report
    result.save_html("data_drift_report.html")
    print("Full report saved to data_drift_report.html - open it in a browser.\n")

    # Print a clean summary to the terminal
    result_dict = result.dict()
    print("=" * 60)
    print("DATA DRIFT SUMMARY (per-column statistical comparison)")
    print("=" * 60)
    for metric in result_dict["metrics"]:
        name = metric["metric_name"]
        if "DriftedColumnsCount" in name:
            print(f"Overall drifted columns: {metric['value']}")
        elif "ValueDrift" in name:
            column = metric["config"].get("column", "?")
            method = metric["config"].get("method", "?")
            p_value = metric["value"]
            drifted = "DRIFT DETECTED" if p_value < 0.05 else "no significant drift"
            print(f"  {column:20s} ({method:20s}) p={p_value:.4f} -> {drifted}")

    print("\nNOTE: Even if no columns show drift here, this only checks whether")
    print("individual feature distributions changed. It does NOT check whether")
    print("the RELATIONSHIP between features and the target has changed -")
    print("that's called concept drift, and it's what we check next.")


if __name__ == "__main__":
    main()
