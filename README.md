# Maritime Predictive Maintenance — Streamlit App

This repository contains a Streamlit app (app.py) for predictive maintenance on marine engine telemetry.

## Overview

- Dataset (Kaggle): `/kaggle/input/preventive-maintenance-for-marine-engines/marine_engine_data.csv`
- Target: `maintenance_status` (binary / categorical). Also `failure_mode` is available for multi-class analysis.
- The app loads or trains a model, provides interactive EDA, shows model metrics, and allows single-record inference.

## File structure

See repository tree. Main entrypoint: `app.py`.

## Quick start (local)

1. Create Python 3.11.0 environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Place dataset at:
   local: data/marine_engine_data.csv
   or if deploying to Kaggle / Streamlit Cloud, use the provided Kaggle input path:
   /kaggle/input/preventive-maintenance-for-marine-engines/marine_engine_data.csv

3. Run locally:
   streamlit run app.py
