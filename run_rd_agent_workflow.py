import os
import sys
import subprocess
import yaml
from pathlib import Path
import pandas as pd
import shutil

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_ROOT / "source"
NORMALIZED_DIR = DATA_ROOT / "normalized"
QLIB_DATA_DIR = DATA_ROOT / "qlib_data"
CONFIG_DIR = PROJECT_ROOT / "configs"

# Ensure directories exist
DATA_ROOT.mkdir(exist_ok=True)
SOURCE_DIR.mkdir(exist_ok=True)
NORMALIZED_DIR.mkdir(exist_ok=True)
QLIB_DATA_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

def run_command(cmd, env=None):
    print(f"Running: {cmd}")
    if env is None:
        env = os.environ.copy()
    # Add project root and qlib repo root to PYTHONPATH
    # PROJECT_ROOT is /Users/air/QT_China
    # We need to add /Users/air/QT_China/qlib so that 'import qlib' works correctly (resolving to qlib/qlib)
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT}/qlib:{env.get('PYTHONPATH', '')}"
    # Set Qlib version to avoid setuptools_scm lookup error
    env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_QLIB"] = "0.9.2"
    
    result = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    print(result.stdout)

def download_data():
    print("=== Step 1: Downloading Data from Baostock ===")
    script_path = PROJECT_ROOT / "qlib/scripts/data_collector/baostock_1d/collector.py"
    # Download data from 2020-01-01 to ensure enough history for training
    cmd = (
        f"{sys.executable} {script_path} download_data "
        f"--source_dir {SOURCE_DIR} "
        f"--start 2020-01-01 "
        f"--interval 1d "
        f"--region CN "
        f"--max_workers 4"
    )
    run_command(cmd)

def normalize_data():
    print("=== Step 2: Normalizing Data ===")
    script_path = PROJECT_ROOT / "qlib/scripts/data_collector/baostock_1d/collector.py"
    cmd = (
        f"{sys.executable} {script_path} normalize_data "
        f"--source_dir {SOURCE_DIR} "
        f"--normalize_dir {NORMALIZED_DIR} "
        f"--interval 1d "
        f"--region CN"
    )
    run_command(cmd)

def dump_data():
    print("=== Step 3: Dumping Data to Qlib Bin Format ===")
    script_path = PROJECT_ROOT / "qlib/scripts/dump_bin.py"
    cmd = (
        f"{sys.executable} {script_path} dump_all "
        f"--data_path {NORMALIZED_DIR} "
        f"--qlib_dir {QLIB_DATA_DIR} "
        f"--freq day "
        f"--exclude_fields date,symbol "
        f"--file_suffix .csv"
    )
    run_command(cmd)
    
    # Copy all.txt to csi300.txt to simulate a benchmark
    instruments_dir = QLIB_DATA_DIR / "instruments"
    if (instruments_dir / "all.txt").exists():
        shutil.copy(instruments_dir / "all.txt", instruments_dir / "csi300.txt")
        print("Created csi300.txt from all.txt")

def create_workflow_config(model_type="lightgbm"):
    print(f"=== Creating Workflow Config for {model_type} ===")
    
    # Define time ranges
    # Train: 2020 - 2022
    # Valid: 2023
    # Test: 2024 - Now
    
    port_analysis_config = {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": "<PRED>",
                "topk": 50,
                "n_drop": 5
            }
        },
        "backtest": {
            "start_time": "2024-01-01",
            "end_time": "2024-11-01",
            "account": 100000000,
            "benchmark": "SH600000",
            "exchange_kwargs": {
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5
            }
        }
    }

    config = {
        "qlib_init": {
            "provider_uri": str(QLIB_DATA_DIR),
            "region": "cn",
        },
        "market": "csi300",
        "benchmark": "csi300",
        "data_handler_config": {
            "start_time": "2020-01-01",
            "end_time": "2024-11-01",  # Adjust dynamically if needed
            "fit_start_time": "2020-01-01",
            "fit_end_time": "2022-12-31",
            "instruments": "csi300",
            "infer_processors": [
                 {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}}
            ],
            "learn_processors": [
                 {"class": "DropnaLabel"},
                 {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
            ],
            "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
        },
        "port_analysis_config": port_analysis_config,
        "task": {
            "model": {
                "class": "LGBModel" if model_type == "lightgbm" else "XGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "colsample_bytree": 0.8879,
                    "learning_rate": 0.0421,
                    "subsample": 0.8789,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": 8,
                    "num_leaves": 210,
                    "num_threads": 20,
                }
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "Alpha158",
                        "module_path": "qlib.contrib.data.handler",
                        "kwargs": {
                            "start_time": "2020-01-01",
                            "end_time": "2024-11-01",
                            "fit_start_time": "2020-01-01",
                            "fit_end_time": "2022-12-31",
                            "instruments": "csi300"
                        }
                    },
                    "segments": {
                        "train": ["2020-01-01", "2022-12-31"],
                        "valid": ["2023-01-01", "2023-12-31"],
                        "test": ["2024-01-01", "2024-11-01"]
                    }
                }
            },
            "record": [
                {
                    "class": "SignalRecord",
                    "module_path": "qlib.workflow.record_temp",
                    "kwargs": {
                        "model": "<MODEL>",
                        "dataset": "<DATASET>"
                    }
                },
                {
                    "class": "SigAnaRecord",
                    "module_path": "qlib.workflow.record_temp",
                    "kwargs": {
                        "ana_long_short": False,
                        "ann_scaler": 252
                    }
                },
                {
                    "class": "PortAnaRecord",
                    "module_path": "qlib.workflow.record_temp",
                    "kwargs": {
                        "config": port_analysis_config
                    }
                }
            ]
        }
    }
    
    if model_type == "xgboost":
        # XGBoost specific tweaks
        config["task"]["model"]["kwargs"].update({
             "eval_metric": "rmse",
             # XGBoost parameters might differ slightly in naming, but Qlib's wrapper usually handles standard ones
        })

    config_path = CONFIG_DIR / f"workflow_config_{model_type}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path

def run_workflow(config_path, exp_name):
    print(f"=== Running Workflow: {exp_name} ===")
    script_path = PROJECT_ROOT / "qlib/qlib/workflow/cli.py"
    # Qlib's CLI usage: qrun <config_path>
    # We can use `qrun` if installed, or python -m qlib.workflow.cli
    # Since we want to use the local qlib, we might need to run the script directly if available, 
    # or use python -m qlib.workflow.cli if PYTHONPATH is set.
    
    # Let's try running via qlib.cli.run module
    cmd = f"{sys.executable} -m qlib.cli.run {config_path} {exp_name}"
    run_command(cmd)

if __name__ == "__main__":
    # 1. Data Preparation
    # download_data() # Commented out to save time if already downloaded
    # normalize_data() # Commented out to save time
    dump_data()
    
    # 2. Run LightGBM
    lgb_config = create_workflow_config("lightgbm")
    run_workflow(lgb_config, "rd_agent_lightgbm")
    
    # 3. Run XGBoost
    xgb_config = create_workflow_config("xgboost")
    run_workflow(xgb_config, "rd_agent_xgboost")
    
    print("\n=== All Workflows Completed ===")
    print("Results can be found in 'mlruns' directory.")
    print("Use 'mlflow ui' to view the results and charts.")

