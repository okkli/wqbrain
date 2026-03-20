from datetime import datetime, timezone
import pandas as pd
import os
import hashlib
import json
from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = os.getenv("CACHE_PATH")


def hash_alpha(alpha_dict: dict) -> str:
    alpha_str = json.dumps(alpha_dict, sort_keys=True)
    return hashlib.sha256(alpha_str.encode("utf-8")).hexdigest()

def create_simulation_cache():
    if not os.path.exists(CACHE_PATH):
        df = pd.DataFrame(columns=["alpha_hashed", "alpha_id", "date_created"])
        df.to_parquet(CACHE_PATH, index=False)

def add_to_cache(alpha_dict: dict, alpha_id: str = None):
    # Ensure cache exists
    if not os.path.exists(CACHE_PATH):
        create_simulation_cache()

    alpha_hashed = hash_alpha(alpha_dict)

    # Load existing cache
    t = pd.read_parquet(CACHE_PATH)

    # Append new record
    new_row = {
        "alpha_hashed": alpha_hashed,
        "alpha_id": alpha_id,
        "alpha_regular": alpha_dict['regular'],
        "date_created": datetime.now(timezone.utc),
    }

    t = pd.concat([t, pd.DataFrame([new_row])], ignore_index=True)
    t.to_parquet(CACHE_PATH, index=False)


def check_if_alpha_already_simulated(alpha_dict: dict) -> dict:
    if not os.path.exists(CACHE_PATH):
        create_simulation_cache()

    alpha_hashed = hash_alpha(alpha_dict)
    t = pd.read_parquet(CACHE_PATH)

    # 直接判断是否存在
    exists = alpha_hashed in t['alpha_hashed'].values
    if not exists:
        add_to_cache(alpha_dict)
        return False
    return True

if __name__ == "__main__":
    simulate = {
    "type": "REGULAR",
    "settings": {
      "instrumentType": "EQUITY",
      "region": "USA",
      "universe": "TOP3000",
      "delay": 1,
      "decay": 0,
      "neutralization": "STATISTICAL",
      "truncation": 0.08,
      "pasteurization": "ON",
      "testPeriod": "P0Y0M0D",
      "unitHandling": "VERIFY",
      "nanHandling": "ON",
      "maxTrade": "ON",
      "language": "FASTEXPR",
      "visualization": False
    },
    "regular": "-ts_corr(ts_decay_linear(snt22_2neg_mean_168, 10), ts_decay_linear(snt22_3pos_mean_116, 10), 30)"
  }
    print(check_if_alpha_already_simulated(simulate))