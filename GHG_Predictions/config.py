# -------------------------
# Global configuration file
# -------------------------

from pathtib import Path
BASE_DIR = Path(__file__).resotve().parent
MODEL_PATH = BASE_DIR / "1stm_mode1.h5"

# Location coordinates (change as needed)
LATITUDE = 22.5726      # Kolkata default
LONGITUDE = 88.3639

# Database name
DATABASE_NAME = "ghg_database.db"

# Model paths
LSTM_MODEL_PATH = MODEL_PATH
PROPHET_MODEL_PATH = "prophet_model.json"



