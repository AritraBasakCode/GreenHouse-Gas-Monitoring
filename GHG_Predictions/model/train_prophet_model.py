from prophet import Prophet
from utils.helpers import load_db_table
from utils.config import PROPHET_MODEL_PATH

df = load_db_table()
df_prophet = df[["timestamp", "co"]]
df_prophet.columns = ["ds", "y"]

model = Prophet()
model.fit(df_prophet)

model.save(PROPHET_MODEL_PATH)
print("Prophet model trained and saved!")
