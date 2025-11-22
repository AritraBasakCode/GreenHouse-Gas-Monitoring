Directory Architecture: 

GHG-Prediction-Project/
│
├── data_pipeline/
│   ├── fetch_realtime_data.py
│   ├── merge_clean_data.py
│   └── ghg_database.db   (auto-generated)
│
├── model/
│   ├── train_lstm_model.py
│   ├── train_prophet_model.py
│   └── saved_models/
│       ├── lstm_model.h5
│       └── prophet_model.json
│
├── dashboard/
│   └── app.py
│
└── utils/
    ├── config.py
    └── helpers.py


Bash commands:

pip install pandas numpy requests scikit-learn tensorflow keras matplotlib joblib sqlite3 prophet fastapi uvicorn flask

{If Prophet installation fails, try:
	pip install prophet --use-pep517
}

Order to run the codes:

python fetch_realtime_data.py
python merge_clean_data.py
python train_lstm_model.py
python train_prophet_model.py
python app.py
