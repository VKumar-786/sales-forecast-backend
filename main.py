
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-and-forecast")
async def upload_and_forecast(file: UploadFile = File(...)):
    try:
        data = pd.read_csv(file.file)

        if 'Month' not in data.columns or 'Sales' not in data.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'Month' and 'Sales' columns.")

        data.drop_duplicates(inplace=True)
        data.dropna(inplace=True)

        try:
            data['Month'] = pd.to_datetime(data['Month'], errors='raise', dayfirst=True)
        except:
            try:
                data['Month'] = pd.to_datetime(data['Month'].astype(str) + ' 2023', format='%B %Y')
            except:
                try:
                    data['Month'] = pd.to_datetime(data['Month'].astype(str) + ' 2023', format='%b %Y')
                except:
                    try:
                        data['Month'] = pd.to_datetime(data['Month'].astype(str), format='%m', errors='coerce')
                        data['Month'] = data['Month'].fillna(pd.to_datetime('2023-' + data['Month'].astype(str).str.zfill(2) + '-01', errors='coerce'))
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Error converting Month data: {str(e)}")

        if data['Month'].isnull().any():
            raise HTTPException(status_code=400, detail="Some dates could not be parsed. Please check your data.")

        data.set_index('Month', inplace=True)
        train = data.iloc[:int(0.75 * len(data))]
        test = data.iloc[int(0.75 * len(data)):]        

        # ARIMA
        arima_model = ARIMA(data['Sales'], order=(1,1,1))
        arima_fit = arima_model.fit()
        forecast_arima = arima_fit.forecast(steps=len(test))

        # Linear Regression
        X_train = np.arange(len(train)).reshape(-1, 1)
        X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
        y_train = train['Sales'].values
        y_test = test['Sales'].values

        lr = LinearRegression().fit(X_train, y_train)
        lr_pred = lr.predict(X_test)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        # Metrics
        metrics = {
            "ARIMA": {
                "RMSE": float(np.sqrt(mean_squared_error(y_test, forecast_arima))),
                "R2": float(r2_score(y_test, forecast_arima)),
                "Accuracy": float(100 - mean_absolute_percentage_error(y_test, forecast_arima) * 100)
            },
            "Linear Regression": {
                "RMSE": float(np.sqrt(mean_squared_error(y_test, lr_pred))),
                "R2": float(r2_score(y_test, lr_pred)),
                "Accuracy": float(100 - mean_absolute_percentage_error(y_test, lr_pred) * 100)
            },
            "Random Forest": {
                "RMSE": float(np.sqrt(mean_squared_error(y_test, rf_pred))),
                "R2": float(r2_score(y_test, rf_pred)),
                "Accuracy": float(100 - mean_absolute_percentage_error(y_test, rf_pred) * 100)
            }
        }

        # Chart
        fig, ax = plt.subplots()
        models = list(metrics.keys())
        accuracies = [metrics[m]['Accuracy'] for m in models]
        bars = ax.bar(models, accuracies, color=['skyblue', 'orange', 'green'])
        ax.set_ylim([0, 100])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Forecasting Model Accuracy Comparison")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height - 5, f'{height:.1f}%', ha='center', va='bottom', color='white')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return JSONResponse({
            "metrics": metrics,
            "chart_base64": img_base64
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
