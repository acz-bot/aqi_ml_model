from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly
import json

app = Flask(__name__)

# Load trained models and encoders
models = joblib.load("aqi_forecast_models.pkl")
city_encoder = joblib.load("city_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Example pollutant values (you can later connect to live data)
pollutant_defaults = {
    'PM2.5': 50, 'PM10': 80, 'NO': 10, 'NO2': 30, 'NOx': 40, 'NH3': 20,
    'CO': 0.8, 'SO2': 10, 'O3': 30, 'Benzene': 3, 'Toluene': 10, 'Xylene': 2,
    'Year': 2025, 'Month': 10, 'Day': 9, 'DayOfWeek': 3, 'Is_Weekend': 0,
    'AQI_Lag1': 120, 'AQI_Lag2': 125, 'AQI_Lag3': 130
}

def predict_aqi(city):
    # Create input dataframe
    input_data = pd.DataFrame([pollutant_defaults], columns=feature_names)
    input_data['City_encoded'] = city_encoder.transform([city])[0]

    preds = {t: round(models[t].predict(input_data)[0], 2) for t in models.keys()}
    return preds

def get_aqi_recommendation(aqi):
    if aqi <= 50:
        return ("#009966", "Good — Air quality is satisfactory.")
    elif aqi <= 100:
        return ("#FFDE33", "Moderate — Acceptable air quality.")
    elif aqi <= 200:
        return ("#FF9933", "Unhealthy for Sensitive Groups.")
    elif aqi <= 300:
        return ("#CC0033", "Unhealthy — Everyone may feel effects.")
    elif aqi <= 400:
        return ("#660099", "Very Unhealthy — Health warnings.")
    else:
        return ("#7E0023", "Hazardous — Serious health effects for everyone.")

@app.route('/')
def index():
    return render_template('index.html', pollutant_defaults=pollutant_defaults)

@app.route('/predict', methods=['POST'])
def predict():
    city = request.form['city']
    try:
        preds = predict_aqi(city)
        color, message = get_aqi_recommendation(preds['AQI_24hr'])

        # Plot AQI forecast
        hours = ['24h', '48h', '72h']
        aqi_values = [preds['AQI_24hr'], preds['AQI_48hr'], preds['AQI_72hr']]
        fig = go.Figure(data=[go.Scatter(x=hours, y=aqi_values, mode='lines+markers', line=dict(color=color, width=3))])
        fig.update_layout(title=f"AQI Forecast for {city}", xaxis_title="Time Horizon", yaxis_title="Predicted AQI", template="plotly_dark")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Pollutant bar chart
        pollutant_fig = go.Figure(data=[go.Bar(
            x=list(pollutant_defaults.keys())[:10],
            y=list(pollutant_defaults.values())[:10],
            marker_color='skyblue'
        )])
        pollutant_fig.update_layout(title="Current Pollutant Levels", template="plotly_dark")
        pollutant_graph = json.dumps(pollutant_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Redirect to forecast_dashboard with city as query param
        return render_template('index.html', city=city, preds=preds, color=color, message=message, graphJSON=graphJSON, pollutant_graph=pollutant_graph, pollutant_defaults=pollutant_defaults)
    except ValueError:
        error_message = f"Sorry, the city '{city}' is not supported by the model."
        return render_template('index.html', error_message=error_message, pollutant_defaults=pollutant_defaults)

@app.route('/forecast_dashboard')
def forecast_dashboard():
    city = request.args.get('city', 'Delhi')
    preds = predict_aqi(city)
    color, message = get_aqi_recommendation(preds['AQI_24hr'])

    # AQI forecast line chart
    hours = ['24h', '48h', '72h']
    aqi_values = [preds['AQI_24hr'], preds['AQI_48hr'], preds['AQI_72hr']]
    fig = go.Figure(data=[go.Scatter(x=hours, y=aqi_values, mode='lines+markers', line=dict(color=color, width=3))])
    fig.update_layout(title=f"AQI Forecast for {city}", xaxis_title="Time Horizon", yaxis_title="Predicted AQI", template="plotly_dark")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Pollutant pie chart
    pollutant_keys = list(pollutant_defaults.keys())[:10]
    pollutant_values = list(pollutant_defaults.values())[:10]
    pie_colors = ['#009966', '#FFDE33', '#FF9933', '#CC0033', '#660099', '#7E0023', '#1E90FF', '#FFD700', '#32CD32', '#FF69B4']
    pie_fig = go.Figure(data=[go.Pie(labels=pollutant_keys, values=pollutant_values, marker=dict(colors=pie_colors), hole=0.3)])
    pie_fig.update_layout(title="Pollutant Distribution", template="plotly_dark")
    pieJSON = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('forecast_dashboard.html', city=city, preds=preds, color=color, message=message, graphJSON=graphJSON, pieJSON=pieJSON)

if __name__ == "__main__":
    app.run(debug=True)
    app.run(debug=True)
