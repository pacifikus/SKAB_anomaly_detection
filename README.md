# SKAB anomaly detection

The [SKAB](https://github.com/waico/SKAB) v0.9 corpus contains 35 individual data files in .csv format. Each file represents a single experiment and contains a single anomaly. The dataset represents a multivariate time series collected from the sensors installed on the testbed. The data folder contains datasets from the benchmark. The structure of the data folder is presented in the structure file. Columns in each data file are following:

- **datetime** - Represents dates and times of the moment when the value is written to the database (YYYY-MM-DD hh:mm:ss)

- **Accelerometer1RMS** - Shows a vibration acceleration (Amount of g units)

- **Accelerometer2RMS** - Shows a vibration acceleration (Amount of g units)

- **Current** - Shows the amperage on the electric motor (Ampere)

- **Pressure** - Represents the pressure in the loop after the water pump (Bar)

- **Temperature** - Shows the temperature of the engine body (The degree Celsius)

- **Thermocouple** - Represents the temperature of the fluid in the circulation loop (The degree Celsius)

- **Voltage** - Shows the voltage on the electric motor (Volt)

- **RateRMS** - Represents the circulation flow rate of the fluid inside the loop (Liter per minute)

- **anomaly** - Shows if the point is anomalous (0 or 1)

- **changepoint** - Shows if the point is a changepoint for collective anomalies (0 or 1)


This project allows train anomaly detection model (isolation forest) in an interactive way via [Streamlit](https://streamlit.io/) framework.
