import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from utils.evaluating import evaluating_change_point
import os

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure


def load_data():
    all_files = []

    for root, dirs, files in os.walk("data/"):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))

    list_of_df = [pd.read_csv(file,
                              sep=';',
                              index_col='datetime',
                              parse_dates=True) for file in all_files if 'anomaly-free' not in file]

    return list_of_df


def train(list_of_df, n_estimators, contamination, bootstrap, max_samples):
    clf = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=0,
        n_jobs=-1,
        contamination=contamination,
        bootstrap=bootstrap)

    predicted_outlier, predicted_cp = [], []

    for df in list_of_df:
        X_train = df[:400].drop(['anomaly', 'changepoint'], axis=1)
        clf.fit(X_train)
        prediction = pd.Series(clf.predict(df.drop(['anomaly', 'changepoint'], axis=1)) * -1,
                               index=df.index).rolling(3).median().fillna(0).replace(-1, 0)

        predicted_outlier.append(prediction)

        prediction_cp = abs(prediction.diff())
        prediction_cp[0] = prediction[0]
        predicted_cp.append(prediction_cp)

    return predicted_outlier, predicted_cp


def evaluate():
    true_cp = [df.changepoint for df in data]
    true_outlier = [df.anomaly for df in data]

    f1, far, mar = evaluating_change_point(true_outlier, outlier_res, metric='binary', numenta_time='30 sec')
    add, missing = evaluating_change_point(true_cp, cp_res, metric='average_delay', numenta_time='30 sec')

    st.subheader('Anomaly prediction metrics')
    st.write('False Alarm Rate:', far)
    st.write('Missing Alarm Rate:', mar)
    st.write('F1:', f1)

    st.subheader('Changepoints metrics')
    st.write('Average delay:', add)
    st.write('A number of missed CPs:', missing)

    return true_outlier, true_cp


def plot_vis(true_outlier, true_cp, predicted_outlier, predicted_cp):
    fig = Figure()
    fig.subplots_adjust(hspace=0.5)

    ax = fig.subplots(nrows=2)
    predicted_outlier[1].plot(figsize=(12, 8), label='predictions', marker='o', markersize=5, ax=ax[0])
    true_outlier[1].plot(marker='o', markersize=2, ax=ax[0])
    ax[0].set_title('Anomalies')
    ax[0].legend()

    predicted_cp[0].plot(figsize=(12, 8), label='predictions', marker='o', markersize=5, ax=ax[1])
    true_cp[0].plot(marker='o', markersize=2, ax=ax[1])
    ax[1].set_title('Changepoints')
    ax[1].legend()
    st.pyplot(fig)


data = load_data()

_lock = RendererAgg.lock
apptitle = 'SKAB Isolation Forest'

st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:")

st.title('Anomaly detection')
st.markdown("""Use the menu at left to configure isolation forest""")
st.sidebar.markdown("## Model params")

n_estimators = st.sidebar.slider('n_estimators', 1, 1000, 100)
max_samples = st.sidebar.selectbox('max_samples', ['auto', 5, 10, 20])
bootstrap = st.sidebar.checkbox('bootstrap?', value=False)
contamination = st.sidebar.slider('contamination', 0.01, 0.5, 0.01)

if st.sidebar.button('Fit model'):
    outlier_res, cp_res = train(data, n_estimators, contamination, bootstrap, max_samples)
    true_outlier, true_cp = evaluate()
    plot_vis(true_outlier, true_cp, outlier_res, cp_res)
