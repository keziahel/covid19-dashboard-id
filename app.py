import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("Dashboard Analisis COVID-19 Indonesia")

@st.cache_data
def load_data():
    df = pd.read_csv("covid_19_indonesia_time_series_all.csv")
    df = df.dropna(subset=["Total Cases", "Total Deaths", "Total Recovered", 
                           "Population Density", "Case Fatality Rate", "Location"])
    df["Case Fatality Rate"] = df["Case Fatality Rate"].str.replace("%", "").astype(float)
    df_latest = df.sort_values("Date").groupby("Location").tail(1)

    features = ["Total Deaths", "Total Recovered", "Population Density", "Case Fatality Rate"]
    target = "Total Cases"
    X = df_latest[features]
    y = df_latest[target]
    model = RandomForestRegressor()
    model.fit(X, y)
    df_latest["Predicted Total Cases"] = model.predict(X).astype(int)

    X_clust = df_latest[["Total Cases", "Total Deaths", "Total Recovered", "Population Density"]]
    X_scaled = StandardScaler().fit_transform(X_clust)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
    df_latest["Cluster"] = kmeans.labels_

    return df, df_latest

df_all, df_latest = load_data()

st.subheader("Peta Interaktif Clustering")
fig_map = px.scatter_geo(df_latest,
                         locations="Location",
                         locationmode="country names",
                         color="Cluster",
                         size="Total Cases",
                         title="Hasil Clustering Wilayah Berdasarkan Kasus COVID-19")
st.plotly_chart(fig_map)

st.subheader("Tren Kasus Harian")
selected_location = st.selectbox("Pilih Lokasi", df_all["Location"].unique())
df_trend = df_all[df_all["Location"] == selected_location].sort_values("Date")
fig_trend = px.line(df_trend, x="Date", y="New Cases", title=f"Tren Kasus Harian di {selected_location}")
st.plotly_chart(fig_trend)

st.subheader("Ringkasan Risiko Wilayah")
st.dataframe(df_latest[["Location", "Total Cases", "Predicted Total Cases", "Cluster"]].reset_index(drop=True))