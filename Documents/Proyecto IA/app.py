import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el modelo entrenado
modelo = joblib.load("modelo_accidentes.pkl")

# Título
st.title("🚦 Predicción de Accidentes en EE.UU.")
st.markdown("Una aplicación interactiva para visualizar y predecir accidentes de tráfico con Machine Learning.")

# Cargar datos de prueba (solo una muestra)
st.subheader("📊 Datos de Entrada")
df = pd.read_csv("US_Accidents_March23.csv", nrows=1000)
st.write(df.head())

# Sidebar para entrada del usuario
st.sidebar.header("📌 Parámetros de Entrada")
hour = st.sidebar.slider("Hora del Día", 0, 23, 12)
day_of_week = st.sidebar.selectbox("Día de la Semana", ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"], index=0)
temperature = st.sidebar.number_input("Temperatura (°F)", min_value=-20, max_value=120, value=70)
humidity = st.sidebar.number_input("Humedad (%)", min_value=0, max_value=100, value=50)
visibility = st.sidebar.number_input("Visibilidad (millas)", min_value=0, max_value=50, value=10)
wind_speed = st.sidebar.number_input("Velocidad del Viento (mph)", min_value=0, max_value=100, value=5)

# Convertir el día de la semana a número
days_dict = {"Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3, "Viernes": 4, "Sábado": 5, "Domingo": 6}
day_of_week = days_dict[day_of_week]

# Crear DataFrame con la entrada del usuario
input_data = pd.DataFrame([[hour, day_of_week, temperature, humidity, visibility, wind_speed]],
                          columns=["hour", "day_of_week", "Temperature(F)", "Humidity(%)", "Visibility(mi)", "Wind_Speed(mph)"])

# Hacer predicción
if st.sidebar.button("📌 Predecir Severidad"):
    prediction = modelo.predict(input_data)[0]
    st.sidebar.write(f"🔹 Severidad Predicha: {prediction}")

# Visualización de Importancia de Variables
st.subheader("🔍 Importancia de Variables en la Predicción")
importances = modelo.feature_importances_
fig, ax = plt.subplots()
sns.barplot(x=input_data.columns, y=importances, palette="viridis", ax=ax)
ax.set_xlabel("Características")
ax.set_ylabel("Importancia")
ax.set_title("Importancia de Variables")
st.pyplot(fig)

# Visualización de Accidentes por Hora del Día
st.subheader("⏰ Accidentes por Hora del Día")
fig, ax = plt.subplots()
sns.countplot(x=df['Start_Time'].apply(lambda x: pd.to_datetime(x).hour), palette="coolwarm", ax=ax)
ax.set_xlabel("Hora del Día")
ax.set_ylabel("Cantidad de Accidentes")
ax.set_title("Accidentes por Hora del Día")
st.pyplot(fig)

st.markdown("---")
st.markdown("📌 *Proyecto desarrollado con Python, Machine Learning y Streamlit.*")
