import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Quant Sales Dashboard", layout="wide")

# --- CONEXIÓN RUST ---
try:
    import ventas_app
    MOTOR_LISTO = True
except ImportError:
    MOTOR_LISTO = False

# --- 2. FUNCIÓN DE CARGA DE DATOS ---
@st.cache_data
def cargar_datos():
    # Buscamos el archivo que subiste (CSV con nombre de Excel)
    archivo = "data-db.xlsx - Hoja1.csv" 
    
    if not os.path.exists(archivo):
        # Intento secundario por si cambiaste el nombre
        archivos_posibles = [f for f in os.listdir('.') if f.startswith('data-db')]
        if archivos_posibles:
            archivo = archivos_posibles[0]
        else:
            return None

    try:
        if archivo.endswith('.csv'):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo, engine='openpyxl')
        
        # Convertir fecha y limpiar nulos
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Error al leer los datos: {e}")
        return None


df = cargar_datos()

if df is not None:
    st.title("📈 Advanced Sales Analytics & Forecasting")
    
    # --- PROCESAMIENTO RUST (KPIs) ---
    cats = df['product_category'].astype(str).tolist()
    sales = df['Sales'].astype(float).tolist()
    
    if MOTOR_LISTO:
        res_rust = ventas_app.calcular_ventas_por_categoria(cats, sales)
        df_cat = pd.DataFrame(list(res_rust.items()), columns=['Cat', 'Ventas']).sort_values('Ventas', ascending=False)
    else:
        df_cat = df.groupby('product_category')['Sales'].sum().reset_index()

    # --- SECCIÓN 1: MODELO ARIMA ---
    st.header("1. Pronóstico Avanzado (ARIMA)")
    
    # Agrupamos por día y preparamos serie temporal
    df_ts = df.groupby('transaction_date')['Sales'].sum().asfreq('D').fillna(0)
    
    try:
        # Ajustamos un modelo ARIMA simple (1,1,1)
        # p=1 (dependencia del ayer), d=1 (tendencia), q=1 (error residual)
        model_arima = ARIMA(df_ts, order=(1, 1, 1))
        model_fit = model_arima.fit()
        forecast_arima = model_fit.get_forecast(steps=30)
        df_fore = forecast_arima.summary_frame()

        fig_arima = go.Figure()
        # Histórico
        fig_arima.add_trace(go.Scatter(x=df_ts.index, y=df_ts, name="Histórico", line=dict(color="#1f77b4")))
        # Predicción
        fig_arima.add_trace(go.Scatter(x=df_fore.index, y=df_fore['mean'], name="Predicción ARIMA", line=dict(color="#ff7f0e", dash='dash')))
        # Intervalo de confianza
        fig_arima.add_trace(go.Scatter(x=df_fore.index, y=df_fore['mean_ci_upper'], fill=None, mode='lines', line_color='rgba(255,127,14,0.1)', showlegend=False))
        fig_arima.add_trace(go.Scatter(x=df_fore.index, y=df_fore['mean_ci_lower'], fill='tonexty', mode='lines', line_color='rgba(255,127,14,0.1)', name="Confianza 95%"))
        
        st.plotly_chart(fig_arima, use_container_width=True)
    except:
        st.warning("Datos insuficientes para el modelo ARIMA. Se requiere una serie temporal continua.")

    st.markdown("---")

    # --- SECCIÓN 2: SIMULACIÓN MONTE CARLO (PROBABILIDAD) ---
    st.header("2. Simulación de Escenarios (Monte Carlo)")
    
    col_m1, col_m2 = st.columns([2, 1])
    
    with col_m1:
        # Parámetros de la simulación basados en stats reales
        mu = df_ts.mean()
        sigma = df_ts.std()
        dias_sim = 30
        iteraciones = 1000
        
        # Generamos 1000 caminos aleatorios
        simulaciones = np.random.normal(mu, sigma, (dias_sim, iteraciones))
        caminos_acumulados = np.cumsum(simulaciones, axis=0)
        
        # Graficar una muestra de los caminos
        fig_mc = go.Figure()
        for i in range(min(iteraciones, 50)): # Dibujamos solo 50 para no saturar
            fig_mc.add_trace(go.Scatter(y=caminos_acumulados[:, i], mode='lines', 
                                      line=dict(width=0.5), opacity=0.3, showlegend=False))
        
        fig_mc.update_layout(title=f"1,000 Proyecciones de Ventas Acumuladas (Próximos {dias_sim} días)")
        st.plotly_chart(fig_mc, use_container_width=True)

    with col_m2:
        st.subheader("Análisis de Riesgo")
        final_sales = caminos_acumulados[-1, :]
        prob_meta = (final_sales > (mu * dias_sim * 1.1)).mean() * 100 # Probabilidad de superar la media + 10%
        
        st.metric("Venta Final Esperada (Promedio)", f"$ {final_sales.mean():,.2f}")
        st.metric("Escenario Pesimista (P10)", f"$ {np.percentile(final_sales, 10):,.2f}")
        st.metric("Escenario Optimista (P90)", f"$ {np.percentile(final_sales, 90):,.2f}")
        
        st.write(f"🎯 **Probabilidad de superar la meta:** {prob_meta:.1f}%")

    st.markdown("---")

    # --- SECCIÓN 3: PARETO (CON RUST) ---
    st.header("3. Estructura de Ingresos (Pareto)")
    df_cat['Venta_Acum'] = df_cat['Ventas'].cumsum()
    df_cat['Perc_Acum'] = (df_cat['Venta_Acum'] / df_cat['Ventas'].sum()) * 100
    
    fig_pareto = px.bar(df_cat, x='Cat', y='Ventas', color='Ventas', title="Concentración de Ventas")
    fig_pareto.add_scatter(x=df_cat['Cat'], y=df_cat['Perc_Acum'], name="% Acumulado", yaxis="y2")
    fig_pareto.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 105]))
    st.plotly_chart(fig_pareto, use_container_width=True)

else:
    st.error("Archivo no encontrado. Asegúrate de tener 'data-db.xlsx - Hoja1.csv' en la carpeta.")