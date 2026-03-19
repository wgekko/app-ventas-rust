import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import datetime
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import numpy as np
import os
import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression




# Configuración inicial de la página
st.set_page_config(page_title="BI Dashboard - Ventas Pro", page_icon=":material/analytics:",  layout="wide")

# --- 1. CONEXIÓN CON EL MOTOR RUST ---
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

# --- 3. LÓGICA PRINCIPAL DEL DASHBOARD ---
df = cargar_datos()

if df is not None:
    st.header(":material/analytics:  Business Intelligence & Sales Analytics")
    
    # Barra lateral de estado
    if MOTOR_LISTO:
        st.sidebar.success(":material/bookmark_check: Motor de Cálculo Rust: Activo")
    else:
        st.sidebar.error(":material/error: Motor Rust: No encontrado")
        st.sidebar.info("Copia 'ventas_app.pyd' a la carpeta del proyecto.")

    # --- FILA 1: MÉTRICAS KPI (Usando Rust si está disponible) ---
    st.subheader(":material/chart_data: Métricas de Rendimiento Global")
    
    if MOTOR_LISTO:
        # Usamos Rust para agrupar categorías
        res_rust = ventas_app.calcular_ventas_por_categoria(
            df['product_category'].astype(str).tolist(), 
            df['Sales'].astype(float).tolist()
        )
        df_cat = pd.DataFrame(list(res_rust.items()), columns=['Categoría', 'Ventas']).sort_values('Ventas', ascending=False)
        top_cat = df_cat.iloc[0]['Categoría']
    else:
        # Fallback a Pandas si Rust no está
        df_cat = df.groupby('product_category')['Sales'].sum().reset_index()
        df_cat.columns = ['Categoría', 'Ventas']
        df_cat = df_cat.sort_values('Ventas', ascending=False)
        top_cat = "N/A"

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Ventas Totales", f"$ {df['Sales'].sum():,.2f}")
    col_b.metric("Ticket Promedio", f"$ {df['Sales'].mean():,.2f}")
    col_c.metric("Transacciones", f"{len(df):,}")
    col_d.metric("Categoría Estrella", top_cat)

    st.markdown("---")

    # --- FILA 2: GRÁFICOS DINÁMICOS Y ANIMACIÓN ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader(":material/pie_chart: Mix de Ventas por Categoría")
        fig_pie = px.pie(df_cat, values='Ventas', names='Categoría', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader(":material/browse_activity: Evolución Temporal por Mes")
        # Preparamos datos para animación
        df['Mes'] = df['transaction_date'].dt.strftime('%Y-%m')
        df_mes = df.groupby(['Mes', 'product_category'])['Sales'].sum().reset_index().sort_values('Mes')
        
        fig_anim = px.bar(df_mes, x="product_category", y="Sales", color="product_category",
                          animation_frame="Mes", range_y=[0, df_mes['Sales'].max() * 1.2])
        st.plotly_chart(fig_anim, use_container_width=True)

    st.markdown("---")

    # --- FILA 3: PREDICCIÓN DE VENTAS (SCIKIT-LEARN) ---
    st.subheader(":material/finance_mode: Predicción de Tendencia (Próximos 30 días)")
    
    # Preparar datos para Regresión
    df_daily = df.groupby('transaction_date')['Sales'].sum().reset_index()
    df_daily['date_ordinal'] = df_daily['transaction_date'].map(datetime.datetime.toordinal)
    
    X = df_daily[['date_ordinal']].values
    y = df_daily['Sales'].values
    
    model = LinearRegression().fit(X, y)
    
    # Proyectar futuro
    last_date = df_daily['transaction_date'].max()
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 31)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    predictions = model.predict(future_ordinals)
    
    # Unir datos para gráfico
    df_pred = pd.DataFrame({'Fecha': future_dates, 'Sales': predictions, 'Tipo': 'Predicción'})
    df_hist = df_daily[['transaction_date', 'Sales']].copy()
    df_hist.columns = ['Fecha', 'Sales']
    df_hist['Tipo'] = 'Histórico'
    df_final = pd.concat([df_hist, df_pred])

    fig_forecast = px.line(df_final, x='Fecha', y='Sales', color='Tipo', 
                           line_dash='Tipo', title="Modelo de Regresión Lineal aplicado")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # --- FILA 4: PARETO Y DECISIÓN ---
    st.header(":material/ssid_chart: Análisis de Pareto (80/20)")
    df_cat['Venta_Acum'] = df_cat['Ventas'].cumsum()
    df_cat['Perc_Acum'] = (df_cat['Venta_Acum'] / df_cat['Ventas'].sum()) * 100
    
    fig_pareto = px.bar(df_cat, x='Categoría', y='Ventas')
    fig_pareto.add_scatter(x=df_cat['Categoría'], y=df_cat['Perc_Acum'], name='% Acumulado', yaxis='y2')
    fig_pareto.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 105]))
    st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown("---")

    # --- FILA 2: PREDICCIÓN ESTADÍSTICA (ARIMA) ---
    st.header(":material/display_settings: Pronóstico de Series Temporales (ARIMA)")
    
    # Preparamos la serie temporal diaria
    df_ts = df.groupby('transaction_date')['Sales'].sum().asfreq('D').fillna(0)
    
    try:
        # Modelo ARIMA (1,1,1) - Auto-regresivo con tendencia
        model_arima = ARIMA(df_ts, order=(1, 1, 1))
        model_fit = model_arima.fit()
        forecast_arima = model_fit.get_forecast(steps=30)
        df_fore = forecast_arima.summary_frame()

        fig_arima = go.Figure()
        # Histórico real
        fig_arima.add_trace(go.Scatter(x=df_ts.index, y=df_ts, name="Histórico", line=dict(color="#1f77b4")))
        # Proyección central
        fig_arima.add_trace(go.Scatter(x=df_fore.index, y=df_fore['mean'], name="Proyección ARIMA", line=dict(color="#ff7f0e", dash='dash')))
        # Bandas de confianza (95%)
        fig_arima.add_trace(go.Scatter(x=df_fore.index, y=df_fore['mean_ci_upper'], fill=None, mode='lines', line_color='rgba(255,127,14,0.1)', showlegend=False))
        fig_arima.add_trace(go.Scatter(x=df_fore.index, y=df_fore['mean_ci_lower'], fill='tonexty', mode='lines', line_color='rgba(255,127,14,0.1)', name="Intervalo de Confianza"))
        
        st.plotly_chart(fig_arima, use_container_width=True)
    except:
        st.warning(":material/warning: Datos insuficientes para generar el modelo ARIMA diario.")

    st.markdown("---")

    # --- FILA 3: RIESGO Y PROBABILIDAD (MONTE CARLO) ---
    st.header(":material/simulation: Simulación de Riesgo Monte Carlo")
    col_mc1, col_mc2 = st.columns([2, 1])
    
    with col_mc1:
        # Simulación de 1,000 caminos posibles basados en volatilidad histórica
        mu = df_ts.mean()
        sigma = df_ts.std()
        dias_sim = 30
        iteraciones = 1000
        
        simulaciones = np.random.normal(mu, sigma, (dias_sim, iteraciones))
        caminos_acumulados = np.cumsum(simulaciones, axis=0)
        
        fig_mc = go.Figure()
        for i in range(min(iteraciones, 60)): # Mostramos 60 caminos para claridad
            fig_mc.add_trace(go.Scatter(y=caminos_acumulados[:, i], mode='lines', line=dict(width=0.5), opacity=0.2, showlegend=False))
        
        fig_mc.update_layout(title=f"Evolución Probable de Ingresos Acumulados ({dias_sim} días)")
        st.plotly_chart(fig_mc, use_container_width=True)

    with col_mc2:
        st.subheader("Análisis de Escenarios")
        final_sales = caminos_acumulados[-1, :]
        meta_ventas = mu * dias_sim * 1.15 # Meta: superar el promedio actual en 15%
        prob_exito = (final_sales > meta_ventas).mean() * 100

        st.metric("Venta Esperada (Media)", f"$ {final_sales.mean():,.2f}")
        st.metric("Escenario de Riesgo (P10)", f"$ {np.percentile(final_sales, 10):,.2f}", delta="- Peor Caso", delta_color="inverse")
        st.metric("Escenario Optimista (P90)", f"$ {np.percentile(final_sales, 90):,.2f}", delta="+ Mejor Caso")
        
        st.markdown(f"**Probabilidad de superar meta (+15%):**")
        st.progress(int(prob_exito))
        st.write(f"Probabilidad calculada: **{prob_exito:.1f}%**")

    st.markdown("---")




else:
    st.error(":material/warning: No se pudo cargar el archivo de datos. Verifica que el archivo CSV/Excel esté en la carpeta.")