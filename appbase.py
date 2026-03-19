import streamlit as st
import pandas as pd
#import ventas_app  # Ahora el nombre coincide
import plotly.express as px
import os

st.set_page_config(page_title="Dashboard de Ventas", layout="wide", page_icon=":material/analytics:")

# 1. Importar el motor que acabas de renombrar
try:
    import ventas_app
    st.sidebar.success("✅ Motor Rust cargado")
except ImportError:
    st.sidebar.error("❌ No se encontró ventas_app.pyd")




@st.cache_data
def cargar_datos():
    # Intenta cargar el archivo con el nombre exacto
    archivo = "data-db.xlsx" 
    if not os.path.exists(archivo):
        # Si no existe, busca cualquier archivo que empiece con 'data-db'
        archivos_en_carpeta = [f for f in os.listdir('.') if f.startswith('data-db')]
        if archivos_en_carpeta:
            archivo = archivos_en_carpeta[0]
        else:
            st.error("No se encontró el archivo de datos.")
            return None
    
    # Carga según la extensión
    if archivo.endswith('.csv'):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_excel(archivo)
        
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df

df = cargar_datos()

if df is not None:
    st.title("📊 Análisis de Ventas")

    # --- PROCESAMIENTO CON RUST ---
    # Aseguramos que los datos sean del tipo que Rust espera (String y f64)
    cats = df['product_category'].astype(str).tolist()
    sales = df['Sales'].astype(float).tolist()
    
    # Llamamos a la función de Rust
    dict_resultados = ventas_app.calcular_ventas_por_categoria(cats, sales)
    
    # Convertimos a DataFrame para Plotly
    df_rust = pd.DataFrame(list(dict_resultados.items()), columns=['Categoría', 'Total Ventas'])
    df_rust = df_rust.sort_values('Total Ventas', ascending=False)

    # --- DASHBOARD VISUAL ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ventas por Categoría (Calculado en Rust)")
        fig_bar = px.bar(df_rust, x='Categoría', y='Total Ventas', color='Categoría')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Distribución de Ventas por País")
        fig_pie = px.pie(df, names='store_location', values='Sales', hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Evolución Temporal")
    df_temp = df.groupby('transaction_date')['Sales'].sum().reset_index()
    fig_line = px.line(df_temp, x='transaction_date', y='Sales')
    st.plotly_chart(fig_line, use_container_width=True)