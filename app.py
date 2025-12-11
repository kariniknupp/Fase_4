import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
from joblib import load
from datetime import timedelta

# Configuração da página para layout mais amplo
st.set_page_config(layout="wide")

# ----------------------------------------------------
# 1. FUNÇÃO PARA CARREGAR O MODELO COM CACHING -- joblib
# ----------------------------------------------------
@st.cache_resource
def carregar_modelo(caminho_modelo):
    """Carrega o modelo salvo com joblib."""
    try:
        modelo = joblib.load(caminho_modelo)
        return modelo
    except FileNotFoundError:
        st.error(f"Erro: O arquivo de modelo '{caminho_modelo}' não foi encontrado. Certifique-se de que ele está no mesmo diretório do app.py.")
        st.stop()
        return None

# Carregamento do modelo
MODELO_ARQUIVO = 'modelo_XGB.joblib' 
modelo_ml = carregar_modelo(MODELO_ARQUIVO)

if modelo_ml is None:
    st.stop() # Para a execução se o modelo não puder ser carregado

# ----------------------------------------------------
# 2. CARREGAMETNO E PRÉ-PROCESSAMENTO DOS DADOS
# ----------------------------------------------------

# Carregamento e processamento dos dados históricos brutos
df= pd.read_csv('https://raw.githubusercontent.com/kariniknupp/Fase_4/refs/heads/main/Dados_Ibovespa_jan15_nov25.csv', parse_dates=['Data'], dayfirst=True)

df.rename(columns={'Data' : 'ds', 'Último': 'fechamento', 'Abertura': 'abertura', 'Máxima': 'max', 'Mínima': 'min', 'Vol.':'vol', 'Var%': 'var'}, inplace=True)
df.drop(columns={'var', 'vol'}, inplace=True)
df['ds'] = df['ds'].dt.date
df = df.sort_values(by='ds', ascending=True).reset_index(drop=True)

# Carrega o DataFrame PROCESSADO para identificar as features e o target 'y'
df_processado=pd.read_csv('https://raw.githubusercontent.com/kariniknupp/Fase_4/refs/heads/main/dados_processados.csv')
df_processado = df_processado.sort_values(by='ds', ascending=True).reset_index(drop=True)

# Identifica as colunas de Features (X) que o modelo espera
# EXCLUIMOS: 'ds', 'y' e 'fechamento' (que são o target e a data)
FEATURES = [col for col in df_processado.columns if col not in ['ds', 'y', 'fechamento']]

# ----------------------------------------------------
# 3. LAYOUT E INPUT DO USUÁRIO
# ----------------------------------------------------
st.title('ANÁLISE E PREVISÃO DE TENDÊNCIA DO IBOVESPA :moneybag:')

# ----------------------------------------------------
# 4. PAINEL DE MÉTRICAS (st.sidebar)
# ----------------------------------------------------

# Resultados obtidos pelo XGBOOST na atividade 2
METRICAS = {
    "Acurácia Direcional": 77.27,
    "R² Score": -0.061,
    "MAE (Erro Absoluto)": 0.908,
    "WMAPE": 1.035,
}

st.sidebar.title("📈 Métricas de Validação (XGBoost)")
st.sidebar.markdown("Performance do Modelo Treinado:")

st.sidebar.metric("Acurácia Direcional", f"{METRICAS['Acurácia Direcional']:.2f}%", delta=None) 
st.sidebar.metric("R² Score", f"{METRICAS['R² Score']:.3f}", delta=None) 
st.sidebar.metric("MAE", f"{METRICAS['MAE (Erro Absoluto)']:.3f}", delta=None)
st.sidebar.metric("WMAPE", f"{METRICAS['WMAPE']:.3f}", delta=None)

st.sidebar.markdown("---")
st.sidebar.info(f"O modelo está usando {len(FEATURES)} features preditoras.")

st.sidebar.info(f"O modelo prevê a tendência (+1 subida, -1 descida) com base em {len(FEATURES)} features.")


st.write('Período Histórico Analisado:', df['ds'].min(), 'a', df['ds'].max())


#Dados disponibilizados para análise

coluna1, coluna2, coluna3, coluna4 = st.columns(4)
with coluna1:
    st.metric('Quantidade de Dias Analisados', df['ds'].count())
with coluna2:
    st.metric('Máxima do Índice', df['max'].max())
with coluna3:
    st.metric('Mínima do Índice', df['min'].min())
with coluna4:
    st.metric('Média do Índice', df['fechamento'].mean().round(3))


# NOVO INPUT: RADIO BUTTON PARA SELEÇÃO DE DIAS
st.write('Escolha para quantos dias deseja a previsão de tendência:')
opcoes_dias = {
    'Próximo Dia (1)': 1,
    'Próximos 5 Dias': 5,
    'Próximos 10 Dias': 10
}

selecao = st.radio("Selecione a Janela de Previsão", list(opcoes_dias.keys()), horizontal=True)
input_qtd_dias = opcoes_dias[selecao]

# ----------------------------------------------------
# 5. GRÁFICO E TABELA DE PREVISÃO (FOCADOS)
# ----------------------------------------------------

if modelo_ml and df_processado is not None and not df_processado.empty:
    
    # --- GERAÇÃO DAS DATAS FUTURAS ---
    ultima_data_historica = pd.to_datetime(df_processado['ds'].max())
    datas_com_inicio = pd.date_range(
        start=ultima_data_historica, 
        periods=input_qtd_dias + 1, 
        freq='B' # 'B' para dias úteis (Business days)
    )
    datas_futuras = datas_com_inicio[1:]
    
    # --- PREPARAÇÃO DO DF FUTURO ---
    ultimo_df_historico = df_processado.iloc[[-1]].copy()
    df_futuro = pd.DataFrame()
    df_futuro['ds'] = datas_futuras
    
    for feature in FEATURES:
         df_futuro[feature] = ultimo_df_historico[feature].iloc[0]
         
    # --- PREVISÃO ---
    X_futuro = df_futuro[FEATURES]
    previsoes = modelo_ml.predict(X_futuro)
    df_futuro['y_pred'] = previsoes
    
    
    # ------------------------------------------------------------------
    # 5.1 VISUALIZAÇÃO DO GRÁFICO (Foco nos últimos 30 dias + Previsão)
    # ------------------------------------------------------------------
    
    st.header(f"Projeção de Tendência ({input_qtd_dias} Dias) - Resultado: {selecao}")
    
    # Prepara os dados históricos (LIMITADO A 30 DIAS)
    DIAS_HISTORICOS_A_MOSTRAR = 30
    df_historico_plot = df_processado[['ds', 'y']].copy().rename(columns={'y': 'Tendência'})
    df_historico_plot['Tipo'] = 'Histórico (Real)'
    
    # Prepara os dados de previsão
    df_futuro_plot = df_futuro[['ds', 'y_pred']].copy().rename(columns={'y_pred': 'Tendência'})
    df_futuro_plot['Tipo'] = 'Previsão'
    
    # Seleciona apenas os últimos 30 dias do histórico
    df_historico_slice = df_historico_plot.tail(DIAS_HISTORICOS_A_MOSTRAR)
    
    # Junta os dois DataFrames (slice do histórico + previsão)
    df_combinado_visualizacao = pd.concat([df_historico_slice, df_futuro_plot])
    df_combinado_visualizacao['ds'] = pd.to_datetime(df_combinado_visualizacao['ds'])
    
    # Cria o gráfico Plotly FOCADO
    fig = px.line(
        df_combinado_visualizacao, 
        x='ds', 
        y='Tendência', 
        color='Tipo', 
        title='Histórico Recente e Previsão de Tendência (+1 Sobe, -1 Desce)',
        labels={'Tendência': 'Direção do Movimento', 'ds': 'Data'},
        color_discrete_map={'Histórico (Real)': '#1f77b4', 'Previsão': '#d62728'}
    )
    
    # Configurações de layout
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    fig.add_vline(x=ultima_data_historica, line_width=2, line_dash="dash", line_color="#333333")
    fig.update_layout(yaxis=dict(
        tickvals=[-1, 0, 1], 
        ticktext=['-1 (Desce)', '0 (Neutro)', '+1 (Sobe)'],
        range=[-1.5, 1.5]
    ))
    fig.update_layout(hovermode="x unified")
    
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 5.2 TABELA DE PREVISÃO ESTILIZADA
    # ------------------------------------------------------------------
    st.subheader("Resultados da Previsão Detalhada")
    
    # Cria o DataFrame para a tabela
    df_tabela = df_futuro_plot.copy()
    df_tabela['Data'] = df_tabela['ds'].dt.strftime('%d/%m/%Y')
    
    # Mapeia o valor numérico para o texto
    df_tabela['Previsão'] = df_tabela['Tendência'].apply(lambda x: 'Subida (+1)' if x == 1 else 'Descida (-1)')

    # Função para aplicar cor com base na coluna 'Previsão'
    def cor_tendencia(val):
        """Aplica cor verde se for subida, vermelho se for descida."""
        if 'Subida' in val:
            color = 'green'
        elif 'Descida' in val:
            color = 'red'
        else:
            color = 'black'
        return f'color: {color}; font-weight: bold;'

    # Exibe a tabela estilizada
    st.dataframe(
        df_tabela[['Data', 'Previsão']].style.applymap(cor_tendencia, subset=['Previsão']),
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning("Aguardando o carregamento do modelo ou do DataFrame processado.")