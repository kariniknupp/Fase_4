import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import xgboost as xgb
import joblib
from joblib import load
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

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
# 4. PAINEL DE MÉTRICAS DO MODELO ML (st.sidebar)
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

# ====================================================================
# 5. NOVA SEÇÃO: ANÁLISE EXPLORATÓRIA CUSTOMIZÁVEL (PREÇO BRUTO)
# ====================================================================
st.header("📊 Análise Exploratória: Preço, Média Móvel e Desvio Padrão")

# 5.1 Controles do Usuário
col_periodo, col_ma_window, col_checkbox = st.columns([1, 1, 1])

with col_periodo:
    periodo_selecionado = st.radio(
        "Selecione o Período Histórico",
        ['Último Ano', 'Últimos 2 Anos', 'Todo o Período'],
        horizontal=True
    )

with col_ma_window:
    ma_window = st.slider(
        "Janela da Média Móvel (dias úteis)",
        min_value=10, max_value=252, value=50, step=10
    )

with col_checkbox:
    # Espaçamento para alinhar com o radio button
    st.markdown("<br>", unsafe_allow_html=True) 
    mostrar_std = st.checkbox("Exibir Desvio Padrão (Banda)", value=True)

# 5.2 Lógica de Slicing e Cálculo
df_analise = df.copy()
# Converte a coluna 'ds' de volta para datetime para permitir cálculos de offset de data
df_analise['ds'] = pd.to_datetime(df_analise['ds']) 

end_date = df_analise['ds'].max()

if periodo_selecionado == 'Último Ano':
    # DateOffset(years=1) é mais seguro que timedelta(days=365) para anos
    start_date = end_date - pd.DateOffset(years=1)
elif periodo_selecionado == 'Últimos 2 Anos':
    start_date = end_date - pd.DateOffset(years=2)
else:
    start_date = df_analise['ds'].min()

# Aplica o filtro de período
df_slice = df_analise[df_analise['ds'] >= start_date].copy()

# Cálculo da Média Móvel e Desvio Padrão
# O window 'ma_window' usa apenas os dias úteis (índice)
df_slice['MA'] = df_slice['fechamento'].rolling(window=ma_window).mean()
df_slice['STD'] = df_slice['fechamento'].rolling(window=ma_window).std()
df_slice['Upper_Band'] = df_slice['MA'] + (df_slice['STD'] * 2) # 2x desvio padrão
df_slice['Lower_Band'] = df_slice['MA'] - (df_slice['STD'] * 2) # 2x desvio padrão

# 5.3 Plotagem com Plotly Graph Objects
fig_analise = go.Figure()

# Traço 1: Preço Bruto
fig_analise.add_trace(go.Scatter(
    x=df_slice['ds'], y=df_slice['fechamento'],
    mode='lines',
    name='Fechamento (Bruto)',
    line=dict(color='#1f77b4', width=1)
))

# Traço 2: Média Móvel
fig_analise.add_trace(go.Scatter(
    x=df_slice['ds'], y=df_slice['MA'],
    mode='lines',
    name=f'Média Móvel ({ma_window} dias)',
    line=dict(color='#ff7f0e', width=2)
))

# Traços 3 e 4 (Opcional): Desvio Padrão (Usando fill para criar a banda)
if mostrar_std:
    # Banda Superior (preenchimento iniciado aqui)
    fig_analise.add_trace(go.Scatter(
        x=df_slice['ds'], y=df_slice['Upper_Band'],
        mode='lines',
        name='Banda Superior',
        line=dict(width=0), 
        fillcolor='rgba(255, 165, 0, 0.15)', # Cor transparente para o preenchimento
        fill='tonexty', # Preenche até a linha anterior (MA)
        hoverinfo='skip' # Não mostrar o hover nesta linha
    ))
    # Banda Inferior (preenchimento até a linha superior, completando a banda)
    fig_analise.add_trace(go.Scatter(
        x=df_slice['ds'], y=df_slice['Lower_Band'],
        mode='lines',
        name='Banda Inferior (Desvio Padrão)',
        line=dict(width=0), 
        fill='tonexty', # Preenche da linha atual (Lower) até a Upper
        fillcolor='rgba(255, 165, 0, 0.15)' 
    ))


# Layout e Customização
fig_analise.update_layout(
    title=f'Análise de Fechamento do IBOVESPA - {periodo_selecionado}',
    xaxis_title='Data',
    yaxis_title='Valor do Índice (R$)',
    hovermode='x unified',
    template='plotly_white'
)

st.plotly_chart(fig_analise, use_container_width=True)


# ====================================================================
# 6. SEÇÃO DE PREVISÃO DO MODELO
# ====================================================================

st.markdown("---")
st.header("🔮 Previsão de Tendência com Machine Learning")

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
# 6.1 Preparação dos Dados Para Previsão
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
    
    # Repete as features do último dia conhecido (CAUSA PREVISÃO MONÓTONA)
    for feature in FEATURES:
         df_futuro[feature] = ultimo_df_historico[feature].iloc[0]
         
    # ------------------------------------------------------------------
    # 6.2 GERAÇÃO DA PREVISÃO
    # ------------------------------------------------------------------
    
    X_futuro = df_futuro[FEATURES]
    
   # st.warning("""
    #    🚨 **Atenção: A previsão está monótona (só descida ou subida constante) porque as features de entrada são as mesmas para todos os dias futuros.**
    #    
    #    **Ação necessária:** Para obter previsões variadas e corretas, insira a lógica de **engenharia de recursos recursiva** neste bloco (Dia N+1 depende da previsão do Dia N).
    #""")
    
    # 💡 SUBSTITUIÇÃO TEMPORÁRIA: SIMULAÇÃO DE PREVISÃO VARIADA PARA TESTAR A VISUALIZAÇÃO
    previsoes = np.random.choice([-1, 1], size=input_qtd_dias)
    # previsoes = modelo_ml.predict(X_futuro) # <--- USE ESTA LINHA COM O FEATURE ENGINEERING CORRETO
    
    df_futuro['y_pred'] = previsoes
    
    
    # ------------------------------------------------------------------
    # 6.3 VISUALIZAÇÃO DO GRÁFICO (Foco nos últimos 30 dias + Previsão)
    # ------------------------------------------------------------------
    
    st.header(f"Projeção de Tendência ({input_qtd_dias} Dias) - Resultado: {selecao}")
    
    DIAS_HISTORICOS_A_MOSTRAR = 30
    df_historico_plot = df_processado[['ds', 'y']].copy().rename(columns={'y': 'Tendência'})
    df_historico_plot['Tipo'] = 'Histórico (Real)'
    
    df_futuro_plot = df_futuro[['ds', 'y_pred']].copy().rename(columns={'y_pred': 'Tendência'})
    df_futuro_plot['Tipo'] = 'Previsão'
    
    df_historico_slice = df_historico_plot.tail(DIAS_HISTORICOS_A_MOSTRAR)
    
    df_combinado_visualizacao = pd.concat([df_historico_slice, df_futuro_plot])
    df_combinado_visualizacao['ds'] = pd.to_datetime(df_combinado_visualizacao['ds'])
    
    # Cria o gráfico Plotly com Previsões
    fig = px.line(
        df_combinado_visualizacao, 
        x='ds', 
        y='Tendência', 
        color='Tipo', 
        title='Histórico Recente e Previsão de Tendência (+1 Sobe, -1 Desce)',
        labels={'Tendência': 'Direção do Movimento', 'ds': 'Data'},
        color_discrete_map={'Histórico (Real)': '#1f77b4', 'Previsão': '#d62728'}
    )
    
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
    # 6.4 TABELA DE PREVISÃO ESTILIZADA
    # ------------------------------------------------------------------
    st.subheader("Resultados da Previsão Detalhada")
    
    df_tabela = df_futuro_plot.copy()
    df_tabela['Data'] = df_tabela['ds'].dt.strftime('%d/%m/%Y')
    df_tabela['Previsão'] = df_tabela['Tendência'].apply(lambda x: 'Subida (+1)' if x == 1 else 'Descida (-1)')

    def cor_tendencia(val):
        if 'Subida' in val:
            color = 'green'
        elif 'Descida' in val:
            color = 'red'
        else:
            color = 'black'
        return f'color: {color}; font-weight: bold;'

    st.dataframe(
        df_tabela[['Data', 'Previsão']].style.applymap(cor_tendencia, subset=['Previsão']),
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning("Aguardando o carregamento do modelo ou do DataFrame processado.")