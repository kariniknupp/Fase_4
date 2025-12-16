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
st.title('ANÁLISE DE PERÍODOS E PREVISÃO DE TENDÊNCIA DO IBOVESPA :moneybag:')

# ----------------------------------------------------
# 4. PAINEL DE MÉTRICAS DO MODELO ML (st.sidebar)
# ----------------------------------------------------

# Resultados obtidos pelo XGBOOST na atividade da Fase 2
METRICAS = {
    "Acurácia Direcional": 77.27,
    "R² Score": -0.061,
    "MAE (Erro Absoluto)": 0.908,
    "WMAPE": 1.035,
}

st.sidebar.title("📈 Métricas de Validação (XGBoost)")
st.sidebar.markdown("Performance do Modelo Treinado:")

st.sidebar.write("Métricas:")

st.sidebar.metric("Acurácia Direcional", f"{METRICAS['Acurácia Direcional']:.2f}%", delta=None) 

st.sidebar.metric("R² Score", f"{METRICAS['R² Score']:.3f}", delta=None) 
st.sidebar.metric("MAE", f"{METRICAS['MAE (Erro Absoluto)']:.3f}", delta=None)
st.sidebar.metric("WMAPE", f"{METRICAS['WMAPE']:.3f}", delta=None)

st.sidebar.markdown("---")
st.sidebar.info(f"O modelo está usando {len(FEATURES)} features preditoras.")

st.sidebar.info(f"O modelo prevê a tendência (+1 subida, -1 descida) com base em {len(FEATURES)} features.")
st.sidebar.info(f"Features utilizadas: Valor de abertura do dia, Valores do dia anterior: tendência, valor de fechamento, \
                máxima, mínima, aplitude, delta (fechamento - abertura), variação (delta/abertura), \
                volatilidade e média móvel - semanal, mensal, trimestral, semestral e anual")

st.write('Período Histórico Analisado:', df['ds'].min(), 'a', df['ds'].max())


#Dados disponibilizados para análise

coluna1, coluna2, coluna3, coluna4 = st.columns(4)
with coluna1:
    st.metric('Quantidade de Dias Analisados', df['ds'].count())
with coluna2:
    st.metric('Máxima do Índice', df['max'].max().round(2))
with coluna3:
    st.metric('Mínima do Índice', df['min'].min().round(2))
with coluna4:
    st.metric('Média do Índice', df['fechamento'].mean().round(2))

#  ------------------------------------------------------------
# 5. NOVA SEÇÃO: ANÁLISE EXPLORATÓRIA CUSTOMIZÁVEL (PREÇO BRUTO)
# --------------------------------------------------------------
st.header("📊 Análise Exploratória Customizável: Preço, Média Móvel e Desvio Padrão")

# 5.1 Controles do Usuário
col_periodo, col_ma_window, col_checkbox = st.columns([1, 1, 1])

# Lógica de conversão para cálculos
df_analise = df.copy()
df_analise['ds'] = pd.to_datetime(df_analise['ds']) 
min_data_disponivel = df_analise['ds'].min()
max_data_disponivel = df_analise['ds'].max()


# --- INPUT 1: Seleção de Período ---
with col_periodo:
    periodo_selecionado = st.radio(
        "Selecione o Período Histórico",
        ['Último Ano', 'Últimos 2 Anos', 'Todo o Período', 'Customizar Intervalo'],
        horizontal=True
    )

# --- INPUT 2: Janela da Média Móvel ---
with col_ma_window:
    ma_window = st.slider(
        "Janela da Média Móvel (dias úteis)",
        min_value=10, max_value=252, value=50, step=10
    )

# --- INPUT 3: Exibir Desvio Padrão ---
with col_checkbox:
    st.markdown("<br>", unsafe_allow_html=True) 
    mostrar_std = st.checkbox("Exibir Desvio Padrão (Banda)", value=True)


# --- INPUT 4: Customização de Data (Aparece somente se selecionado) ---
start_date_custom = None
end_date_custom = None
data_valida = True

if periodo_selecionado == 'Customizar Intervalo':
    st.markdown("##### Selecione o Intervalo Desejado")
    col_start, col_end = st.columns(2)
    
    with col_start:
        start_date_custom = st.date_input(
            "Data de Início",
            # Padrão de 6 meses atrás
            value=max_data_disponivel - pd.DateOffset(months=6), 
            min_value=min_data_disponivel,
            max_value=max_data_disponivel
        )
        
    with col_end:
        end_date_custom = st.date_input(
            "Data Final",
            value=max_data_disponivel,
            min_value=min_data_disponivel,
            max_value=max_data_disponivel
        )


# 5.2 Lógica de Slicing e Cálculo
start_date = min_data_disponivel
end_date = max_data_disponivel

# Lógica de Filtro
if periodo_selecionado == 'Último Ano':
    start_date = max_data_disponivel - pd.DateOffset(years=1)
elif periodo_selecionado == 'Últimos 2 Anos':
    start_date = max_data_disponivel - pd.DateOffset(years=2)
elif periodo_selecionado == 'Customizar Intervalo':
    # Converte os objetos date_input (date) para datetime para o filtro
    start_date = pd.to_datetime(start_date_custom)
    end_date = pd.to_datetime(end_date_custom)
    
    # Validação de data
    if start_date > end_date:
        st.error("Erro: A Data de Início não pode ser posterior à Data Final. Ajuste o intervalo.")
        data_valida = False
        st.stop()

if data_valida:
    # Aplica o filtro de período ao DataFrame
    df_slice = df_analise[(df_analise['ds'] >= start_date) & (df_analise['ds'] <= end_date)].copy()

    # Cálculo da Média Móvel e Desvio Padrão
    df_slice['MA'] = df_slice['fechamento'].rolling(window=ma_window).mean()
    df_slice['STD'] = df_slice['fechamento'].rolling(window=ma_window).std()
    df_slice['Upper_Band'] = df_slice['MA'] + (df_slice['STD'] * 2) 
    df_slice['Lower_Band'] = df_slice['MA'] - (df_slice['STD'] * 2) 

# 5.3 Painel de Métricas do Período (MÁXIMA, MÍNIMA, MEDIANA)
    st.markdown("#### Estatísticas do Período Selecionado")
    
    # Calcula as métricas
    max_val = df_slice['fechamento'].max()
    min_val = df_slice['fechamento'].min()
    mediana_val = df_slice['fechamento'].median()
    media_val = df_slice['fechamento'].mean()

    # Layout de 4 colunas para as métricas
    col_max, col_min, col_median, col_mean = st.columns(4)

    with col_max:
        st.metric("Máxima", f"{max_val:,.2f}")
    
    with col_min:
        st.metric("Mínima", f"{min_val:,.2f}")
        
    with col_median:
        st.metric("Mediana", f"{mediana_val:,.2f}")
        
    with col_mean:
        st.metric("Média", f"{media_val:,.2f}")
        
    st.markdown("---") # Separador visual

# 5.4 Plotagem com Plotly Graph Objects
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
    fig_analise.add_trace(go.Scatter(
        x=df_slice['ds'], y=df_slice['Upper_Band'],
        mode='lines',
        name='Banda Superior',
        line=dict(width=0), 
        fillcolor='rgba(255, 165, 0, 0.15)', 
        fill='tonexty', 
        hoverinfo='skip' 
    ))
    fig_analise.add_trace(go.Scatter(
        x=df_slice['ds'], y=df_slice['Lower_Band'],
        mode='lines',
        name='Banda Inferior (Desvio Padrão)',
        line=dict(width=0), 
        fill='tonexty', 
        fillcolor='rgba(255, 165, 0, 0.15)' 
    ))


# Layout e Customização
fig_analise.update_layout(
    title=f'Análise de Fechamento do IBOVESPA - Período Selecionado',
    xaxis_title='Data',
    yaxis_title='Valor do Índice (R$)',
    hovermode='x unified',
    template='plotly_white'
)

st.plotly_chart(fig_analise, use_container_width=True)

#  ----------------------------------------------------
# 6. SEÇÃO DE PREVISÃO DO MODELO
#  ----------------------------------------------------

st.markdown("---")
st.header("🔮 Previsão de Tendência com Machine Learning")

# NOVO INPUT: RADIO BUTTON PARA SELEÇÃO DE DIAS
st.write('Escolha para quantos dias deseja a previsão de tendência:')
opcoes_dias = {
    'Próximo Dia (1)': 1,
    'Próximos 5 Dias': 5,
    'Próximos 10 Dias': 10,
    'Próximos 15 Dias': 15
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
    # 6.2 GERAÇÃO DA PREVISÃO: LÓGICA RECURSIVA PARA REGRESSOR
    # ------------------------------------------------------------------

    # --- 1. PREPARAÇÃO DO PONTO DE PARTIDA ---
    # Recupera o último valor real o índice para iniciar a recursão
    df_ultimo = df_processado.iloc[[-1]]
    P_last_real = df_ultimo['fechamento'].values[0] # Valor de fechamento real

    # O preço de referência para o primeiro cálculo de tendência (será o P_last_real)
    P_referencia = P_last_real 

    # Cópia do vetor de features do último dia (base para o loop)
    X_base = df_ultimo[FEATURES].values[0].copy()

    # Encontra o índice da feature de lag do fechamento no vetor FEATURES
    try:
        fechamento_lag_index = FEATURES.index('fechamento_lag_1')
    except ValueError:
        st.error("Erro: A feature 'fechamento_lag_1' não foi encontrada na lista de FEATURES. Verifique se o nome está correto no seu DataFrame processado.")
        st.stop()

    # Lista para armazenar as previsões futuras (data e y_pred de TENDÊNCIA)
    resultados_recursivos = []
    # Lista para armazenar o valor predito (opcional, para debug)
    precos_preditos = [] 

    # --- 2. LOOP RECURSIVO ---
    for i, data_futura in enumerate(datas_futuras):
        # a. Prepara o vetor X para o modelo e prevê o VALOR
        X_novo = pd.DataFrame([X_base], columns=FEATURES)
    
        # O modelo prevê o VALOR do fechamento para o dia N+1
        P_predito = modelo_ml.predict(X_novo)[0] 
    
        # b. CALCULA A TENDÊNCIA (+1 ou -1)
        # A tendência é baseada na variação do valor predito (P_predito) em relação ao valor de referência (P_referencia)
        T_predita = 1 if P_predito > P_referencia else -1
    
        # c. Armazena o resultado (a TENDÊNCIA é o que será plotado)
        resultados_recursivos.append({
            'ds': data_futura,
            'y_pred': T_predita 
        })
    
        # d. ATUALIZAÇÃO RECURSIVA para a próxima iteração
        # A nova referência de preço é o preço que acabamos de prever
        P_referencia = P_predito 
    
        # Atualiza a feature 'fechamento_lag_1' para o próximo dia com o P_predito
        X_base[fechamento_lag_index] = P_predito
    
    # Converte os resultados em DataFrame
    df_futuro = pd.DataFrame(resultados_recursivos)

    st.info("""
        ✅ **Lógica de Regressão Aplicada:** O modelo prevê o valor do índice. A tendência (+1/-1) é calculada comparando o valor predito com o valor do dia anterior (recursivamente).
    """)

 
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