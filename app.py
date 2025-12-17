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
    title=dict(
        text=f'Análise de Fechamento do IBOVESPA - Período de {start_date.strftime("%d/%m/%Y")} a {end_date.strftime("%d/%m/%Y")}',
        font=dict(color='black', size=18)
    ),
    xaxis_title='Data',
    yaxis_title='Valor do Índice (R$)',
    hovermode='x unified',
    template='plotly_white',  # Define o tema claro
    paper_bgcolor='white',    # Fundo externo branco
    plot_bgcolor='white',
    font_color="black", # Define a cor da fonte global como preto
    xaxis=dict(
        showgrid=True, 
        gridcolor='LightGray',
        title_font=dict(color='black'),
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        showgrid=True, 
        gridcolor='LightGray',
        title_font=dict(color='black'),
        tickfont=dict(color='black')
    ),
    legend=dict(font=dict(color='black'))
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
    'Próximos 15 Dias': 15,
    'Próximos 30 Dias': 30
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
# 6.2 GERAÇÃO DA PREVISÃO: LÓGICA RECURSIVA COMPLETA
# ------------------------------------------------------------------

# --- 0. MAPEAMENTO DE JANELAS (AJUSTE SE NECESSÁRIO) ---
# Assumindo estas janelas para os lags: S (Short), M (Medium), A (Annual)
    WINDOW_MAP = {
        'S': 5,     # Ex: volatilidadeS_lag_1 -> Janela de 5 dias
        'M': 20,    # Ex: maM_lag_1 -> Janela de 20 dias
        'A': 252    # Ex: maA_lag_1 -> Janela de 252 dias
    }
# Acha a maior janela necessária para inicializar o histórico
    MAX_WINDOW = max(WINDOW_MAP.values())

# --- 1. PREPARAÇÃO DO PONTO DE PARTIDA E SÉRIE HISTÓRICA ---

# Recupera o último valor real de preço e o vetor completo de features
    df_ultimo = df_processado.iloc[[-1]]
    P_referencia = df_ultimo['fechamento'].values[0] 
    X_base = df_ultimo[FEATURES].values[0].copy()

# 1.1 Inicializa a série de preços: Pega o tail da coluna 'fechamento' para os cálculos de rolling
# Usamos MAX_WINDOW + 1 para ter certeza que temos o suficiente para o primeiro cálculo
# Histórico de preços para cálculo de médias e volatilidade
    price_series_history = df_processado['fechamento'].tail(MAX_WINDOW).tolist()

# Localização de índices para atualização recursiva
    try:
        idx_p_lag = FEATURES.index('fechamento_lag_1')
        ma_indices = {n: FEATURES.index(n) for n in FEATURES if n.startswith('ma') and n.endswith('_lag_1')}
        vol_indices = {n: FEATURES.index(n) for n in FEATURES if n.startswith('volatilidade') and n.endswith('_lag_1')}
    except ValueError as e:
        st.error(f"Erro ao localizar features: {e}")
        st.stop()

    resultados_recursivos = []


# --- 2. LOOP RECURSIVO ---
    for i, data_futura in enumerate(datas_futuras):
        X_novo = pd.DataFrame([X_base], columns=FEATURES)
    
        # Modelo prevê o VALOR EXATO (Regressão)
        P_predito = float(modelo_ml.predict(X_novo)[0])
    
        # Calcula tendência baseada no valor anterior
        T_predita = 1 if P_predito > P_referencia else -1
    
        # Armazena resultados
        resultados_recursivos.append({
            'ds': data_futura,
            'valor_previsto': P_predito,
            'tendencia': T_predita
        })
    
        # ATUALIZAÇÃO RECURSIVA PARA O PRÓXIMO PASSO (N+1)
        price_series_history.append(P_predito)
        price_series_history = price_series_history[1:]
        ps = pd.Series(price_series_history)
    
        # Atualiza Lags de Médias Móveis
        for name, idx in ma_indices.items():
            w = WINDOW_MAP.get(name[2], 20)
            X_base[idx] = ps.tail(w).mean()
        
        # Atualiza Lags de Volatilidade
        for name, idx in vol_indices.items():
            w = WINDOW_MAP.get(name[14], 20)
            X_base[idx] = ps.tail(w).std()
    
        # Atualiza Lag de Preço
        X_base[idx_p_lag] = P_predito
        P_referencia = P_predito 

    df_futuro = pd.DataFrame(resultados_recursivos)
 
# ------------------------------------------------------------------
# 6.3 VISUALIZAÇÃO DO GRÁFICO (CONEXÃO PERFEITA HISTÓRICO/PREVISÃO)
# ------------------------------------------------------------------
    st.header(f"📈 Projeção de Tendência ({input_qtd_dias} Dias)")

    df_historico_plot = df_processado[['ds', 'y']].copy().rename(columns={'y': 'Tendência'})
    df_historico_plot['Tipo'] = 'Histórico (Real)'

    df_futuro_plot = df_futuro[['ds', 'tendencia']].copy().rename(columns={'tendencia': 'Tendência'})
    df_futuro_plot['Tipo'] = 'Previsão'

    # Ponto de ponte para não haver buraco no gráfico
    ponto_ponte = df_historico_plot.iloc[[-1]].copy()
    ponto_ponte['Tipo'] = 'Previsão'

    df_previsao_full = pd.concat([ponto_ponte, df_futuro_plot])
    df_comp = pd.concat([df_historico_plot.tail(30), df_previsao_full])
    df_comp.drop_duplicates(subset=['ds'], keep='last', inplace=True)

    fig_prev = px.line(
        df_comp, x='ds', y='Tendência', color='Tipo',
        title='Movimentação Prevista: Subida (+1) vs Descida (-1)',
        labels={'Tendência': 'Tendência', 'ds': 'Data'},
        color_discrete_map={'Histórico (Real)': '#1f77b4', 'Previsão': '#d62728'},
        template='plotly_white' # Tema base branco
    )

    fig_prev.update_layout(
        title=dict(font=dict(color='black', size=18)),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font_color="black",
        yaxis=dict(
            tickvals=[-1, 1], 
            ticktext=['Descida (-1)', 'Subida (+1)'], 
            range=[-1.5, 1.5],
            showgrid=True, 
            gridcolor='LightGray',
            title_font=dict(color='black'),
            tickfont=dict(color='black')
        ),
        xaxis=dict(
            showgrid=True, 
            gridcolor='LightGray',
            title_font=dict(color='black'),
            tickfont=dict(color='black')
        ),
        legend=dict(font=dict(color='black')),
        hovermode="x unified"
    )
    st.plotly_chart(fig_prev, use_container_width=True)

# ------------------------------------------------------------------
# 6.4 TABELA DE PREVISÃO DETALHADA (COM VALOR DO ÍNDICE)
# ------------------------------------------------------------------
    st.subheader("📋 Tabela de Previsões Detalhada")

    st.info("""✅
        **Nota Técnica:** Este modelo é um **XGBRegressor**. Ele foi treinado para estimar o valor exato do índice IBOVESPA. 
        A tendência (+1 ou -1) exibida abaixo é derivada da comparação do valor previsto com o fechamento do dia anterior.
    """)

    df_tab = df_futuro.copy()
    df_tab['Data'] = pd.to_datetime(df_tab['ds']).dt.strftime('%d/%m/%Y')   
    df_tab['Movimento'] = df_tab['tendencia'].apply(lambda x: 'Subida (+1)' if x == 1 else 'Descida (-1)')

    def style_mov(val):
        color = 'green' if '+1' in val else 'red'
        return f'background-color: {color}; color: white; font-weight: bold;'

    st.dataframe(
        df_tab[['Data', 'Movimento']].style.applymap(
            style_mov, subset=['Movimento']
        ),
        use_container_width=True, hide_index=True
    )

else:
    st.warning("Aguardando o carregamento do modelo ou do DataFrame processado.")