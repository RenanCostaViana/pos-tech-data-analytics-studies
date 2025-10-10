import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import DropFeatures, OrdinalFeature, MinMaxWithFeatNames
from sklearn.pipeline import Pipeline
import joblib
from joblib import load

st.write('# Avaliação de Doença Cardíaca')

# Carregando dados
url = 'https://raw.githubusercontent.com/FIAP/Pos_Tech_DTAT/dd201a034223a16732c6f639b40600b26bd3129c/Desafio/Doen%C3%A7aVascular.xlsx'
dados = pd.read_excel(url, engine='openpyxl')

# --- Entradas do usuário ---------------------------------------------------
with st.form("formulario_paciente"):
    st.write("### Idade")
    input_idade = float(st.number_input('Idade do paciente (em anos)', 1))

    st.write("### Gênero")
    input_genero = st.radio('Gênero biológico do paciente', ['Homem', 'Mulher'])
    input_genero = {'Homem': 1, 'Mulher': 0}.get(input_genero)

    st.write("### Altura")
    input_altura = float(st.number_input('Altura (cm)', 1))

    st.write("### Peso")
    input_peso = float(st.number_input('Peso (kg)', 1))

    st.write("### Pressão Arterial Sistólica")
    input_pressao_sistolica = float(st.number_input('Pressão sistólica', 1))

    st.write("### Pressão Arterial Diastólica")
    input_pressao_diastolica = float(st.number_input('Pressão diastólica', 1))

    st.write("### Colesterol")
    input_colesterol = st.radio('Nível de colesterol', ['Normal', 'Acima do Normal', 'Muito Acima do Normal'])
    input_colesterol = {'Normal': 1, 'Acima do Normal': 2, 'Muito Acima do Normal': 3}.get(input_colesterol)

    st.write("### Glicose")
    input_glicose = st.radio('Nível de glicose', ['Normal', 'Acima do Normal', 'Muito Acima do Normal'])
    input_glicose = {'Normal': 1, 'Acima do Normal': 2, 'Muito Acima do Normal': 3}.get(input_glicose)

    st.write("### Fumante")
    input_fumante = st.radio('É fumante?', ['Sim', 'Não'])
    input_fumante = {'Sim': 1, 'Não': 0}.get(input_fumante)

    st.write("### Álcool")
    input_alcool = st.radio('Consome álcool?', ['Sim', 'Não'])
    input_alcool = {'Sim': 1, 'Não': 0}.get(input_alcool)

    st.write("### Atividade Física")
    input_ativo = st.radio('É fisicamente ativo?', ['Sim', 'Não'])
    input_ativo = {'Sim': 1, 'Não': 0}.get(input_ativo)

    enviar = st.form_submit_button("Enviar")

# Lista de todas as variáveis:
novo_cliente = [0, # index
                0, # id
                input_idade, # Idade
                input_genero, # Genero
                input_altura, # Altura
                input_peso, # Peso
                input_pressao_sistolica,  # PressaoArterialSistolica
                input_pressao_diastolica,  # PressaoArterialDiastolica
                input_colesterol, # Colesterol
                input_glicose, # Glicose
                input_fumante, # Fumante
                input_alcool, # UsaAlcool
                input_ativo, # AtivoFisicamente
                0
                ]

# Separando os dados em treino e teste
def data_split(df, test_size):
    SEED = 1561651
    treino_df, teste_df = train_test_split(df, test_size=test_size, random_state=SEED)
    return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

treino_df, teste_df = data_split(dados, 0.2)

#Criando novo cliente
cliente_predict_df = pd.DataFrame([novo_cliente],columns=teste_df.columns)

#Concatenando novo cliente ao dataframe dos dados de testecccccc
teste_novo_cliente  = pd.concat([teste_df,cliente_predict_df],ignore_index=True)

# Criação do pipeline
pipeline = Pipeline([
    ('feature_dropper', DropFeatures()),
    ('ordinal_feature', OrdinalFeature()),
    ('min_max_scaler', MinMaxWithFeatNames())
])

# Ajusta o pipeline com os dados de treino
pipeline.fit(treino_df)

# Transforma o novo cliente com o mesmo pipeline
X_cliente = pipeline.transform(cliente_predict_df)

# Cria DataFrame com as mesmas colunas do treino transformado
X_treino = pipeline.transform(treino_df)
X_cliente_df = pd.DataFrame(X_cliente, columns=X_treino.columns)

# Verifica valores ausentes antes da predição
if enviar:
    if X_cliente_df.isnull().any().any():
        st.warning("⚠️ Há valores ausentes nos dados do paciente após transformação. Verifique os dados ou o pipeline.")
    else:
        # Remove a coluna alvo
        X_cliente_pred = X_cliente_df.drop(['DoencaVascular'], axis=1)

        # Carrega o modelo e faz a predição
        model = joblib.load('modelo/logistico.joblib')
        final_pred = model.predict(X_cliente_pred)

        # Exibe o resultado
        if final_pred[-1] == 0:
            st.success('### O paciente não tem chance de ter doenças cardiovasculares')
        else:
            st.error('### O paciente provavelmente tem doenças cardiovasculares')


