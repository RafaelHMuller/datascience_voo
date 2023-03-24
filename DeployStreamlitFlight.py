#!/usr/bin/env python
# coding: utf-8

# ##### Deploy do Modelo de Previsão:

# In[1]:


import pandas as pd
import joblib
import streamlit as st


# Ler a base de dados final para garantir a sequência exata das colunas/features para o modelo de previsão

# In[2]:


df = pd.read_csv('Base de Dados Final.csv')


# In[3]:


# lista da sequência de colunas, excluindo-se 'price'
df = df.drop('price', axis=1)
sequencia_colunas = list(df.columns)
#print(sequencia_colunas)


# Importar as features da base de dados final

# In[4]:


# print(X.columns) no código do modelo


# Criar dicionários a partir das features e seus valores iniciais

# In[5]:


# dicionário das features numéricas
x_numericos = {
    'duration': 0,
    'days_left': 0,
}

# dicionário das features de lista de valores
x_listas = {
    'airline':['SpiceJet','AirAsia','Vistara','GO_FIRST','Indigo','Air_India'],
    'source_city':['Delhi','Mumbai','Bangalore','Kolkata','Hyderabad','Chennai'],
    'destination_city':['Mumbai','Bangalore','Kolkata','Hyderabad','Chennai','Delhi'],
    'class':['Business','Economy'],
}


# Criar botões para cada feature (streamlit)

# In[6]:


# botões das features numéricas

for chave in x_numericos:
    if chave == 'duration':
        valor = st.number_input(f'{chave}', step=0.01, value=float(0), format='%.2f')
    else:
        valor = st.number_input(f'{chave}', step=1, value=int(0))
    x_numericos[chave] = valor
    
# botões das features de lista de valores

x_listas2 = {}

for chave in x_listas:     
    x_listas2[chave] = 0

for chave in x_listas:
    valor = st.selectbox(f'{chave}', x_listas[chave])
    i = int(x_listas[chave].index(valor))
    x_listas2[chave] = i

# botão final (previsão)  

botao = st.button('Prever Valor do Vôo')


if botao:
    x_listas2.update(x_numericos) # juntar os dicionários
    dataframe_X = pd.DataFrame(x_listas2, index=[0]) # criar um dataframe dos valores X com a mesma ordem de colunas que o dataframe final do projeto de ciência de dados
    dataframe_X = dataframe_X[sequencia_colunas] # reordenar as colunas do dataframe
    modelo = joblib.load('Modelo de Previsão.joblib') # importar o modelo
    preco = modelo.predict(dataframe_X) # usar a previsão do modelo
    st.write(f'R$ {preco[0]:,.2f}') # escrever o valor previsto no streamlit 


# Concluído o código:
# - retirar todos os displays e prints
# - exportar o arquivo para a extensão .py
# - abrir o prompt do anaconda:
#     - encontrar o local onde está o arquivo (comando cd)
#     - executar: streamlit run arquivo.py

# In[ ]:




