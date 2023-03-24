#!/usr/bin/env python
# coding: utf-8

# # Projeto Semana Ciência de Dados

# #### Passo a Passo de um Projeto de Ciência de Dados:
# 
# - 1: Entendimento do Desafio
# - 2: Entendimento da Área/Empresa
# - 3: Extração/Obtenção de Dados
# - 4: Ajuste de Dados (Tratamento/Limpeza)
# - 5: Análise Exploratória (Excluir Outliers)
# - 6: Encoding + Modelagem + Algoritmos (Inteligência Artificial)
# - 7: Interpretação de Resultados
# - 8: Deploy (Visualização Final)

# ### 1-2: Entendimento do Desafio e da Área/Empresa
#     The objective of the study is to analyse the flight booking dataset obtained from “Ease My Trip” website and to conduct various statistical hypothesis tests in order to get meaningful information from it. 
# 
#     Kaggle Dataset contains information about flight booking options from the website Easemytrip for flight travel between India's top 6 metro cities.
# 
#     The aim of our study is to answer the below research questions:
#        - Does price vary with Airlines?
#        - How is the price affected when tickets are bought in just 1 or 2 days before departure?
#        - Does ticket price change based on the departure time and arrival time?
#        - How the price changes with change in Source and Destination?
#        - How does the ticket price vary between Economy and Business class?
#        - Does the number of stops influence the final price?

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px
import time

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# ### 3: Extração/Obtenção de Dados

# In[8]:


df = pd.read_csv('Dataset.csv')
display(df)


# ### 4: Ajuste de Dados

# In[9]:


# excluir coluna unnamed
df = df.drop('Unnamed: 0', axis=1)


# In[10]:


# verificar NaN e dtypes do df
df.info()
df.describe()


# ### 5: Análise Exploratória

# Coluna flight
# 
# Análise de dados: quais os vôos mais populares?

# In[11]:


# coluna de códigos dos vôos
df_flight = df['flight'].value_counts().to_frame()
display(df_flight)


# In[12]:


# provavelmente, esta coluna não vai ajudar em nada no modelo
df = df.drop('flight', axis=1)


# Coluna price
# 
# Análise de Dados: Qual a distribuição dos valores de vôos na base de dados?

# In[13]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.title("Distribuição dos valores dos vôos")
sns.boxplot(data=df, x="price")
plt.subplot(1,2,2)
plt.title("Distribuição dos valores dos vôos")
sns.histplot(data=df, x='price', kde=True)


#     R: Há alguns vôos cujos valores extrapolam a média dos valores da base de dados; estes valores outliers estão acima dos 100.000.

# Análise de Dados: Does price vary with Airlines?

# In[14]:


# comparação dos preços cobrados por cada companhia
df_airlines_mean = df[['airline', 'price']].groupby('airline').mean().sort_values('price', ascending=False)
df_airlines_min = df[['airline', 'price']].groupby('airline').min().sort_values('price', ascending=False)
df_airlines_max = df[['airline', 'price']].groupby('airline').max().sort_values('price', ascending=False)

df_airlines = df_airlines_mean.copy()
df_airlines = pd.concat([df_airlines, df_airlines_min, df_airlines_max], keys=['mean', 'min', 'max'])

display(df_airlines)


# In[15]:


# gráfico boxplot dos preços de cada companhias aéreas
plt.figure(figsize=(15, 5))
sns.boxplot(x=df['price'], y=df['airline'])


#     R: As companhias que vendem os tickets mais caros são Vistara E Air_India.

# Análise de Dados: How does the ticket price vary between Economy and Business class?

# In[16]:


# comparação entre as classes executiva e econômica
df_classes_mean = df[['class', 'price']].groupby('class').mean()
df_classes_min = df[['class', 'price']].groupby('class').min()
df_classes_max = df[['class', 'price']].groupby('class').max()

df_classes = df_classes_mean.copy()
df_classes = pd.concat([df_classes, df_classes_min, df_classes_max], keys=['mean', 'min', 'max'])

display(df_classes)

# porcentagem de vôos de cada classe
df_porcentagem_classe = df['class'].value_counts(normalize=True).to_frame()
porcentagem_business = df_porcentagem_classe.iloc[0,0]
porcentagem_economy = df_porcentagem_classe.iloc[1,0]

print(f'{porcentagem_business:.2%} de todos os valores pertencem à classe Executiva.')
print(f'{porcentagem_economy:.2%} de todos os valores pertencem à classe Econômica.')
diferenca = float(df_classes.iloc[0,0]) / float(df_classes.iloc[1,0])
print(f'O valor médio do ticket da classe Executiva é {diferenca:.2f} vezes mais caro que o da classe Econômica.')


# In[17]:


# gráfico boxplot dos preços de cada classe
plt.figure(figsize=(15, 5))
sns.boxplot(x=df['price'], y=df['class'])


#     R: Os tickets mais baratos são quase que exclusivamente da classe econômica, já os mais caros, da classe executiva.

# Será que as companhias Vistara e Air_India, que são as mais caras, vendem muito mais tickets de classe executiva do que econômica?

# In[18]:


df_companhia_classe_mean = df[['airline', 'class', 'price']].groupby(['airline', 'class']).mean().sort_values('price', ascending=False)
df_companhia_classe_min = df[['airline', 'class', 'price']].groupby(['airline', 'class']).min().sort_values('price', ascending=False)
df_companhia_classe_max = df[['airline', 'class', 'price']].groupby(['airline', 'class']).max().sort_values('price', ascending=False)

df_companhia_classe = df_companhia_classe_mean.copy()
df_companhia_classe = pd.concat([df_companhia_classe, df_companhia_classe_min, df_companhia_classe_max], keys=['mean', 'min', 'max'])

display(df_companhia_classe)


# In[19]:


# gráfico boxplot dos preços de cada companhias aéreas e suas respectivas classes
plt.figure(figsize=(15, 5))
sns.boxplot(x=df['price'], y=df['airline'], hue=df['class'])


#     R: As companhias mais caras, Air_India e Vistara, são as únicas que vendem tickets de classe executiva. Porém, mesmo entre todos os ticketes vendidos da classe econômica, essas duas companhias seguem sendo as mais caras.

# Análise de Dados: How is the price affected when tickets are bought in just 1 or 2 days before departure?

# In[20]:


df_daysleft = df[['days_left', 'price']].groupby('days_left').mean().sort_values('price', ascending=False)
df_daysleft = df_daysleft.reset_index()

display(df_daysleft)


# In[21]:


# gráfico scatterplot dos preços conforme a data em que foram comprados
plt.figure(figsize=(10, 5))
sns.scatterplot(y=df_daysleft['price'], x=df_daysleft['days_left'])


#     R: Os preços são maiores conforme mais próximo da data do vôo; a partir de 20 dias antes do vôo, os preços se estabilizam.

# Será que os valores dos tickets, de acordo com a data em que foram comprados, se alteram em relação à classe?

# In[22]:


df_daysleft = df[['days_left', 'class', 'price']].groupby(['days_left', 'class']).mean().sort_values('price', ascending=False)
df_daysleft = df_daysleft.reset_index()

display(df_daysleft)


# In[23]:


# gráfico scatterplot dos preços conforme a data em que foram comprados e a classe
plt.figure(figsize=(10, 5))
sns.scatterplot(y=df_daysleft['price'], x=df_daysleft['days_left'], hue=df_daysleft['class'])


#     R: Os preços dos tickets aumentam quanto mais próxima a data do vôo, porém este aumento é mais expressivo nos tickets da classe executiva; ainda, na classe executiva, os preços aumentam somente a uns 10 dias do vôo, já na classe econômica, este aumento se dá antes, uns 17 dias antes do vôo.

# Análise de Dados: Does ticket price change based on the departure time and arrival time?

# In[24]:


# analisando os tickets quanto ao horário de SAÍDA e CHEGADA
df_departure = df[['departure_time', 'class', 'price']].groupby(['departure_time', 'class']).mean()
df_departure = df_departure.reset_index()
df_departure2 = df[['departure_time', 'class', 'price']].groupby(['departure_time', 'class']).count()
df_departure2 = df_departure2.reset_index()
df_departure2 = df_departure2.rename(columns={'price': 'count'})
df_departure['count'] = df_departure2['count']

display(df_departure)

df_arrival = df[['arrival_time', 'class', 'price']].groupby(['arrival_time', 'class']).mean()
df_arrival = df_arrival.reset_index()
df_arrival2 = df[['arrival_time', 'class', 'price']].groupby(['arrival_time', 'class']).count()
df_arrival2 = df_arrival2.reset_index()
df_arrival2 = df_arrival2.rename(columns={'price': 'count'})
df_arrival['count'] = df_arrival2['count']

display(df_arrival)


# In[25]:


# gráfico barplot dos valores dos tickets
plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
sns.barplot(y=df['price'], x=df['departure_time'])
plt.subplot(1,2,2)
sns.barplot(y=df['price'], x=df['arrival_time'])


# In[26]:


# gráfico boxplot dos valores dos tickets por horário de chegada e de saída e por classe
df_business = df.loc[df['class']=='Business', :]
df_economy = df.loc[df['class']=='Economy', :]

plt.figure(figsize=(15, 10))
plt.subplot(2,2,1)
plt.title('Saída do vôo / Classe Executiva')
sns.boxplot(y=df_business['price'], x=df_business['departure_time'])
plt.subplot(2,2,2)
plt.title('Saída do vôo / Classe Econômica')
sns.boxplot(y=df_economy['price'], x=df_economy['departure_time'])
plt.subplot(2,2,3)
plt.title('Chegada do vôo / Classe Executiva')
sns.boxplot(y=df_business['price'], x=df_business['arrival_time'])
plt.subplot(2,2,4)
plt.title('Chegada do vôo / Classe Econômica')
sns.boxplot(y=df_economy['price'], x=df_economy['arrival_time'])


#     R: A tendência é que o ticket seja mais barato, em ambas as classes, se o horário de saída do vôo é tarde da noite. Ainda, a tendência é que o ticket seja mais barato, em ambas as classes, se o horário de chegada do vôo é no início da manhã ou de tarde.

# Análise de Dados: How the price changes with change in Source and Destination?

# In[27]:


# verificar os valores dos tickets conforme o local de origem e destino, sem classe
df_origem = df[['source_city', 'price']].groupby('source_city').mean()
print('Origem, ambas as classes')
display(df_origem)

df_destino = df[['destination_city', 'price']].groupby('destination_city').mean()
print('Destino, ambas as classes')
display(df_destino)

# verificar os valores dos tickets conforme o local de origem e destino e a classe
df_business = df.loc[df['class']=='Business', :]
df_economy = df.loc[df['class']=='Economy', :]

df_origem_business = df_business[['source_city', 'price']].groupby('source_city').mean()
df_origem_business = df_origem_business.reset_index()
print('Origem, Executiva')
display(df_origem_business)

df_destino_business = df_business[['destination_city', 'price']].groupby('destination_city').mean()
df_destino_business = df_destino_business.reset_index()
print('Destino, Executiva')
display(df_destino_business)

df_origem_economy = df_economy[['source_city', 'price']].groupby('source_city').mean()
df_origem_economy = df_origem_economy.reset_index()
print('Origem, Econômica')
display(df_origem_economy)

df_destino_economy = df_economy[['destination_city', 'price']].groupby('destination_city').mean()
df_origem_economy = df_origem_economy.reset_index()
print('Destino, Econômica')
display(df_destino_economy)


# In[28]:


# grafico boxplot dos locais de SAÍDA e de CHEGADA
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=df, y='price', x='source_city')
plt.subplot(1,2,2)
sns.boxplot(data=df, y='price', x='destination_city')


# In[29]:


# graficos boxplot dos locais de SAÍDA e CHEGADA de acordo com ambas as classes
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.title('Origem, classe Executiva')
sns.lineplot(data=df_origem_business, y='price', x='source_city')
plt.subplot(2,2,2)
plt.title('Origem, classe Econômica')
sns.lineplot(data=df_origem_economy, y='price', x='source_city')
plt.subplot(2,2,3)
plt.title('Destino, classe Executiva')
sns.lineplot(data=df_destino_business, y='price', x='destination_city')
plt.subplot(2,2,4)
plt.title('Destino, classe Econômica')
sns.lineplot(data=df_destino_economy, y='price', x='destination_city')


#     R: Os tickets mais baratos, em ambas as classes, são os que tem como origem e destino as cidades de Delhi e Hyderabad; por outro lado, os tickets mais caros são os que tem a cidade de Kolkata como origem e destino.

# Coluna stops
# 
# Análise de Dados: Does the number of stops influence the final price?

# In[30]:


# gráfico barplot da distribuição de paradas nos vôos na base de dados
plt.figure(figsize=(10,5))
sns.barplot(data=df, x='stops', y='price')


# In[31]:


df_stops = df[['stops', 'price']].groupby('stops').mean()
df_stops = df_stops.reset_index()
df_stops = df_stops.rename(columns={'price':'mean'})
df_stops2 = df[['stops', 'price']].groupby('stops').count()
df_stops2 = df_stops2.reset_index()
df_stops2 = df_stops2.rename(columns={'price':'count'})
df_stops['count'] = df_stops2['count']

display(df_stops)


# In[32]:


# gráficos boxplot do valor do ticket de acordo com o número de paradas e as classes
plt.figure(figsize=(10,5))
sns.barplot(data=df, x='stops', y='price', hue='class')


# ### 6: Encoding + Modelagem + Algoritmos

# ###### Encoding
#     - Ajustar todos os dados/features para facilitar o trabalho do modelo; para isso, deve-se codificar os dados de texto em números;
#     - Features de Valores True ou False - substitui-se True por 1 e False por 0
#     - Features de Categoria/Textos - utiliza-se o método de encoding de variáveis dummies (pegam-se todas as categorias da coluna, transforma-se cada categoria em coluna e atribuem-se os valores 1 ou 0)

# In[33]:


print(df.columns)
print(df.dtypes)


# In[34]:


# criar um df copia, no qual será feito o encoding das features
df_copia = df.copy()


# In[35]:


# 'class' - coluna de texto com apenas 2 variáveis
print(df_copia['class'].unique())


# In[36]:


df_copia.loc[df_copia['class']=='Business', 'class'] = 0
df_copia.loc[df_copia['class']=='Economy', 'class'] = 1


# In[37]:


# 'stops' - coluna de texto com muitas categorias/variáveis que podem números definidos
print(df_copia['stops'].unique())


# In[38]:


df_copia.loc[df['stops']=='zero', 'stops'] = 0
df_copia.loc[df['stops']=='one', 'stops'] = 1
df_copia.loc[df['stops']=='two_or_more', 'stops'] = 2


# In[39]:


# 'airline' - coluna de texto com muitas categorias/variáveis que podem números definidos
lista = df_copia['airline'].unique()
print(lista)


# In[40]:


contador = 0
for airline in lista:
    df_copia.loc[df['airline']==airline, 'airline'] = contador
    contador += 1


# In[41]:


# 'source_city' - coluna de texto com muitas categorias/variáveis que podem números definidos
lista = df_copia['source_city'].unique()
print(lista)


# In[42]:


contador = 0
for cidade in lista:
    df_copia.loc[df['source_city']==cidade, 'source_city'] = contador
    contador += 1


# In[43]:


# 'departure_time' - coluna de texto com muitas categorias/variáveis que podem números definidos
lista = df_copia['departure_time'].unique()
print(lista)


# In[44]:


contador = 0
for horario in lista:
    df_copia.loc[df_copia['departure_time']==horario, 'departure_time'] = contador
    contador += 1


# In[45]:


# 'arrival_time' - coluna de texto com muitas categorias/variáveis que podem números definidos
lista = df_copia['arrival_time'].unique()
print(lista)


# In[46]:


contador = 0
for horario in lista:
    df_copia.loc[df_copia['arrival_time']==horario, 'arrival_time'] = contador
    contador += 1


# In[47]:


# 'destination_city' - coluna de texto com muitas categorias/variáveis que podem números definidos
lista = df_copia['destination_city'].unique()
print(lista)


# In[48]:


contador = 0
for cidade in lista:
    df_copia.loc[df_copia['destination_city']==cidade, 'destination_city'] = contador
    contador += 1


# In[49]:


# dataframe codificado para o modelo
display(df_copia)
df_copia.dtypes


# In[50]:


# transformar todos os dtypes em int/float
for coluna in df_copia:
    if coluna != 'duration':
        df_copia[coluna] = df_copia[coluna].astype('int')
    
df_copia.dtypes


# In[51]:


# mapa heatmap da correlação das features 
plt.figure(figsize=(15,10))
sns.heatmap(df_copia.corr(), data=df_copia, annot=True, cmap='RdBu_r')


# ######  Modelagem + Algoritmos

# 7 etapas para aplicar a Inteligência Artificial/Machine Learning:
# 1. definir se é classificação ou regressão
# 2. escolher as métricas de avaliação
# 3. escolher os modelos a serem testados
# 4. treinamento dos modelos (machine learning)
# 5. comparar os resultados dos modelos e escolher o melhor
# 6. analisar o melhor modelo mais a fundo
# 7. fazer os ajustes do melhor modelo

# ###### 1. classificação ou regressão
# - Classificação: objetivo é classificar/criar categorias (ex: categorias A, B, C);
# - Regressão: objetivo é encontrar um valor específico (ex: preço).

# In[52]:


# Nosso modelo de previsão deve prever o preço do ticket de um voô, logo deve ser um modelo de previsão.


# ##### 2. métricas de avaliação
# - Métrica R²: o valor resultante mede o quanto o modelo consegue explicar a variância dos dados, a capacidade de previsão correta;
# - Métrica RSME (Erro Quadrático Médio): o valor resultante mede o quanto o modelo erra.

# In[53]:


def avaliacao_modelo(modelo, y_test, previsao):
    '''Parâmetros:
    modelo: nome do modelo escolhido,
    y_teste: valores a serem testados (no caso, o preço real dos imóveis),
    previsao: valor previsto
    '''
    r2 = r2_score(y_test, previsao)
    RSME = np.sqrt(mean_squared_error(y_test, previsao))
    return f'Modelo {modelo}:\nR² = {r2:.2%}\nRSME = {RSME:.2f}\n\n'


# ##### 3. escolher os modelos 
#         - Regressão Linear: este modelo traça a melhor reta que minimiza os erros dos dados/valores; é um modelo muito rápido, porém não existe uma reta que minimize todos os erros, deixando alguns dados muito fora do valor médio;
#         - Árvore de Decisão: este modelo de árvores de decisão pode ser correlacionado com o jogo do Akinator; este modelo cria várias árvores de decisão (forest) com partes aleatórias menores da base de dados e, a cada árvore, os dados vão sendo filtrados, por meio da melhor pergunta/filtro possível logo no início, até o melhor valor possível; no fim, cria-se uma média destes valores;
#         - Extra Trees: este modelo de árvores de decisão é parecido com o anterior, porém a diferença principal é que as perguntas/os filtros feitas pelo modelo para alcançar o melhor valor final são aleatórias.

# In[54]:


modelo_lr = LinearRegression()
modelo_rf = RandomForestRegressor()
modelo_et = ExtraTreesRegressor()

dicionario_modelos = {
    'LinearRegression': modelo_lr,
    'RandomForest': modelo_rf,
    'ExtraTrees': modelo_et,
}


# ##### 4. treinamento dos modelos / machine learning
#         Separa-se a base de dados em Treino e Teste:
#         - Treino: parte dos dados usados para o modelo aprender (proporção maior, 80-90%);
#         - Teste: parte dos dados usados para o modelo prever (proporção menor, 20-10%); verifica-se se o modelo funciona bem em uma situação real.
#         
#         Train_test_split: por padrão, essa função do sklearn seleciona aleatoriamente dados de X e Y e os separa em dados de treino e teste; ao aplicar o parâmetro random_state, essa seleção é sempre a mesma, ou seja, os dados de treino e teste serão sempre os mesmos.
#         
#         Os modelos usam os dados de treino (X_train, y_train) para treinar (.fit) e, então, usam os dados de teste (X_test) para tentar prever (.predict) o Y_test.

# In[55]:


# criam-se as variáveis X (dados de todas as colunas/features, menos a que será prevista) e Y (valores a serem previstos)

y = df_copia['price']
X = df_copia.drop('price', axis=1)

# separar os dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3, shuffle=True)

# treinar e testar os modelos

for nome_modelo, modelo in dicionario_modelos.items():
    #treino
    inicio = time.time()
    modelo.fit(X_train, y_train)
    
    #teste
    previsao = modelo.predict(X_test)
    fim = time.time()
    
    print(avaliacao_modelo(nome_modelo, y_test, previsao))
    print(f'Timer: {fim-inicio:,.2f} seconds\n')
    
    


# ##### 5. escolher o melhor modelo
#         A partir das métricas de avaliação escolhidas (2), comparam-se os modelos (3) e escolhe-se o melhor: o modelo com o maior valor de R² + menor valor de RSME + menor tempo de previsão + menor número de dados necessários + outros.

# In[56]:


# pelas métricas de avaliação R² e RSME e pela velocidade de treino e previsão, o melhor modelo é:

#Modelo RandomForest:
#R² = 98.53%
#RSME = 2746.16
#Timer: 52.351210832595825 seconds


# ##### 6. analisar o melhor modelo
#         - Verifica-se gráfico scatterplot dos valores previstos pelo modelo;
#         - Identifica-se a importância de cada feature (.feature_importances_) para tentar aprimorar o modelo; modelos que usam menos informações são, teoricamente, mais eficientes na previsão.

# In[57]:


# visualização gráfica scatterplot e lineplot do modelo vencedor

# scatterplot
modelo_rf.fit(X_train, y_train)
previsao = modelo_rf.predict(X_test)

fig,ax = plt.subplots(figsize=(15,15))
ax.set_title("Preço",fontsize=20)
ax.set_ylabel('Preço Previsto no Teste',fontsize=12)
ax.set_xlabel('Preço Real no Teste',fontsize=12)
ax.scatter(y_test,previsao)

r2 = r2_score(y_test, previsao)
RSME = np.sqrt(mean_squared_error(y_test, previsao))
plt.text(70000,15000,'$ R^{2} $=' + str(round(r2, 4)),fontsize=20)
plt.text(70000,10000,'MAE =' + str(round(RSME)),fontsize=20)
plt.show()

# lineplot
df_resultado = pd.DataFrame()
df_resultado['y_teste'] = y_test
df_resultado['y_previsao'] = previsao
df_resultado = df_resultado.reset_index(drop=True)

plt.figure(figsize=(15, 5))
sns.lineplot(data=df_resultado)


# In[58]:


# o método .feature_importances_ do modelo escolhido traz a importância de cada feature na mesma ordem dos dados de treino/teste

lista_valores = modelo_rf.feature_importances_
lista_colunas = X_train.columns

df_importancia_features = pd.DataFrame(index=lista_colunas, data=lista_valores)
df_importancia_features = df_importancia_features.sort_values(by=0, ascending=False)
display(df_importancia_features)

plt.figure(figsize=(15,5))
grafico = sns.barplot(x=df_importancia_features.index, y=df_importancia_features[0])
grafico.tick_params(axis='x', rotation=90)


# ###### 7. ajustar o melhor modelo
#         A partir da análise anterior (6), altera-se a base de dados (ex: excluir coluna, retirar outlier, recuperar outliers excluídos, etc.) e, a cada etapa, treina-se e testa-se o modelo novamente, para comparar a avaliação do modelo (R² e RSME) antes e depois da mudança, sempre lembrando que modelos que usam menos informações são, teoricamente, mais eficientes na previsão.
#         Dica: antes de alterar a base de dados original ao realizar os testes de ajuste do modelo, melhor criar uma cópia; caso o ajuste não for bom, teria que rodar todo o código do início para atualizar a base de dados original.

# In[59]:


# teste: excluir as colunas departure_time, arrival_time e stops
df_teste = df_copia.copy()
df_teste = df_teste.drop(['departure_time', 'arrival_time', 'stops'], axis=1)
display(df_teste)

# rodar novamente os modelos
y = df_teste['price']
X = df_teste.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

inicio = time.time()
modelo_rf.fit(X_train, y_train)

previsao = modelo_rf.predict(X_test)
fim = time.time()

print(avaliacao_modelo(nome_modelo, y_test, previsao))
print(f'Timer: {fim-inicio} seconds\n')


#     R: Houve uma mudança mínima no modelo (1% pior) então, como um modelo simples é um modelo mais eficiente, mantém-se o modelo final para o deploy.

# ### 8: Deploy
#     1. exportar a base de dados atualizada 
#     2. criar arquivo do modelo (joblib)
#     3. escolher a forma de deploy:
#         - arquivo executável (tkinter)
#         - criar um microsite na internet (flask)
#         - criar um microsite para uso direto (streamlit)
#     4. abrir outro arquivo python para criar o código do deploy
#     5. atribuir ao botão o carregamento do modelo

# 1. exportar a base de dados atualizada

# In[60]:


df_teste.to_csv('Base de Dados Final.csv', index=False)


# 2. criar arquivo do modelo (joblib)

# In[61]:


import joblib

joblib.dump(modelo_rf, 'Modelo de Previsão.joblib')


# 3. escolher a forma de deploy: Streamlit
# <br>
# 4. abrir outro arquivo python para criar o código do deploy: DeployStreamlitFlight
# <br>
# 5. atribuir ao botão o carregamento do modelo

# In[62]:


# o arquivo do jupyter deve estar na mesma pasta onde está o modelo (.joblib)

# o passo a passo da criação do deploy pela biblioteca streamlit está no outro arquivo

# após a criação de todo o deploy pelo streamlit, o arquivo deverá ser exportado como .py para rodar


# In[ ]:




