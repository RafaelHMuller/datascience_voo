<h1 align="center">
📄<br>README - Projeto Ciência de Dados Vôos
</h1>

## Índice 

* [Descrição do Projeto](#descrição-do-projeto)
* [Funcionalidades e Demonstração da Aplicação](#funcionalidades-e-demonstração-da-aplicação)
* [Pré requisitos](#pré-requisitos)
* [Execução](#execução)
* [Bibliotecas](#bibliotecas)

# Descrição do projeto
> Este repositório é um projeto Python de ciência de dados. O objetivo deste projeto foi, a partir de uma base de dados de vôos de empresas aéreas indianas obtida pelo site “Ease My Trip”, conduzir vários testes estatísticos a fim de conseguir informações relevantes. Com isso, por meio da ciência de dados, buscou-se criar um modelo de previsão capaz de prever com elevada precisão o preço de um vôo.

# Funcionalidades e Demonstração da Aplicação
Previsão do valor do ticket de um vôo a partir das principais características influenciadoras dos preços da base de dados.

Métricas de avaliação dos modelos de previsão testados:<br>
![Screenshot_1](https://user-images.githubusercontent.com/128300382/227525837-8d586f93-b297-4238-82ca-f3cadb076567.png)

Deploy do projeto (via streamlit):<br>
![Screenshot_2](https://user-images.githubusercontent.com/128300382/227525863-42901d3d-0373-4ebe-b2f0-60805f8916e7.png)


## Pré requisitos

* Sistema operacional Windows
* IDE de python (ambiente de desenvolvimento integrado de python)
* Base de dados (planilha csv)
* Prompt de comando do programa python (onde será executado o comando do streamlit para visualizar o deploy do projeto e, consequentemente, a previsão do valor do vôo) 

## Execução

* Primeiramente deve-se executar o arquivo 'Semana Cientista de Dados.py', no qual faz-se toda a análise de dados estatística, ciência de dados, machine learning e definição do melhor modelo de previsão;
*  Em seguida, executa-se o arquivo 'DeployStreamlitFlight.py', onde será gerado o deploy do modelo de previsão;
*  Por fim, dentro do prompt de comando do programa python escolhido (no meu caso, utilizei o Anaconda Prompt (anaconda3)), encontrar a pasta onde se encontra o arquivo "DeployStreamlitFlight.py" e executar o comando "streamlit run DeployStreamlitFlight.py" para visualizar, no navegador padrão do computador, o modelo preditivo do valor do vôo.

## Bibliotecas
* pandas: biblioteca que permite, no caso, a integração de arquivo excel
* seaborn, matplotlib.pyplot, plotly.express: bibliotecas de visualização gráfica
* time: biblioteca de gerenciamento de tempo no código
* sklearn: biblioteca de predição de dados (machine learning/inteligência artificial)
* joblib: biblioteca de criação do deploy do modelo de previsão
* streamlit: biblioteca de visualização do deploy do modelo de previsão no navegador padrão
