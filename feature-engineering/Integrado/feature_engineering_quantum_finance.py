#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Engineering - QuantumFinance
Análise e Preparação de Dados para Modelo de Risco de Crédito
"""

# %% [markdown]
# # Importação de Bibliotecas
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier

# Configurações de visualização
plt.style.use('default')  # Usando o estilo padrão do matplotlib
sns.set_style('whitegrid')  # Configurando o estilo do seaborn
sns.set_palette('husl')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# %% [markdown]
# # Carregamento e Preparação Inicial dos Dados
# %%
# Carregamento dos dados
url = "https://raw.githubusercontent.com/diogenesjusto/FIAP/master/dados/credit.csv"
dados_credito = pd.read_csv(url, encoding='ISO-8859-1', sep='\t', low_memory=False)

# Nomes das colunas
nomes_cols = ['ID_CLIENTE','TIPO_FUNCIONARIO','DIA_PAGAMENTO','TIPO_ENVIO_APLICACAO','QUANT_CARTOES_ADICIONAIS',
              'TIPO_ENDERECO_POSTAL','SEXO','ESTADO_CIVIL','QUANT_DEPENDENTES','NIVEL_EDUCACIONAL',
              'ESTADO_NASCIMENTO','CIDADE_NASCIMENTO','NACIONALIDADE','ESTADO_RESIDENCIAL','CIDADE_RESIDENCIAL',
              'BAIRRO_RESIDENCIAL','FLAG_TELEFONE_RESIDENCIAL','CODIGO_AREA_TELEFONE_RESIDENCIAL','TIPO_RESIDENCIA',
              'MESES_RESIDENCIA','FLAG_TELEFONE_MOVEL','FLAG_EMAIL','RENDA_PESSOAL_MENSAL','OUTRAS_RENDAS',
              'FLAG_VISA','FLAG_MASTERCARD','FLAG_DINERS','FLAG_AMERICAN_EXPRESS','FLAG_OUTROS_CARTOES',
              'QUANT_CONTAS_BANCARIAS','QUANT_CONTAS_BANCARIAS_ESPECIAIS','VALOR_PATRIMONIO_PESSOAL','QUANT_CARROS',
              'EMPRESA','ESTADO_PROFISSIONAL','CIDADE_PROFISSIONAL','BAIRRO_PROFISSIONAL','FLAG_TELEFONE_PROFISSIONAL',
              'CODIGO_AREA_TELEFONE_PROFISSIONAL','MESES_NO_TRABALHO','CODIGO_PROFISSAO','TIPO_OCUPACAO',
              'CODIGO_PROFISSAO_CONJUGE','NIVEL_EDUCACIONAL_CONJUGE','FLAG_DOCUMENTO_RESIDENCIAL','FLAG_RG',
              'FLAG_CPF','FLAG_COMPROVANTE_RENDA','PRODUTO','FLAG_REGISTRO_ACSP','IDADE','CEP_RESIDENCIAL',
              'CEP_PROFISSIONAL','ROTULO_ALVO_MAU']

dados_credito.columns = nomes_cols

# %% [markdown]
# # Análise Exploratória Inicial
# %%
# Informações básicas do dataset
print(f"Número de linhas: {dados_credito.shape[0]}")
print(f"Número de colunas: {dados_credito.shape[1]}")
print("\nTipos de dados:")
print(dados_credito.dtypes.value_counts())

# Verificação de valores nulos
print("\nValores nulos por coluna:")
print(dados_credito.isnull().sum().sort_values(ascending=False))

# Distribuição da variável alvo
print("\nDistribuição da variável alvo (ROTULO_ALVO_MAU):")
print(dados_credito['ROTULO_ALVO_MAU'].value_counts(normalize=True))

# %% [markdown]
# # Análise Exploratória Detalhada
# %%
# Separando variáveis numéricas e categóricas
variaveis_numericas = dados_credito.select_dtypes(include=['int64', 'float64']).columns
variaveis_categoricas = dados_credito.select_dtypes(include=['object']).columns

print(f"Variáveis numéricas: {len(variaveis_numericas)}")
print(f"Variáveis categóricas: {len(variaveis_categoricas)}")

# Análise de distribuição das variáveis numéricas
plt.figure(figsize=(15, 10))
for i, col in enumerate(variaveis_numericas[:12]):  # Limitando a 12 variáveis para visualização
    plt.subplot(3, 4, i+1)
    sns.histplot(data=dados_credito, x=col, kde=True)
    plt.title(f'Distribuição de {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Análise de correlação entre variáveis numéricas
plt.figure(figsize=(15, 10))
correlation_matrix = dados_credito[variaveis_numericas].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlação')
plt.show()

# %% [markdown]
# # Tratamento de Dados Faltantes
# %%
# Identificando percentual de dados faltantes por coluna
missing_data = dados_credito.isnull().sum() / len(dados_credito) * 100
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

plt.figure(figsize=(12, 6))
missing_data.plot(kind='bar')
plt.title('Percentual de Dados Faltantes por Coluna')
plt.ylabel('Percentual (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Função para imputação de dados faltantes
def imputar_dados_faltantes(df):
    # Criando cópia do dataframe
    df_imputado = df.copy()
    
    # Imputação para variáveis numéricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_imputer = SimpleImputer(strategy='median')
    df_imputado[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    
    # Imputação para variáveis categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df_imputado[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    
    return df_imputado

# Aplicando imputação
dados_credito_imputado = imputar_dados_faltantes(dados_credito)

# Verificando se ainda existem dados faltantes
print("Dados faltantes após imputação:")
print(dados_credito_imputado.isnull().sum().sum())

# %% [markdown]
# # Normalização e Codificação de Variáveis
# %%
# Separando variáveis para normalização
variaveis_para_normalizar = ['RENDA_PESSOAL_MENSAL', 'OUTRAS_RENDAS', 'VALOR_PATRIMONIO_PESSOAL', 'IDADE']

# Aplicando StandardScaler
scaler = StandardScaler()
dados_credito_imputado[variaveis_para_normalizar] = scaler.fit_transform(
    dados_credito_imputado[variaveis_para_normalizar]
)

# Verificando a normalização
print("Estatísticas após normalização:")
print(dados_credito_imputado[variaveis_para_normalizar].describe())

# Codificação de variáveis categóricas
variaveis_categoricas_para_codificar = ['SEXO', 'ESTADO_CIVIL', 'TIPO_RESIDENCIA', 'TIPO_OCUPACAO']

# Aplicando One-Hot Encoding
dados_credito_final = pd.get_dummies(
    dados_credito_imputado, 
    columns=variaveis_categoricas_para_codificar,
    drop_first=True
)

# Identificando e codificando todas as variáveis categóricas restantes
variaveis_categoricas_restantes = dados_credito_final.select_dtypes(include=['object']).columns
if len(variaveis_categoricas_restantes) > 0:
    print(f"Variáveis categóricas restantes a serem codificadas: {variaveis_categoricas_restantes.tolist()}")
    dados_credito_final = pd.get_dummies(
        dados_credito_final,
        columns=variaveis_categoricas_restantes,
        drop_first=True
    )

print(f"Número de colunas após codificação: {dados_credito_final.shape[1]}")

# %% [markdown]
# # Seleção de Features
# %%
# Separando features e target
X = dados_credito_final.drop('ROTULO_ALVO_MAU', axis=1)
y = dados_credito_final['ROTULO_ALVO_MAU']

# Usando Random Forest para seleção de features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Obtendo importância das features
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Plotando importância das features
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
plt.title('Top 20 Features mais Importantes')
plt.tight_layout()
plt.show()

# Selecionando features com importância maior que 0.01
features_selecionadas = feature_importance[feature_importance['importance'] > 0.01]['feature'].tolist()
X_selecionado = X[features_selecionadas]

print(f"Número de features selecionadas: {len(features_selecionadas)}")

# %% [markdown]
# # Treinamento e Avaliação do Modelo
# %%
# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_selecionado, y, test_size=0.2, random_state=42, stratify=y
)

# Lista de modelos para comparação
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# Treinando e avaliando cada modelo
for nome_modelo, modelo in modelos.items():
    print(f"\n=== {nome_modelo} ===")
    
    # Treinando o modelo
    modelo.fit(X_train, y_train)
    
    # Fazendo previsões
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    
    # Avaliando o modelo
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\nAUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Plotando a importância das features para modelos que suportam
    if hasattr(modelo, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_selecionado.columns,
            'importance': modelo.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
        plt.title(f'Top 20 Features mais Importantes - {nome_modelo}')
        plt.tight_layout()
        plt.show()
    elif hasattr(modelo, 'coef_'):
        coeficientes = pd.DataFrame({
            'feature': X_selecionado.columns,
            'coeficiente': modelo.coef_[0]
        }).sort_values('coeficiente', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=coeficientes, x='coeficiente', y='feature')
        plt.title(f'Coeficientes do Modelo - {nome_modelo}')
        plt.tight_layout()
        plt.show() 