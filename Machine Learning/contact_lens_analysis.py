import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
df = pd.read_csv('contact_lens.csv')

# Análise exploratória inicial
print("\nInformações do Dataset:")
print("-" * 50)
print(df.info())

print("\nDistribuição das Prescrições de Lentes:")
print("-" * 50)
print(df['lente'].value_counts())
print("\nPorcentagem:")
print(df['lente'].value_counts(normalize=True).mul(100).round(1))

print("\nRelação entre Diagnóstico e Tipo de Lente:")
print("-" * 50)
print(pd.crosstab(df['diagnostico'], df['lente']))

print("\nRelação entre Produção Lacrimal e Tipo de Lente:")
print("-" * 50)
print(pd.crosstab(df['lacrimal'], df['lente']))

# Preparação para Machine Learning
# Converter variáveis categóricas em numéricas
le_idade = LabelEncoder()
le_diagnostico = LabelEncoder()
le_astigmatismo = LabelEncoder()
le_lacrimal = LabelEncoder()
le_lente = LabelEncoder()

# Preparar os dados de treino
X = df[['idade', 'diagnostico', 'astigmatismo', 'lacrimal']]
y = df['lente']

# Aplicar Label Encoding para cada coluna
X_encoded = pd.DataFrame({
    'idade': le_idade.fit_transform(X['idade']),
    'diagnostico': le_diagnostico.fit_transform(X['diagnostico']),
    'astigmatismo': le_astigmatismo.fit_transform(X['astigmatismo']),
    'lacrimal': le_lacrimal.fit_transform(X['lacrimal'])
})
y_encoded = le_lente.fit_transform(y)

# Criar e treinar o modelo de árvore de decisão
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(X_encoded, y_encoded)

# Calcular a importância das características
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nImportância das Características:")
print("-" * 50)
print(feature_importance)

# Criar visualização da árvore de decisão
plt.figure(figsize=(15,10))
plot_tree(clf, feature_names=X.columns, class_names=le_lente.classes_, filled=True)
plt.savefig('arvore_decisao.png')
plt.close()

print("\nUma visualização da árvore de decisão foi salva como 'arvore_decisao.png'")

# Agrupar por todas as características exceto ID e lente
caracteristicas = ['idade', 'diagnostico', 'astigmatismo', 'lacrimal']
grupos = df.groupby(caracteristicas)

print("\nAnálise de Casos com Mesmas Características:")
print("-" * 50)

# Procurar grupos com prescrições diferentes
for caracteristica, grupo in grupos:
    if len(grupo['lente'].unique()) > 1:
        print(f"\nPacientes com características idênticas mas prescrições diferentes:")
        print(f"Características: {dict(zip(caracteristicas, caracteristica))}")
        print("\nCasos encontrados:")
        print(grupo[['id', 'lente']].to_string())
        print("\n")

# Verificar se não encontrou nenhum caso
if all(len(grupo['lente'].unique()) == 1 for _, grupo in grupos):
    print("Não foram encontrados casos de pacientes com mesmas características e prescrições diferentes.")

# Novos pacientes para previsão
novos_pacientes = pd.DataFrame([
    ['ID31', 'pre-presbiópico', 'hipermetrope', 'sim', 'normal'],
    ['ID32', 'presbiópico', 'hipermetrope', 'sim', 'normal'],
    ['ID33', 'jovem', 'míope', 'sim', 'normal'],
    ['ID34', 'presbiópico', 'hipermetrope', 'sim', 'normal']
], columns=['id', 'idade', 'diagnostico', 'astigmatismo', 'lacrimal'])

# Preparar dados dos novos pacientes
X_novo = novos_pacientes[['idade', 'diagnostico', 'astigmatismo', 'lacrimal']]
X_novo_encoded = pd.DataFrame({
    'idade': le_idade.transform(X_novo['idade']),
    'diagnostico': le_diagnostico.transform(X_novo['diagnostico']),
    'astigmatismo': le_astigmatismo.transform(X_novo['astigmatismo']),
    'lacrimal': le_lacrimal.transform(X_novo['lacrimal'])
})

# Fazer previsões
previsoes = le_lente.inverse_transform(clf.predict(X_novo_encoded))

print("\nPrevisões de Lentes para Novos Pacientes:")
print("-" * 50)
for i, (_, paciente) in enumerate(novos_pacientes.iterrows()):
    print(f"\nPaciente {paciente['id']}:")
    print(f"Características: {paciente['idade']}, {paciente['diagnostico']}, "
          f"astigmatismo: {paciente['astigmatismo']}, lacrimal: {paciente['lacrimal']}")
    print(f"Lente Prescrita: {previsoes[i]}")
    
    # Encontrar casos similares na base de dados
    casos_similares = df[
        (df['idade'] == paciente['idade']) &
        (df['diagnostico'] == paciente['diagnostico']) &
        (df['astigmatismo'] == paciente['astigmatismo']) &
        (df['lacrimal'] == paciente['lacrimal'])
    ]
    
    if not casos_similares.empty:
        print("Justificativa: Baseado em casos similares na base de dados:")
        print(casos_similares[['id', 'lente']].to_string())
    else:
        print("Justificativa: Baseado no padrão aprendido pelo modelo de árvore de decisão")
