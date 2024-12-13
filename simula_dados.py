import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 1. DADOS SIMULADOS
# -----------------
data = pd.DataFrame({
    'processos': [50, 70, 100, 120, 150, 180],
    'memoria_usada': [1200, 1350, 1800, 2000, 2500, 3000]
})

# Exibir os dados
print("\n📊 DADOS ORIGINAIS:\n", data)

# 2. CÁLCULOS BÁSICOS
# -------------------
Y = data['memoria_usada']
X = data[['processos']]  # Variável independente

# 1️⃣ Média, Moda e Mediana
media = Y.mean()
moda = Y.mode()[0] if not Y.mode().empty else None
mediana = Y.median()

# 2️⃣ Variância e Desvio Padrão
variancia = Y.var()
desvio_padrao = Y.std()

# 3️⃣ Coeficiente de Variação
coeficiente_variacao = desvio_padrao / media

# 4️⃣ Índice de Assimetria e Curtose
assimetria = stats.skew(Y)
curtose = stats.kurtosis(Y)

# 5️⃣ Intervalo de Confiança para a média
n = len(Y)
t_critico = stats.t.ppf(1 - 0.025, df=n-1)  # 95% de confiança
margem_erro = t_critico * (desvio_padrao / np.sqrt(n))
intervalo_confianca = (media - margem_erro, media + margem_erro)

modelo = LinearRegression()
modelo.fit(X, Y)
Y_pred = modelo.predict(X)

# Exibir os resultados
print("\n📊 MÉTRICAS BÁSICAS 📊")
print(f"Média: {media:.2f}")
print(f"Moda: {moda}")
print(f"Mediana: {mediana:.2f}")
print(f"Variância: {variancia:.2f}")
print(f"Desvio Padrão: {desvio_padrao:.2f}")
print(f"Coeficiente de Variação: {coeficiente_variacao:.4f}")
print(f"Índice de Assimetria: {assimetria:.4f}")
print(f"Curtose: {curtose:.4f}")
print(f"Intervalo de Confiança (95%): {intervalo_confianca}")

# 3. FREQUÊNCIAS
# ---------------
frequencias = data['memoria_usada'].value_counts().sort_index()
frequencia_acumulada = frequencias.cumsum()
frequencia_relativa = frequencias / frequencias.sum()
frequencia_percentual = frequencia_relativa * 100

# Criar a tabela de frequências
frequencias_tabela = pd.DataFrame({
    'Frequência Simples': frequencias,
    'Frequência Acumulada': frequencia_acumulada,
    'Frequência Relativa': frequencia_relativa,
    'Frequência Percentual': frequencia_percentual
})

# Exibir a tabela de frequências
print("\n📊 TABELA DE FREQUÊNCIAS 📊\n", frequencias_tabela)

# 4. PERCENTAGENS SUPERADAS POR 5%, 20%, 25%, 50%, 75%, 12%
percentis = {
    '5%': np.percentile(Y, 5),
    '12%': np.percentile(Y, 12),
    '20%': np.percentile(Y, 20),
    '25%': np.percentile(Y, 25),
    '50%': np.percentile(Y, 50),
    '75%': np.percentile(Y, 75)
}

# Exibir os percentis
print("\n📊 PERCENTUAIS DOS DADOS 📊")
for chave, valor in percentis.items():
    print(f"{chave} dos valores: {valor:.2f}")

# 5. PLOTAGEM DOS GRÁFICOS
# -------------------------

# Boxplot
title_boxplot = 'Boxplot - Memória Usada (MB)'
plt.figure(figsize=(10, 5))
sns.boxplot(x=data['memoria_usada'])
plt.title(title_boxplot)
plt.show()

# Resíduos
residuos = Y - Y_pred

# Resíduos Normalizados
residuos_normalizados = (residuos - residuos.mean()) / residuos.std()

# Histograma dos Resíduos Normalizados
plt.figure(figsize=(10, 5))
sns.histplot(residuos_normalizados, kde=True, bins=10, color="blue")
plt.title('Distribuição dos Resíduos Normalizados')
plt.xlabel('Resíduos Normalizados')
plt.ylabel('Frequência')
plt.show()

# 6. EXIBIÇÃO FINAL DOS RESULTADOS
# ---------------------------------
print("\n📊 RESULTADOS FINAIS 📊")
print("Média:", round(media, 2))
print("Moda:", moda)
print("Mediana:", round(mediana, 2))
print("Variância:", round(variancia, 2))
print("Desvio Padrão:", round(desvio_padrao, 2))
print("Coeficiente de Variação:", round(coeficiente_variacao, 4))
print("Índice de Assimetria:", round(assimetria, 4))
print("Curtose:", round(curtose, 4))
print("Intervalo de Confiança (95%):", intervalo_confianca)
print("\n📊 TABELA DE FREQUÊNCIAS 📊\n", frequencias_tabela)

print("\n📊 PERCENTUAIS DOS DADOS 📊")
for chave, valor in percentis.items():
    print(f"{chave} dos valores: {valor:.2f}")
