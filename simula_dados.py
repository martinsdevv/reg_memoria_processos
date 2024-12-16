import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score

# Configuração do estilo dos gráficos
sns.set(style="whitegrid")

def carregar_dados(csv_path):
    data = pd.read_csv(csv_path, sep=',', skipinitialspace=True)
    data.columns = ['Name', 'Id', 'PM']
    data['PM'] = pd.to_numeric(data['PM'], errors='coerce') / (1024 * 1024)
    data = data.dropna().reset_index(drop=True)
    return data.groupby('Name', as_index=False)['PM'].sum()

def calcular_metricas(df):
    Y = df['memoria_usada']
    media = Y.mean()
    moda = Y.mode()[0] if not Y.mode().empty else None
    mediana = Y.median()
    variancia = Y.var()
    desvio_padrao = Y.std()
    coeficiente_variacao = desvio_padrao / media
    assimetria = stats.skew(Y)
    curtose = stats.kurtosis(Y)
    n = len(Y)
    t_critico = stats.t.ppf(1 - 0.025, df=n-1)
    margem_erro = t_critico * (desvio_padrao / np.sqrt(n))
    intervalo_confianca = (media - margem_erro, media + margem_erro)

    metricas = pd.DataFrame({
        'Métrica': ["Média", "Moda", "Mediana", "Variância", "Desvio Padrão", 
                    "Coeficiente de Variação", "Assimetria", "Curtose", "Intervalo de Confiança (95%)"],
        'Valor (MB)': [
            round(media, 2), round(moda, 2), round(mediana, 2), 
            round(variancia, 2), round(desvio_padrao, 2), 
            round(coeficiente_variacao, 4), round(assimetria, 4), 
            round(curtose, 4), f"({round(intervalo_confianca[0], 2)}, {round(intervalo_confianca[1], 2)})"
        ]
    })
    return metricas

def separar_outliers(df, fator=1.5):
    Q1 = df['memoria_usada'].quantile(0.25)
    Q3 = df['memoria_usada'].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - fator * IQR
    limite_superior = Q3 + fator * IQR

    print(f"Limite Inferior: {round(limite_inferior, 2)} | Limite Superior: {round(limite_superior, 2)}")

    # Dados sem outliers
    dados_sem_outliers = df[(df['memoria_usada'] >= limite_inferior) & (df['memoria_usada'] <= limite_superior)].copy()

    # Dados considerados outliers
    dados_outliers = df[(df['memoria_usada'] < limite_inferior) | (df['memoria_usada'] > limite_superior)].copy()

    return dados_sem_outliers, dados_outliers

def clusterizar_dados(df, n_clusters=3):
    X = df[['processos', 'memoria_usada']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='processos', y='memoria_usada', hue='Cluster', palette='Set2', s=100)
    plt.title("Clusterização dos Dados")
    plt.xlabel("Processos")
    plt.ylabel("Memória Usada (MB)")
    plt.legend()
    plt.show()

def exibir_tabela(df, titulo):
    print(f"\n{titulo}")
    print(df)

def regressao_linear(df):
    X = df[['processos']] 
    Y = df['memoria_usada'] 
    modelo = LinearRegression()
    modelo.fit(X, Y)
    Y_pred = modelo.predict(X)
    
    # Cálculo de R²
    r2 = r2_score(Y, Y_pred)
    print(f"R² Linear: {r2:.4f}")
    print("X: Número de processos | Y: Memória usada (em MB)")
    
    plt.scatter(X, Y, color='blue', label='Dados Reais')
    plt.plot(X, Y_pred, color='red', label='Regressão Linear')
    plt.title(f"Regressão Linear: Memória Usada vs Número de Processos\nR²: {r2:.4f}")
    plt.xlabel("Número de Processos (X)")
    plt.ylabel("Memória Usada (MB) - Y")
    plt.legend()
    plt.show()



def regressao_polinomial(df):
    grau = int(input("Digite o grau da regressão polinomial (ex: 2, 3): "))
    X = df[['processos']]
    Y = df['memoria_usada']
    poly = PolynomialFeatures(degree=grau)
    X_poly = poly.fit_transform(X)
    modelo = LinearRegression()
    modelo.fit(X_poly, Y)
    Y_pred = modelo.predict(X_poly)
    r2 = r2_score(Y, Y_pred)
    print(f"R² Polinomial (Grau {grau}): {r2:.4f}")
    print("X: Número de processos | Y: Memória usada (em MB)")
    
    plt.scatter(X, Y, color='blue', label='Dados Reais')
    plt.plot(X, Y_pred, color='red', label=f'Regressão Polinomial - Grau {grau}')
    plt.title(f"Regressão Polinomial: Memória Usada vs Número de Processos\nR²: {r2:.4f}")
    plt.xlabel("Número de Processos (X)")
    plt.ylabel("Memória Usada (MB) - Y")
    plt.legend()
    plt.show()


def avaliar_modelo_polinomial(df, grau=2):
    X = df[['processos']]
    Y = df['memoria_usada']
    poly = PolynomialFeatures(degree=grau)
    X_poly = poly.fit_transform(X)

    modelo = LinearRegression()
    scores = cross_val_score(modelo, X_poly, Y, cv=5, scoring='r2')  # 5-fold CV
    print(f"R² Médio (Validação Cruzada - Grau {grau}): {np.mean(scores):.4f}")

def boxplot(Y, titulo):
    sns.boxplot(x=Y)
    plt.title(titulo)
    plt.show()

def histograma_residuos(df):
    X = df[['processos']]
    Y = df['memoria_usada']
    modelo = LinearRegression()
    modelo.fit(X, Y)
    residuos = Y - modelo.predict(X)
    sns.histplot(residuos, kde=True)
    plt.title("Histograma dos Resíduos")
    plt.show()

def tabela_percentis(df):
    Y = df['memoria_usada']
    percentis = {f"{p}%": round(np.percentile(Y, p), 2) for p in [5, 25, 50, 75, 95]}
    print("\nTabela de Percentis:")
    for k, v in percentis.items():
        print(f"{k}: {v} MB")

def prever_memoria(modelo, processos, memoria_total=16):  # Exemplo: 16 GB total
    memoria_usada = modelo.predict([[processos]])
    memoria_livre = memoria_total - memoria_usada
    print(f"Memória Prevista Usada: {memoria_usada[0]:.2f} GB")
    print(f"Memória Livre: {memoria_livre[0]:.2f} GB")


# Função principal
if __name__ == "__main__":
    csv_path = r"C:\\Users\\pmgtec\\Documents\\dados.csv"
    data_agrupada = carregar_dados(csv_path)
    data_agrupada['processos'] = range(1, len(data_agrupada) + 1)
    data_agrupada = data_agrupada.rename(columns={'PM': 'memoria_usada'})
    
    processos_comuns, processos_pesados = separar_outliers(data_agrupada)

    while True:
        print("\nMENU PRINCIPAL")
        print("1. Processos Comuns")
        print("2. Processos Pesados")
        print("3. Sair")
        opcao = input("Escolha uma opção: ")
        
        if opcao == "1":
            grupo = processos_comuns
            print("\n--- PROCESSOS COMUNS ---")
        elif opcao == "2":
            grupo = processos_pesados
            print("\n--- PROCESSOS PESADOS ---")
        elif opcao == "3":
            print("Saindo...")
            break
        else:
            print("Opção inválida!")
            continue

        while True:
            print("\nMENU DE ANÁLISE")
            print("1. Dados")
            print("2. Gráfico Linear")
            print("3. Métricas")
            print("4. Boxplot")
            print("5. Histograma")
            print("6. Percentis")
            print("7. Regressão Polinomial")
            if opcao == "2":
                print("8. Clusterizar Dados")
            print("9. Podar Outliers")
            print("10. Voltar")

            opcao_analise = input("Escolha uma opção: ")
            
            if opcao_analise == "1":
                exibir_tabela(grupo, "Dados")
            elif opcao_analise == "2":
                regressao_linear(grupo)
            elif opcao_analise == "3":
                metricas = calcular_metricas(grupo)
                exibir_tabela(metricas, "Métricas")
            elif opcao_analise == "4":
                boxplot(grupo['memoria_usada'], "Boxplot")
            elif opcao_analise == "5":
                histograma_residuos(grupo)
            elif opcao_analise == "6":
                tabela_percentis(grupo)
            elif opcao_analise == "7":
                regressao_polinomial(grupo)
            elif opcao_analise == "9":
                fator = float(input("Digite o fator IQR para poda (ex: 1.0, 0.75): "))
                grupo, outliers = separar_outliers(grupo, fator)

                print("\nDados após poda (sem outliers):")
                print(grupo)

                print("\nOutliers identificados:")
                print(outliers)
            elif opcao_analise == "8" and opcao == "2":
                n_clusters = int(input("Digite o número de clusters (ex: 2, 3): "))
                clusterizar_dados(grupo, n_clusters)
            elif opcao_analise == "8" and opcao == "2":
                n_clusters = int(input("Digite o número de clusters (ex: 2, 3): "))
                clusterizar_dados(grupo, n_clusters)

            elif opcao_analise == "10":
                break
            else:
                print("Opção inválida!")
