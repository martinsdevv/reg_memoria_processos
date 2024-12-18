import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Configuração do estilo dos gráficos
sns.set(style="whitegrid")

def carregar_dados(csv_path):
    data = pd.read_csv(csv_path, sep=',', skipinitialspace=True)
    
    # Garantir que as colunas 'PM' e 'CPU' existam
    colunas_disponiveis = data.columns
    if 'PM' not in colunas_disponiveis or 'CPU' not in colunas_disponiveis:
        raise ValueError("O arquivo CSV deve conter as colunas 'PM' e 'CPU'.")

    # Criar a coluna 'Processes' automaticamente se não existir
    if 'Processes' not in colunas_disponiveis:
        print("A coluna 'Processes' não foi encontrada. Será criada automaticamente.")
        data['Processes'] = range(1, len(data) + 1)
    
    # Garantir valores numéricos
    data['PM'] = pd.to_numeric(data['PM'].astype(str).str.replace(',', '.'), errors='coerce') / (1024 * 1024)  # MB
    data['CPU'] = pd.to_numeric(data['CPU'].astype(str).str.replace(',', '.'), errors='coerce')  # CPU em %
    data['Processes'] = pd.to_numeric(data['Processes'], errors='coerce')
    
    # Tratar valores ausentes
    data = data.dropna(subset=['PM', 'CPU', 'Processes']).reset_index(drop=True)
    
    if len(data) < 30:
        print("Aviso: O conjunto de dados tem menos de 30 amostras. O modelo pode não ser preciso.")
    
    return data

def calcular_metricas(Y):
    media = Y.mean()
    mediana = Y.median()
    variancia = Y.var()
    desvio_padrao = Y.std()
    coeficiente_variacao = desvio_padrao / media if media != 0 else 0
    
    metricas = pd.DataFrame({
        'Métrica': ["Média", "Mediana", "Variância", "Desvio Padrão", "Coeficiente de Variação"],
        'Valor': [
            round(media, 2), round(mediana, 2), round(variancia, 2), 
            round(desvio_padrao, 2), round(coeficiente_variacao, 4)
        ]
    })
    return metricas

def mostrar_boxplot(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['PM', 'CPU', 'Processes']])
    plt.title("Boxplot das Variáveis")
    plt.show()

def boxplot_relacional(df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Processes', y='PM', data=df, color="blue")
    plt.title("Dispersão: Memória Usada (PM) vs Número de Processos")
    plt.xlabel("Número de Processos")
    plt.ylabel("Memória Usada (MB)")
    plt.grid(True)
    plt.show()
    
    # Boxplot categorizado por faixas de CPU
    sns.boxplot(x=pd.qcut(df['CPU'], q=4, labels=["Baixa", "Média", "Alta", "Muito Alta"]), y='PM', data=df, showfliers=True)
    plt.title("Boxplot Relacional: PM vs CPU (Categorizado)")
    plt.xlabel("Uso de CPU (Categorizado)")
    plt.ylabel("Memória Usada (MB)")
    plt.show()


def tratar_outliers(df):
    Q1 = df['PM'].quantile(0.25)
    Q3 = df['PM'].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = df[(df['PM'] < limite_inferior) | (df['PM'] > limite_superior)]
    print(f"Número de outliers detectados: {len(outliers)}")

    opcao = input("Deseja remover os outliers? (s/n): ")
    if opcao.lower() == 's':
        df = df[(df['PM'] >= limite_inferior) & (df['PM'] <= limite_superior)]
        print("Outliers removidos.")
    else:
        print("Os outliers não foram removidos.")
    return df

def calcular_frequencias(df):
    frequencias = {
        'PM': df['PM'].value_counts().reset_index(name='Frequência'),
        'CPU': df['CPU'].value_counts().reset_index(name='Frequência'),
        'Processes': df['Processes'].value_counts().reset_index(name='Frequência')
    }
    for coluna, freq in frequencias.items():
        print(f"\nFrequência da variável '{coluna}':")
        print(freq.head(10))

def calcular_percentis(Y):
    percentis = np.percentile(Y, [5, 20, 25, 50, 75, 95])
    percentis_df = pd.DataFrame({
        "Percentil": ["5%", "20%", "25%", "50%", "75%", "95%"],
        "Valor": np.round(percentis, 4)
    })
    print("\nPercentis da Memória Usada (PM):")
    print(percentis_df)
    return percentis_df


def regressao_multipla(df):
    X = df[['CPU', 'Processes']]
    Y = df['PM']  # Memória Usada (MB)
    
    if X.empty or Y.empty:
        print("Erro: Dados insuficientes para executar a regressão múltipla.")
        return None, None, None
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    modelo = LinearRegression()
    modelo.fit(X_train, Y_train)
    Y_pred = modelo.predict(X_test)
    
    r2 = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    
    coef_cpu, coef_processes = modelo.coef_
    intercepto = modelo.intercept_
    
    print("\nCoeficientes da Regressão Múltipla:")
    print(f"CPU: {coef_cpu:.4f}")
    print(f"Processes: {coef_processes:.4f}")
    print(f"Intercepto: {intercepto:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(df[['CPU', 'Processes']].corr())
    
    print("\nFórmula da Regressão:")
    print(f"PM = {coef_cpu:.4f} * CPU + {coef_processes:.4f} * Processes + {intercepto:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['CPU'], Y_test, color='blue', label='Dados Reais (CPU)')
    plt.scatter(X_test['Processes'], Y_test, color='green', label='Dados Reais (Processes)')
    plt.plot(X_test['CPU'], modelo.predict(X_test), color='red', linewidth=2, label='Predição')
    plt.title("Regressão Múltipla: PM vs CPU e Processes")
    plt.xlabel("CPU (%) e Quantidade de Processos")
    plt.ylabel("Memória Usada (MB)")
    plt.legend()
    plt.show()
    
    return modelo, r2, mse

def grafico_residuos(df, modelo):
    X = df[['CPU', 'Processes']]
    Y = df['PM']
    Y_pred = modelo.predict(X)
    residuos = Y - Y_pred
    sns.histplot(residuos, kde=True, color='red')
    plt.title("Histograma dos Resíduos")
    plt.xlabel("Resíduos")
    plt.show()

def menu():
    csv_path = r"C:\\Users\\pmgtec\\Documents\\martins\\memoria_processos\\reg_memoria_processos\\dados_cap.csv"
    df = carregar_dados(csv_path)
    
    while True:
        print("\nMENU PRINCIPAL")
        print("1. Exibir Dados")
        print("2. Calcular Métricas")
        print("3. Mostrar Boxplot Relacional")
        print("4. Tratar Outliers")
        print("5. Calcular Frequências")
        print("6. Regressão Múltipla")
        print("7. Gráfico de Resíduos")
        print("8. Calcular Percentis")
        print("9. Sair")
        
        opcao = input("Escolha uma opção: ")
        if opcao == "1":
            print(df.head())
        elif opcao == "2":
            metricas = calcular_metricas(df['PM'])
            print(metricas)
        elif opcao == "3":
            boxplot_relacional(df)
        elif opcao == "4":
            df = tratar_outliers(df)
        elif opcao == "5":
            calcular_frequencias(df)
        elif opcao == "6":
            modelo, r2, mse = regressao_multipla(df)
        elif opcao == "7":
            modelo, _, _ = regressao_multipla(df)
            if modelo is not None:
                grafico_residuos(df, modelo)
        elif opcao == "8":
            calcular_percentis(df['PM'])
        elif opcao == "9":
            print("Saindo...")
            break
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    menu()