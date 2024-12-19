import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuração do estilo dos gráficos
sns.set(style="whitegrid")
matplotlib.use('TkAgg')

def carregar_dados(csv_path):
    data = pd.read_csv(csv_path, sep=',', skipinitialspace=True)
    colunas_obrigatorias = ['PM', 'CPU', 'Threads']
    for coluna in colunas_obrigatorias:
        if coluna not in data.columns:
            raise ValueError(f"O arquivo CSV deve conter a coluna '{coluna}'.")
    for coluna in colunas_obrigatorias:
        data[coluna] = pd.to_numeric(data[coluna].astype(str).str.replace(',', '.'), errors='coerce')
    data = data.dropna(subset=colunas_obrigatorias).reset_index(drop=True)
    data = data[data['CPU'] != 0]
    return data

def tratar_outliers(df, colunas):
    for coluna in colunas:
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 3.0 * IQR
        limite_superior = Q3 + 3.0 * IQR
        df = df[(df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)]
    return df



def calcular_metricas(df):
    # Calcula as métricas básicas
    metricas = {
        "Média": df[['PM', 'CPU', 'Threads']].mean(),
        "Moda": df[['PM', 'CPU', 'Threads']].mode().iloc[0],
        "Mediana": df[['PM', 'CPU', 'Threads']].median(),
        "Variância": df[['PM', 'CPU', 'Threads']].var(),
        "Desvio Padrão": df[['PM', 'CPU', 'Threads']].std(),
        "Coeficiente de Variação": df[['PM', 'CPU', 'Threads']].std() / df[['PM', 'CPU', 'Threads']].mean()
    }

    # Adicionando o intervalo de confiança de 95% a cada variável
    ic = {}
    for coluna in ['PM', 'CPU', 'Threads']:
        n = len(df[coluna])
        mean = df[coluna].mean()
        sem = df[coluna].std() / np.sqrt(n)
        margin = t.ppf(0.975, df=n-1) * sem
        ic[coluna] = f"[{mean - margin:.2f}, {mean + margin:.2f}]"
    
    # Adicionando o intervalo de confiança às métricas
    metricas["Intervalo de Confiança (95%)"] = pd.Series(ic)
    
    # Converte o dicionário para DataFrame
    metricas_df = pd.DataFrame(metricas)
    
    print("\nMétricas Calculadas:")
    print(metricas_df)

def regressao_multipla(df):
    X = df[['CPU', 'Threads']]
    Y = np.log1p(df['PM'])  # Transformação logarítmica
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, Y_train)
    Y_pred = modelo.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    coeficientes = modelo.coef_
    intercepto = modelo.intercept_

    # Exibir coeficientes e métricas
    coeficientes_df = pd.DataFrame({'Variável': ['CPU', 'Threads'], 'Coeficiente': coeficientes})
    print("\nCoeficientes da Regressão Múltipla:")
    print(coeficientes_df)
    print(f"Intercepto: {intercepto:.4f}")
    print(f"R²: {r2:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    # Exibir a fórmula da regressão
    formula = f"PM = {intercepto:.4f} + {coeficientes[0]:.4f} * CPU + {coeficientes[1]:.4f} * Threads"
    print(f"\nFórmula da Regressão: {formula}")

    return modelo, r2, mse, formula

def calcular_percentis(df, colunas):
    percentis = {}
    for coluna in colunas:
        percentis[coluna] = {
            "5%": df[coluna].quantile(0.05),
            "25%": df[coluna].quantile(0.25),
            "50% (Mediana)": df[coluna].quantile(0.5),
            "75%": df[coluna].quantile(0.75),
            "95%": df[coluna].quantile(0.95)
        }
    percentis_df = pd.DataFrame(percentis)
    print("\nPercentis Calculados:")
    print(percentis_df)

def calcular_frequencias(df, coluna):
    frequencias = {
        "Frequência Simples": df[coluna].value_counts(),
        "Frequência Relativa": df[coluna].value_counts(normalize=True)
    }
    frequencias_df = pd.DataFrame(frequencias)
    frequencias_df["Frequência Acumulada"] = frequencias_df["Frequência Simples"].cumsum()
    print("\nFrequências Calculadas:")
    print(frequencias_df)

def exibir_boxplot(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='PM')
    plt.title("Boxplot de PM (Memória)")
    plt.show()


def grafico_residuos(df, modelo):
    X = df[['CPU', 'Threads']]
    Y = np.log1p(df['PM'])
    Y_pred = modelo.predict(X)
    residuos = Y - Y_pred
    sns.histplot(residuos, kde=True, color='red')
    plt.title("Histograma dos Resíduos")
    plt.xlabel("Resíduos")
    plt.show()

def menu():
    csv_path = "dados_cap.csv"
    df = carregar_dados(csv_path)
    df_tratado = tratar_outliers(df, ['PM', 'CPU', 'Threads'])

    while True:
        print("\nMENU PRINCIPAL")
        print("1. Exibir Dados")
        print("2. Regressão Múltipla")
        print("3. Gráfico de Resíduos")
        print("4. Calcular Métricas")
        print("5. Calcular Percentis")
        print("6. Calcular Frequências")
        print("7. Exibir Boxplot")
        print("8. Sair")

        opcao = input("Escolha uma opção: ")

        if opcao == "1":
            print("\nExibindo as primeiras 5 linhas do DataFrame:")
            print(df_tratado.head())
        elif opcao == "2":
            modelo, r2, mse, formula = regressao_multipla(df_tratado)  # Captura todos os valores retornados
        elif opcao == "3":
            modelo, r2, mse, formula = regressao_multipla(df_tratado)  # Atualizado para capturar a fórmula
            if modelo is not None:
                grafico_residuos(df_tratado, modelo)
        elif opcao == "4":
            calcular_metricas(df_tratado)
        elif opcao == "5":
            calcular_percentis(df_tratado, ['PM', 'CPU', 'Threads'])
        elif opcao == "6":
            coluna = input("Digite o nome da coluna para calcular frequências: ")
            if coluna in df_tratado.columns:
                calcular_frequencias(df_tratado, coluna)
            else:
                print(f"Coluna '{coluna}' não encontrada no DataFrame.")
        elif opcao == "7":
            exibir_boxplot(df_tratado)
        elif opcao == "8":
            print("Saindo...")
            break
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    menu()
