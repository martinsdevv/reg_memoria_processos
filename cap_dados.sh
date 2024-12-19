#!/bin/bash

# Caminho do arquivo CSV de saída
CSV_PATH="/home/martins/Documentos/Atividades/IA/regressao/reg_memoria_processos/dados_cap.csv"

# Verifica se o arquivo CSV já existe e o remove para evitar duplicação de dados
if [ -f "$CSV_PATH" ]; then
    rm "$CSV_PATH"
    echo "Arquivo CSV anterior removido."
fi

# Cabeçalho do CSV
echo "Name,PID,PM,CPU,Threads,Priority,FileSize,RSS,ExecTime" > "$CSV_PATH"

# Captura os 50 principais processos ativos, ordenados pelo uso de memória
ps aux --sort=-%mem | awk 'NR>1 {printf "%s,%s,%.2f,%.2f,%s,%s\n", $11, $2, $6/1024, $3, $4, $8}' | head -n 50 | while IFS=',' read -r NAME PID PM CPU THREADS PRIORITY; do
    
    # Captura o número de threads do processo
    THREAD_COUNT=$(ls /proc/$PID/task/ 2>/dev/null | wc -l)

    # Captura o tamanho do arquivo executável do processo (em MB)
    if [ -L /proc/$PID/exe ]; then
        FILESIZE=$(stat -c%s /proc/$PID/exe 2>/dev/null)
        if [ -z "$FILESIZE" ]; then
            FILESIZE="0"
        else
            FILESIZE=$(echo "scale=2; $FILESIZE / 1024 / 1024" | bc)
        fi
    else
        FILESIZE="0"
    fi

    # Captura o uso de memória residente (RSS)
    RSS=$(ps -o rss= -p $PID 2>/dev/null)
    if [ -z "$RSS" ]; then
        RSS="0"
    else
        RSS=$(echo "scale=2; $RSS / 1024" | bc)  # Converte para MB
    fi

    # Captura o tempo de execução do processo
    EXEC_TIME=$(awk '{print $22}' /proc/$PID/stat 2>/dev/null)
    if [ -z "$EXEC_TIME" ]; then
        EXEC_TIME="0"
    else
        EXEC_TIME=$(echo "scale=2; $EXEC_TIME / 100" | bc)  # Converte para segundos
    fi

    # Grava as informações no CSV
    echo "$NAME,$PID,$PM,$CPU,$THREAD_COUNT,$PRIORITY,$FILESIZE,$RSS,$EXEC_TIME" >> "$CSV_PATH"
done

# Exibe mensagem de conclusão
echo "Arquivo CSV salvo em: $CSV_PATH"
