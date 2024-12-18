### Script para capturar dados de processos do Windows e salvar em CSV ###

# Caminho do arquivo CSV de saída
$csvPath = "C:\Users\pmgtec\Documents\martins\memoria_processos\reg_memoria_processos\dados_cap.csv"

# Verifica se o arquivo CSV já existe e o remove para evitar duplicação de dados
if (Test-Path $csvPath) {
    Remove-Item $csvPath
}

# Captura os 50 primeiros processos ativos no sistema, ordenados pelo uso de memória
$processos = Get-Process | Sort-Object -Property PM -Descending | Select-Object -First 50 -Property `
    @{Name='Name'; Expression={$_.Name}}, `
    @{Name='Id'; Expression={$_.Id}}, `
    @{Name='PM'; Expression={$_.PagedMemorySize64 / 1024}}, `
    @{Name='CPU'; Expression={$_.CPU}}, `
    @{Name='Threads'; Expression={$_.Threads.Count}}, `
    @{Name='Prioridade'; Expression={$_.PriorityClass}}, `
    @{Name='Tamanho'; Expression={(Get-ItemProperty -Path $_.Path -ErrorAction SilentlyContinue).Length / 1024}}  # Tamanho do binário em KB

# Remove processos que não têm dados de CPU ou caminho de execução
$processosFiltrados = $processos | Where-Object {
    $_.CPU -ne $null -and $_.Tamanho -ne $null
}

# Converte o objeto em formato CSV
$processosFiltrados | Export-Csv -Path $csvPath -NoTypeInformation -Force -Delimiter ","

# Exibe mensagem de conclusão
Write-Host "Arquivo CSV salvo em: $csvPath"
