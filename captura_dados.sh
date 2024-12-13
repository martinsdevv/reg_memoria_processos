for i in {1..10}
do
  processos=$(ps aux | wc -l)
  memoria_usada=$(free -m | awk 'NR==2{print $3}')
  echo "$processos,$memoria_usada" >> dados_memoria.csv
  sleep 60
done
