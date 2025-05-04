#!/bin/bash

# Uso:
# ./run_config_for_apps.sh --cores=4 --warps=8 --threads=64 --policy=1 --args=n128

# Caminho para o script do simulador
SIMULATOR_SCRIPT="./ci/blackbox.sh"

# Lista de aplicativos a serem testados
APPS=("nearn" "sgemm" "vecadd")

# Parsing de argumentos
for ARG in "$@"; do
    case $ARG in
        --cores=*) CORES="${ARG#*=}" ;;
        --warps=*) WARPS="${ARG#*=}" ;;
        --threads=*) THREADS="${ARG#*=}" ;;
        --policy=*) POLICY="${ARG#*=}" ;;
        --args=*) ARGS="${ARG#*=}" ;;
        *) echo "Parâmetro inválido: $ARG" && exit 1 ;;
    esac
done

# Verificação obrigatória de parâmetros
if [ -z "$CORES" ] || [ -z "$WARPS" ] || [ -z "$THREADS" ] || [ -z "$POLICY" ] || [ -z "$ARGS" ]; then
    echo "Uso: $0 --cores=N --warps=N --threads=N --policy=N --args=ARGS"
    exit 1
fi

# Loop pelos apps pré-definidos
for APP in "${APPS[@]}"; do
    echo "Executando configuração para o app: $APP"

    # Criar diretório de resultados
    RESULTS_DIR="results/${APP}"
    mkdir -p "$RESULTS_DIR"

    # Nome do arquivo de resultado com identificação dos parâmetros
    LOG_FILE="${RESULTS_DIR}/log_cores${CORES}_warps${WARPS}_threads${THREADS}_policy${POLICY}_args${ARGS}.txt"

    # Executar simulador
    $SIMULATOR_SCRIPT \
        --rebuild=1 \
        --perf=1 \
        --debug=1 \
        --app=$APP \
        --cores=$CORES \
        --warps=$WARPS \
        --threads=$THREADS \
        --policy=$POLICY \
        --args=$ARGS \
        --log="$LOG_FILE"

    # Checar sucesso
    if [ $? -eq 0 ]; then
        echo "[$APP] Execução finalizada com sucesso. Log salvo em $LOG_FILE"
    else
        echo "[$APP] Erro durante a execução. Verifique o log em $LOG_FILE"
    fi
done