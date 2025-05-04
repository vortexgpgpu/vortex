#!/bin/bash

# Uso:
# ./run_config_combinations.sh [--app=APP] [--args=ARGS]

# Caminho para o script do simulador
SIMULATOR_SCRIPT="./ci/blackbox.sh "

# Valores default
APP="nearn"
ARGS=""

# Listas de combinações
CORES_LIST=(2 4 8)
WARPS_LIST=(4 8 16)
THREADS_LIST=(8 16 24 32)
POLICY_LIST=(0 5)

# Parsing dos argumentos
for ARG in "$@"; do
    case $ARG in
        --app=*) APP="${ARG#*=}" ;;
        --args=*) ARGS="${ARG#*=}" ;;
        *) echo "Parâmetro inválido: $ARG" && exit 1 ;;
    esac
done

# Criar diretório de resultados
RESULTS_DIR="results/${APP}"
mkdir -p "$RESULTS_DIR"

# Loop de combinações
for CORES in "${CORES_LIST[@]}"; do
    for WARPS in "${WARPS_LIST[@]}"; do
        for THREADS in "${THREADS_LIST[@]}"; do
            for POLICY in "${POLICY_LIST[@]}"; do

                # Nome do arquivo de log
                SAFE_ARGS=${ARGS:-"default"}
                RESULTS_FILE="${RESULTS_DIR}/results_cores${CORES}_warps${WARPS}_threads${THREADS}_policy${POLICY}_args${SAFE_ARGS}.txt"
                LOG_FILE="${RESULTS_DIR}/log_cores${CORES}_warps${WARPS}_threads${THREADS}_policy${POLICY}_args${SAFE_ARGS}.txt"


                echo "Rodando: app=$APP, cores=$CORES, warps=$WARPS, threads=$THREADS, policy=$POLICY, args=$SAFE_ARGS"

                # Executa e filtra a saída
                if [ -n "$ARGS" ]; then
                    ./ci/blackbox.sh --rebuild=1 --perf=1 --debug=1 --app="$APP" --args="$ARGS" --cores="$CORES" --warps="$WARPS" --threads="$THREADS" --policy="$POLICY" --log="$LOG_FILE" > "$LOG_FILE" 2>&1
                else
                    ./ci/blackbox.sh --rebuild=1 --perf=1 --debug=1 --app="$APP" --cores="$CORES" --warps="$WARPS" --threads="$THREADS" --policy="$POLICY" --log="$LOG_FILE" > "$LOG_FILE" 2>&1
                fi

                cat $LOG_FILE | grep '^PERF' > "$RESULTS_FILE"

                if [ $? -eq 0 ]; then
                    echo "Finalizado com sucesso. Log: $LOG_FILE"
                else
                    echo "Erro ao executar. Verifique: $LOG_FILE"
                fi
                echo
            done
        done
    done
done
