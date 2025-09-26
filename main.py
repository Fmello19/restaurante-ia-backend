# Importa o FastAPI, nosso principal utensílio de cozinha.
from fastapi import FastAPI, HTTPException
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import pytz # Para lidar com fusos horários

# Carrega as variáveis de ambiente (segredos como a senha do banco)
load_dotenv()

# Cria a aplicação principal, o nosso "Chef"
app = FastAPI()

# --- FUNÇÃO DE CONEXÃO (Ajustada para mais robustez) ---
def get_db_connection():
    """
    Cria e retorna uma conexão com o banco de dados usando o Pooler.
    Lança uma exceção em caso de falha.
    """
    try:
        conn_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:6543/{os.getenv('DB_NAME')}"
        conn = psycopg2.connect(conn_string)
        return conn
    except Exception as e:
        # Imprime o erro no log do EasyPanel para depuração
        print(f"ERRO CRÍTICO DE CONEXÃO COM O BANCO: {e}")
        # Lança uma exceção HTTP que o navegador entende
        raise HTTPException(status_code=503, detail="Não foi possível conectar ao banco de dados.")

# --- ENDPOINTS EXISTENTES (Nossas receitas antigas) ---

@app.get("/")
def read_root():
    """
    Endpoint inicial para verificar se o Chef (API) está online.
    """
    return {"status": "Chef está na cozinha e pronto para trabalhar!"}

@app.get("/clientes")
def get_clientes():
    """
    Busca e retorna a lista dos 10 últimos clientes cadastrados.
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT id, nome, telefone, created_at FROM \"Clientes_Unificados\" ORDER BY created_at DESC LIMIT 10;")
            clientes = cursor.fetchall()
        return {"clientes": clientes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar clientes: {e}")
    finally:
        if conn:
            conn.close()

# --- NOVAS RECEITAS: DASHBOARDS HISTÓRICOS ---

def calcular_metricas_periodo(inicio_periodo, fim_periodo):
    """
    Função reutilizável para calcular métricas em um intervalo de tempo específico.
    """
    query = """
        SELECT
            COALESCE(SUM(valor_total), 0)::float AS faturamento_total,
            COUNT(id) AS total_pedidos,
            COALESCE(AVG(valor_total), 0)::float AS ticket_medio,
            COUNT(DISTINCT cliente_id) AS clientes_unicos
        FROM "Pedidos_Unificados"
        WHERE created_at >= %s AND created_at < %s;
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Passando os parâmetros de forma segura para evitar SQL Injection
            cursor.execute(query, (inicio_periodo, fim_periodo))
            dados = cursor.fetchone()
        return dados
    finally:
        if conn:
            conn.close()

@app.get("/dashboard/hoje")
def get_dashboard_hoje():
    """
    Calcula e retorna as métricas principais para o dia atual (de 00:00 até agora).
    Usa o fuso horário de São Paulo.
    """
    tz_sp = pytz.timezone('America/Sao_Paulo')
    agora_sp = datetime.now(tz_sp)
    inicio_dia_sp = agora_sp.replace(hour=0, minute=0, second=0, microsecond=0)

    try:
        metricas_hoje = calcular_metricas_periodo(inicio_dia_sp, agora_sp)
        
        # Adiciona um comparativo com o dia anterior
        ontem_fim = inicio_dia_sp
        ontem_inicio = ontem_fim - timedelta(days=1)
        metricas_ontem = calcular_metricas_periodo(ontem_inicio, ontem_fim)

        return {
            "hoje": metricas_hoje,
            "comparativo_ontem": metricas_ontem
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular métricas do dia: {e}")


@app.get("/dashboard/semana")
def get_dashboard_semana():
    """
    Calcula as métricas para a semana atual (de segunda-feira até agora).
    """
    tz_sp = pytz.timezone('America/Sao_Paulo')
    agora_sp = datetime.now(tz_sp)
    # weekday() retorna 0 para segunda, 1 para terça...
    dias_para_subtrair = agora_sp.weekday()
    inicio_semana = (agora_sp - timedelta(days=dias_para_subtrair)).replace(hour=0, minute=0, second=0, microsecond=0)

    try:
        return calcular_metricas_periodo(inicio_semana, agora_sp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular métricas da semana: {e}")


@app.get("/dashboard/mes")
def get_dashboard_mes():
    """
    Calcula as métricas para o mês atual (do dia 1º até agora).
    """
    tz_sp = pytz.timezone('America/Sao_Paulo')
    agora_sp = datetime.now(tz_sp)
    inicio_mes = agora_sp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    try:
        return calcular_metricas_periodo(inicio_mes, agora_sp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular métricas do mês: {e}")

