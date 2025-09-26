# Importa o FastAPI, nosso principal utensílio de cozinha.
from fastapi import FastAPI, HTTPException
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
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

# --- NOVA RECEITA: DASHBOARD DO DIA ---

@app.get("/dashboard/hoje")
def get_dashboard_hoje():
    """
    Calcula e retorna as métricas principais para o dia atual (de 00:00 até agora).
    Usa o fuso horário de São Paulo para definir o que é "hoje".
    """
    query = """
    WITH hoje_bounds AS (
        SELECT
            date_trunc('day', now() AT TIME ZONE 'America/Sao_Paulo') AS inicio_dia,
            now() AT TIME ZONE 'America/Sao_Paulo' AS agora
    )
    SELECT
        COALESCE(SUM(valor_total), 0)::float AS faturamento_total,
        COUNT(id) AS total_pedidos,
        COALESCE(AVG(valor_total), 0)::float AS ticket_medio,
        COUNT(DISTINCT cliente_id) AS clientes_unicos
    FROM "Pedidos_Unificados", hoje_bounds
    WHERE created_at >= (inicio_dia AT TIME ZONE 'America/Sao_Paulo')
      AND created_at <= (agora AT TIME ZONE 'America/Sao_Paulo');
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            # fetchone() pega a única linha de resultado da nossa query
            dados_hoje = cursor.fetchone()
        return dados_hoje
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular métricas do dia: {e}")
    finally:
        if conn:
            conn.close()

