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

# --- CONSTANTES DE NEGÓCIO ---
# No futuro, estes valores podem vir de uma tabela de configuração no banco.
MARGEM_LUCRO_PADRAO = 0.25 # 25%
CHURN_RATE_MENSAL_PADRAO = 0.10 # 10%

# --- FUNÇÃO DE CONEXÃO ---
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
        print(f"ERRO CRÍTICO DE CONEXÃO COM O BANCO: {e}")
        raise HTTPException(status_code=503, detail="Não foi possível conectar ao banco de dados.")

# --- ENDPOINTS DE DASHBOARD E TESTE ---

@app.get("/")
def read_root():
    return {"status": "Chef está na cozinha e pronto para trabalhar!"}

@app.get("/clientes")
def get_clientes():
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

def calcular_metricas_periodo(inicio_periodo, fim_periodo):
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
            cursor.execute(query, (inicio_periodo, fim_periodo))
            dados = cursor.fetchone()
        return dados
    finally:
        if conn:
            conn.close()

@app.get("/dashboard/hoje")
def get_dashboard_hoje():
    tz_sp = pytz.timezone('America/Sao_Paulo')
    agora_sp = datetime.now(tz_sp)
    inicio_dia_sp = agora_sp.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        metricas_hoje = calcular_metricas_periodo(inicio_dia_sp, agora_sp)
        ontem_fim = inicio_dia_sp
        ontem_inicio = ontem_fim - timedelta(days=1)
        metricas_ontem = calcular_metricas_periodo(ontem_inicio, ontem_fim)
        return {"hoje": metricas_hoje, "comparativo_ontem": metricas_ontem}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular métricas do dia: {e}")

@app.get("/dashboard/semana")
def get_dashboard_semana():
    tz_sp = pytz.timezone('America/Sao_Paulo')
    agora_sp = datetime.now(tz_sp)
    dias_para_subtrair = agora_sp.weekday()
    inicio_semana = (agora_sp - timedelta(days=dias_para_subtrair)).replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        return calcular_metricas_periodo(inicio_semana, agora_sp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular métricas da semana: {e}")

@app.get("/dashboard/mes")
def get_dashboard_mes():
    tz_sp = pytz.timezone('America/Sao_Paulo')
    agora_sp = datetime.now(tz_sp)
    inicio_mes = agora_sp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    try:
        return calcular_metricas_periodo(inicio_mes, agora_sp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular métricas do mês: {e}")

# --- ENDPOINTS DE INSIGHTS (IA) ---

@app.get("/insights/churn-prediction")
def get_churn_prediction():
    query = """
    WITH ranked_orders AS (
        SELECT cliente_id, valor_total, created_at,
               ROW_NUMBER() OVER(PARTITION BY cliente_id ORDER BY created_at DESC) as rn
        FROM "Pedidos_Unificados"
    ),
    customer_stats AS (
        SELECT
            cliente_id,
            COUNT(cliente_id) AS total_pedidos,
            AVG(valor_total) AS valor_medio,
            MAX(created_at) AS ultimo_pedido_data,
            EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) / 86400 / NULLIF(COUNT(cliente_id) - 1, 0) AS freq_media_dias,
            MAX(CASE WHEN rn = 1 THEN valor_total ELSE NULL END) as valor_ultimo_pedido
        FROM ranked_orders
        GROUP BY cliente_id
    ),
    final_risk AS (
        SELECT
            cs.cliente_id, uc.nome, uc.telefone, cs.total_pedidos, cs.valor_medio,
            cs.ultimo_pedido_data,
            EXTRACT(EPOCH FROM (NOW() - cs.ultimo_pedido_data)) / 86400 AS dias_sem_pedir,
            cs.freq_media_dias, cs.valor_ultimo_pedido,
            (EXTRACT(EPOCH FROM (NOW() - cs.ultimo_pedido_data)) / 86400 / cs.freq_media_dias) *
            (1 - (cs.valor_ultimo_pedido / cs.valor_medio)) AS risk_score
        FROM customer_stats cs
        JOIN "Clientes_Unificados" uc ON uc.id = cs.cliente_id
        WHERE cs.total_pedidos > 1 AND cs.freq_media_dias > 0 AND cs.valor_medio > 0
    )
    SELECT
        cliente_id, nome, telefone, total_pedidos, valor_medio::float, ultimo_pedido_data,
        dias_sem_pedir::float, freq_media_dias::float, valor_ultimo_pedido::float, risk_score::float
    FROM final_risk
    WHERE risk_score > 0.7
    ORDER BY risk_score DESC;
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            clientes_em_risco = cursor.fetchall()
        return {"clientes_em_risco": clientes_em_risco}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular Churn Prediction: {e}")
    finally:
        if conn:
            conn.close()


@app.get("/insights/ltv")
def get_ltv_ranking():
    """
    Calcula e retorna o ranking de clientes por Lifetime Value (LTV).
    """
    query = """
    WITH customer_lifetime_stats AS (
        SELECT
            cliente_id,
            COUNT(id) as total_pedidos,
            AVG(valor_total) as ticket_medio,
            NULLIF(EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) / 86400, 0) as lifetime_em_dias
        FROM "Pedidos_Unificados"
        GROUP BY cliente_id
        HAVING COUNT(id) > 1
    ),
    ltv_components AS (
        SELECT
            cls.cliente_id, uc.nome, uc.telefone, cls.ticket_medio,
            (cls.total_pedidos / cls.lifetime_em_dias) * 30 AS freq_mensal,
            cls.total_pedidos
        FROM customer_lifetime_stats cls
        JOIN "Clientes_Unificados" uc ON uc.id = cls.cliente_id
        WHERE cls.lifetime_em_dias IS NOT NULL AND cls.lifetime_em_dias > 0
    )
    SELECT
        cliente_id, nome, telefone, total_pedidos, ticket_medio::float, freq_mensal::float,
        (ticket_medio * freq_mensal * %(margem)s) / %(churn_rate)s AS ltv
    FROM ltv_components
    ORDER BY ltv DESC
    LIMIT 50;
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            params = { "margem": MARGEM_LUCRO_PADRAO, "churn_rate": CHURN_RATE_MENSAL_PADRAO }
            cursor.execute(query, params)
            ranking_ltv = cursor.fetchall()
        return {"ranking_ltv": ranking_ltv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular o LTV: {e}")
    finally:
        if conn:
            conn.close()

@app.get("/insights/market-basket")
def get_market_basket_analysis():
    """
    Realiza a Análise de Cesta de Mercado para encontrar associações de produtos.
    Retorna pares de produtos com as métricas de Suporte, Confiança e Lift.
    """
    # Esta query é o coração da análise.
    query = """
    WITH 
    -- 1. Contar a frequência de cada item individualmente
    item_counts AS (
        SELECT
            nome_produto,
            COUNT(DISTINCT pedido_id) AS total_pedidos_com_item
        FROM "Itens_Pedido"
        GROUP BY nome_produto
    ),
    -- 2. Encontrar pares de itens que aparecem no mesmo pedido
    pair_counts AS (
        SELECT
            i1.nome_produto AS item_a,
            i2.nome_produto AS item_b,
            COUNT(DISTINCT i1.pedido_id) AS total_pedidos_com_par
        FROM "Itens_Pedido" i1
        JOIN "Itens_Pedido" i2 ON i1.pedido_id = i2.pedido_id AND i1.nome_produto < i2.nome_produto
        GROUP BY i1.nome_produto, i2.nome_produto
    ),
    -- 3. Contar o total de pedidos no período (ex: últimos 90 dias)
    total_orders AS (
        SELECT COUNT(DISTINCT id) AS total_geral_pedidos
        FROM "Pedidos_Unificados"
        WHERE created_at > NOW() - INTERVAL '90 days'
    )
    -- 4. Calcular as métricas finais
    SELECT
        pc.item_a,
        pc.item_b,
        pc.total_pedidos_com_par,
        -- Suporte: Quão frequente é o par em todos os pedidos?
        (pc.total_pedidos_com_par::float / NULLIF(to.total_geral_pedidos, 0)) AS suporte,
        -- Confiança: Se comprou A, qual a chance de comprar B?
        (pc.total_pedidos_com_par::float / NULLIF(ic_a.total_pedidos_com_item, 0)) AS confianca_a_b,
        -- Lift: A compra de A aumenta a probabilidade de comprar B?
        ((pc.total_pedidos_com_par::float / NULLIF(to.total_geral_pedidos, 0)) / 
         (NULLIF(ic_a.total_pedidos_com_item, 0)::float / NULLIF(to.total_geral_pedidos, 0)) *
         (NULLIF(ic_b.total_pedidos_com_item, 0)::float / NULLIF(to.total_geral_pedidos, 0))) AS lift
    FROM pair_counts pc
    JOIN item_counts ic_a ON pc.item_a = ic_a.nome_produto
    JOIN item_counts ic_b ON pc.item_b = ic_b.nome_produto
    CROSS JOIN total_orders to
    -- Filtramos para resultados mais relevantes para evitar ruído
    WHERE pc.total_pedidos_com_par > 1 -- O par precisa ter aparecido mais de uma vez
    ORDER BY lift DESC, confianca_a_b DESC
    LIMIT 50; -- Retorna os 50 pares mais interessantes
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            associacoes = cursor.fetchall()
        return {"associacoes_de_produtos": associacoes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular a Análise de Cesta de Mercado: {e}")
    finally:
        if conn:
            conn.close()

