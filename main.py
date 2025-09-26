# app.py - API FastAPI Melhorada com Async, Pool de Conexões e Boas Práticas
import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from functools import lru_cache

import asyncpg
import pytz
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import uvicorn

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente
load_dotenv()

# ========================= CONFIGURAÇÕES =========================

class Settings:
    """Configurações centralizadas da aplicação"""
    # Database - Configuração para Supabase/EasyPanel
    DB_HOST: str = os.getenv('DB_HOST', '')
    DB_NAME: str = os.getenv('DB_NAME', 'postgres')
    DB_USER: str = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD: str = os.getenv('DB_PASSWORD', '')
    DB_PORT: int = int(os.getenv('DB_PORT', '6543'))
    
    # Pool de Conexões
    DB_POOL_MIN_SIZE: int = int(os.getenv('DB_POOL_MIN_SIZE', '10'))
    DB_POOL_MAX_SIZE: int = int(os.getenv('DB_POOL_MAX_SIZE', '20'))
    
    # Business Rules
    MARGEM_LUCRO_PADRAO: float = float(os.getenv('MARGEM_LUCRO_PADRAO', '0.25'))
    CHURN_RATE_MENSAL_PADRAO: float = float(os.getenv('CHURN_RATE_MENSAL_PADRAO', '0.10'))
    
    # API Settings
    API_VERSION: str = "1.0.0"
    API_TITLE: str = "Chef API - Sistema de Gestão"
    CORS_ORIGINS: List[str] = os.getenv('CORS_ORIGINS', '*').split(',')
    
    # Cache
    CACHE_TTL_SECONDS: int = int(os.getenv('CACHE_TTL_SECONDS', '300'))  # 5 minutos
    
    # Timezone
    TIMEZONE: str = 'America/Sao_Paulo'
    
    def get_connection_string(self) -> str:
        """Constrói a connection string para o banco de dados"""
        # Se DB_HOST já começa com postgresql:// ou postgres://, usa direto
        if self.DB_HOST.startswith(('postgresql://', 'postgres://')):
            return self.DB_HOST
        
        # Senão, constrói a connection string
        if self.DB_PASSWORD:
            return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode=require"
        else:
            return f"postgresql://{self.DB_USER}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode=require"
    
    def validate(self):
        """Valida se todas as configurações necessárias estão presentes"""
        required = ['DB_HOST', 'DB_USER']
        missing = [var for var in required if not getattr(self, var)]
        if missing:
            raise ValueError(f"Variáveis de ambiente obrigatórias faltando: {missing}")
        
        # Log da connection string (sem senha)
        safe_conn = self.get_connection_string().replace(self.DB_PASSWORD, '***') if self.DB_PASSWORD else self.get_connection_string()
        logger.info(f"📊 Connection string configurada: {safe_conn}")
        logger.info("✅ Todas as configurações validadas com sucesso")

settings = Settings()
settings.validate()

# ========================= MODELOS PYDANTIC =========================

class ClienteBase(BaseModel):
    """Modelo base para Cliente"""
    id: int
    nome: str
    telefone: Optional[str] = None
    
class ClienteResponse(ClienteBase):
    """Resposta de Cliente com timestamp"""
    created_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MetricasPeriodo(BaseModel):
    """Métricas de um período específico"""
    faturamento_total: float = Field(..., description="Faturamento total do período")
    total_pedidos: int = Field(..., description="Número total de pedidos")
    ticket_medio: float = Field(..., description="Valor médio dos pedidos")
    clientes_unicos: int = Field(..., description="Número de clientes únicos")
    
    @validator('faturamento_total', 'ticket_medio')
    def round_floats(cls, v):
        return round(v, 2)

class DashboardHoje(BaseModel):
    """Dashboard com comparativo diário"""
    hoje: MetricasPeriodo
    comparativo_ontem: MetricasPeriodo
    variacao_percentual: Optional[Dict[str, float]] = None

class ClienteRisco(BaseModel):
    """Cliente com risco de churn"""
    cliente_id: int
    nome: str
    telefone: Optional[str]
    total_pedidos: int
    valor_medio: float
    ultimo_pedido_data: datetime
    dias_sem_pedir: float
    freq_media_dias: float
    valor_ultimo_pedido: float
    risk_score: float
    risco_nivel: str = Field(..., description="Alto, Médio ou Baixo")
    
    @validator('risk_score')
    def categorize_risk(cls, v, values):
        if v > 1.5:
            values['risco_nivel'] = 'CRÍTICO'
        elif v > 1.0:
            values['risco_nivel'] = 'ALTO'
        elif v > 0.7:
            values['risco_nivel'] = 'MÉDIO'
        else:
            values['risco_nivel'] = 'BAIXO'
        return round(v, 2)

class ClienteLTV(BaseModel):
    """Cliente com Lifetime Value calculado"""
    cliente_id: int
    nome: str
    telefone: Optional[str]
    total_pedidos: int
    ticket_medio: float
    freq_mensal: float
    ltv: float
    categoria_valor: str = Field(..., description="VIP, Gold, Silver, Bronze")
    
    @validator('ltv')
    def categorize_ltv(cls, v, values):
        if v > 5000:
            values['categoria_valor'] = 'VIP'
        elif v > 2000:
            values['categoria_valor'] = 'GOLD'
        elif v > 1000:
            values['categoria_valor'] = 'SILVER'
        else:
            values['categoria_valor'] = 'BRONZE'
        return round(v, 2)

class HealthCheckResponse(BaseModel):
    """Resposta do health check"""
    status: str
    timestamp: datetime
    version: str
    database: str
    cache: str

# ========================= DATABASE =========================

class DatabasePool:
    """Gerenciador do pool de conexões assíncronas"""
    _instance = None
    _pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def init_pool(self):
        """Inicializa o pool de conexões"""
        if self._pool is None:
            try:
                # Usa a connection string construída
                connection_string = settings.get_connection_string()
                
                self._pool = await asyncpg.create_pool(
                    connection_string,
                    min_size=settings.DB_POOL_MIN_SIZE,
                    max_size=settings.DB_POOL_MAX_SIZE,
                    command_timeout=60
                )
                logger.info(f"✅ Pool de conexões criado: min={settings.DB_POOL_MIN_SIZE}, max={settings.DB_POOL_MAX_SIZE}")
            except Exception as e:
                logger.error(f"❌ Erro ao criar pool de conexões: {e}")
                raise
    
    async def close_pool(self):
        """Fecha o pool de conexões"""
        if self._pool:
            await self._pool.close()
            logger.info("Pool de conexões fechado")
    
    @asynccontextmanager
    async def acquire(self):
        """Context manager para adquirir conexão do pool"""
        async with self._pool.acquire() as connection:
            yield connection

db_pool = DatabasePool()

# ========================= CACHE =========================

class CacheManager:
    """Gerenciador de cache em memória"""
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Obtém valor do cache se ainda válido"""
        if key in self._cache:
            if datetime.now() - self._timestamps[key] < timedelta(seconds=settings.CACHE_TTL_SECONDS):
                logger.debug(f"Cache hit: {key}")
                return self._cache[key]
            else:
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        """Armazena valor no cache"""
        self._cache[key] = value
        self._timestamps[key] = datetime.now()
        logger.debug(f"Cache set: {key}")
    
    def clear(self):
        """Limpa todo o cache"""
        self._cache.clear()
        self._timestamps.clear()
        logger.info("Cache limpo")

cache = CacheManager()

# ========================= LIFESPAN MANAGER =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação"""
    # Startup
    logger.info("🚀 Iniciando aplicação...")
    await db_pool.init_pool()
    
    # Testa conexão
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        logger.info("✅ Conexão com banco de dados testada com sucesso")
    except Exception as e:
        logger.error(f"❌ Erro ao testar conexão: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Encerrando aplicação...")
    await db_pool.close_pool()
    cache.clear()

# ========================= APLICAÇÃO FASTAPI =========================

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================= DEPENDÊNCIAS =========================

async def get_db():
    """Dependência para obter conexão do banco"""
    async with db_pool.acquire() as conn:
        yield conn

def get_timezone():
    """Retorna o timezone configurado"""
    return pytz.timezone(settings.TIMEZONE)

# ========================= FUNÇÕES AUXILIARES =========================

async def calcular_metricas_periodo(
    conn: asyncpg.Connection,
    inicio_periodo: datetime,
    fim_periodo: datetime
) -> MetricasPeriodo:
    """Calcula métricas para um período específico"""
    query = """
        SELECT
            COALESCE(SUM(valor_total), 0)::float AS faturamento_total,
            COUNT(id) AS total_pedidos,
            COALESCE(AVG(valor_total), 0)::float AS ticket_medio,
            COUNT(DISTINCT cliente_id) AS clientes_unicos
        FROM "Pedidos_Unificados"
        WHERE created_at >= $1 AND created_at < $2;
    """
    
    result = await conn.fetchrow(query, inicio_periodo, fim_periodo)
    return MetricasPeriodo(**dict(result))

def calcular_variacao_percentual(atual: MetricasPeriodo, anterior: MetricasPeriodo) -> Dict[str, float]:
    """Calcula variação percentual entre dois períodos"""
    def calc_var(atual_val, anterior_val):
        if anterior_val == 0:
            return 100.0 if atual_val > 0 else 0.0
        return round(((atual_val - anterior_val) / anterior_val) * 100, 2)
    
    return {
        "faturamento": calc_var(atual.faturamento_total, anterior.faturamento_total),
        "pedidos": calc_var(atual.total_pedidos, anterior.total_pedidos),
        "ticket_medio": calc_var(atual.ticket_medio, anterior.ticket_medio),
        "clientes": calc_var(atual.clientes_unicos, anterior.clientes_unicos)
    }

# ========================= ENDPOINTS =========================

@app.get("/", response_model=HealthCheckResponse, tags=["Health"])
async def health_check(conn: asyncpg.Connection = Depends(get_db)):
    """Health check endpoint com status do sistema"""
    try:
        # Testa conexão com banco
        await conn.fetchval("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    return HealthCheckResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        timestamp=datetime.now(),
        version=settings.API_VERSION,
        database=db_status,
        cache="healthy"
    )

@app.get("/api/v1/clientes", response_model=List[ClienteResponse], tags=["Clientes"])
async def listar_clientes(
    limit: int = Query(10, ge=1, le=100, description="Número de clientes a retornar"),
    offset: int = Query(0, ge=0, description="Offset para paginação"),
    conn: asyncpg.Connection = Depends(get_db)
):
    """Lista clientes com paginação"""
    cache_key = f"clientes_{limit}_{offset}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    query = """
        SELECT id, nome, telefone, created_at 
        FROM "Clientes_Unificados" 
        ORDER BY created_at DESC 
        LIMIT $1 OFFSET $2;
    """
    
    try:
        rows = await conn.fetch(query, limit, offset)
        result = [ClienteResponse(**dict(row)) for row in rows]
        cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Erro ao buscar clientes: {e}")
        raise HTTPException(status_code=500, detail="Erro ao buscar clientes")

@app.get("/api/v1/dashboard/hoje", response_model=DashboardHoje, tags=["Dashboard"])
async def dashboard_hoje(
    conn: asyncpg.Connection = Depends(get_db),
    tz: pytz.timezone = Depends(get_timezone)
):
    """Dashboard com métricas do dia atual e comparativo com ontem"""
    cache_key = "dashboard_hoje"
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    agora = datetime.now(tz)
    inicio_hoje = agora.replace(hour=0, minute=0, second=0, microsecond=0)
    inicio_ontem = inicio_hoje - timedelta(days=1)
    
    try:
        metricas_hoje = await calcular_metricas_periodo(conn, inicio_hoje, agora)
        metricas_ontem = await calcular_metricas_periodo(conn, inicio_ontem, inicio_hoje)
        
        result = DashboardHoje(
            hoje=metricas_hoje,
            comparativo_ontem=metricas_ontem,
            variacao_percentual=calcular_variacao_percentual(metricas_hoje, metricas_ontem)
        )
        
        cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Erro ao calcular dashboard: {e}")
        raise HTTPException(status_code=500, detail="Erro ao calcular métricas do dashboard")

@app.get("/api/v1/dashboard/semana", response_model=MetricasPeriodo, tags=["Dashboard"])
async def dashboard_semana(
    conn: asyncpg.Connection = Depends(get_db),
    tz: pytz.timezone = Depends(get_timezone)
):
    """Métricas da semana atual"""
    cache_key = "dashboard_semana"
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    agora = datetime.now(tz)
    inicio_semana = agora - timedelta(days=agora.weekday())
    inicio_semana = inicio_semana.replace(hour=0, minute=0, second=0, microsecond=0)
    
    try:
        result = await calcular_metricas_periodo(conn, inicio_semana, agora)
        cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Erro ao calcular métricas da semana: {e}")
        raise HTTPException(status_code=500, detail="Erro ao calcular métricas da semana")

@app.get("/api/v1/dashboard/mes", response_model=MetricasPeriodo, tags=["Dashboard"])
async def dashboard_mes(
    conn: asyncpg.Connection = Depends(get_db),
    tz: pytz.timezone = Depends(get_timezone)
):
    """Métricas do mês atual"""
    cache_key = "dashboard_mes"
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    agora = datetime.now(tz)
    inicio_mes = agora.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    try:
        result = await calcular_metricas_periodo(conn, inicio_mes, agora)
        cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Erro ao calcular métricas do mês: {e}")
        raise HTTPException(status_code=500, detail="Erro ao calcular métricas do mês")

@app.get("/api/v1/insights/churn-prediction", response_model=List[ClienteRisco], tags=["Insights"])
async def prever_churn(
    risk_threshold: float = Query(0.7, ge=0, le=2, description="Threshold do score de risco"),
    limit: int = Query(50, ge=1, le=200, description="Número máximo de resultados"),
    conn: asyncpg.Connection = Depends(get_db)
):
    """
    Identifica clientes com alto risco de churn usando análise comportamental.
    
    O algoritmo considera:
    - Frequência de pedidos
    - Valor do último pedido vs média
    - Tempo desde o último pedido
    """
    cache_key = f"churn_{risk_threshold}_{limit}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    query = """
    WITH ranked_orders AS (
        SELECT 
            cliente_id, 
            valor_total, 
            created_at,
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
            MAX(CASE WHEN rn = 1 THEN valor_total ELSE NULL END) as valor_ultimo_pedido,
            STDDEV(valor_total) as desvio_padrao_valor
        FROM ranked_orders
        GROUP BY cliente_id
    ),
    final_risk AS (
        SELECT
            cs.cliente_id, 
            uc.nome, 
            uc.telefone, 
            cs.total_pedidos, 
            cs.valor_medio,
            cs.ultimo_pedido_data,
            EXTRACT(EPOCH FROM (NOW() - cs.ultimo_pedido_data)) / 86400 AS dias_sem_pedir,
            cs.freq_media_dias, 
            cs.valor_ultimo_pedido,
            cs.desvio_padrao_valor,
            -- Score melhorado com normalização
            LEAST(
                (EXTRACT(EPOCH FROM (NOW() - cs.ultimo_pedido_data)) / 86400 / GREATEST(cs.freq_media_dias, 1)) *
                (1 + ABS(cs.valor_ultimo_pedido - cs.valor_medio) / GREATEST(cs.valor_medio, 1)),
                3.0  -- Cap máximo do score
            ) AS risk_score
        FROM customer_stats cs
        JOIN "Clientes_Unificados" uc ON uc.id = cs.cliente_id
        WHERE cs.total_pedidos > 1 
          AND cs.freq_media_dias > 0 
          AND cs.valor_medio > 0
    )
    SELECT
        cliente_id, 
        nome, 
        telefone, 
        total_pedidos, 
        valor_medio::float, 
        ultimo_pedido_data,
        dias_sem_pedir::float, 
        freq_media_dias::float, 
        valor_ultimo_pedido::float, 
        risk_score::float
    FROM final_risk
    WHERE risk_score > $1
    ORDER BY risk_score DESC
    LIMIT $2;
    """
    
    try:
        rows = await conn.fetch(query, risk_threshold, limit)
        result = []
        for row in rows:
            cliente = ClienteRisco(**dict(row))
            result.append(cliente)
        
        cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Erro ao calcular churn prediction: {e}")
        raise HTTPException(status_code=500, detail="Erro ao calcular previsão de churn")

@app.get("/api/v1/insights/ltv", response_model=List[ClienteLTV], tags=["Insights"])
async def calcular_ltv(
    limit: int = Query(50, ge=1, le=200, description="Número de clientes no ranking"),
    min_pedidos: int = Query(2, ge=1, description="Mínimo de pedidos para considerar"),
    conn: asyncpg.Connection = Depends(get_db)
):
    """
    Calcula o Lifetime Value (LTV) dos clientes.
    
    Fórmula: LTV = (Ticket Médio × Frequência Mensal × Margem) / Churn Rate
    
    Retorna ranking dos clientes mais valiosos.
    """
    cache_key = f"ltv_{limit}_{min_pedidos}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    query = """
    WITH customer_lifetime_stats AS (
        SELECT
            cliente_id,
            COUNT(id) as total_pedidos,
            AVG(valor_total) as ticket_medio,
            MIN(created_at) as primeira_compra,
            MAX(created_at) as ultima_compra,
            NULLIF(EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) / 86400, 0) as lifetime_em_dias,
            SUM(valor_total) as receita_total
        FROM "Pedidos_Unificados"
        GROUP BY cliente_id
        HAVING COUNT(id) >= $1
    ),
    ltv_components AS (
        SELECT
            cls.cliente_id,
            uc.nome,
            uc.telefone,
            cls.ticket_medio,
            cls.total_pedidos,
            cls.receita_total,
            cls.lifetime_em_dias,
            -- Frequência mensal ajustada
            CASE 
                WHEN cls.lifetime_em_dias > 0 THEN (cls.total_pedidos / cls.lifetime_em_dias) * 30
                ELSE cls.total_pedidos  -- Se só tem 1 dia, usa total de pedidos
            END AS freq_mensal,
            -- Tempo como cliente em meses
            GREATEST(cls.lifetime_em_dias / 30, 1) as meses_cliente
        FROM customer_lifetime_stats cls
        JOIN "Clientes_Unificados" uc ON uc.id = cls.cliente_id
        WHERE cls.lifetime_em_dias IS NOT NULL
    )
    SELECT
        cliente_id,
        nome,
        telefone,
        total_pedidos,
        ticket_medio::float,
        freq_mensal::float,
        -- LTV com ajuste por tempo de vida do cliente
        ((ticket_medio * freq_mensal * $2) / $3 * 
         LEAST(meses_cliente / 3, 2))::float AS ltv  -- Bonus por fidelidade
    FROM ltv_components
    ORDER BY ltv DESC
    LIMIT $4;
    """
    
    try:
        rows = await conn.fetch(
            query, 
            min_pedidos,
            settings.MARGEM_LUCRO_PADRAO,
            settings.CHURN_RATE_MENSAL_PADRAO,
            limit
        )
        
        result = []
        for row in rows:
            cliente = ClienteLTV(**dict(row))
            result.append(cliente)
        
        cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Erro ao calcular LTV: {e}")
        raise HTTPException(status_code=500, detail="Erro ao calcular LTV dos clientes")

@app.post("/api/v1/cache/clear", tags=["Admin"])
async def limpar_cache():
    """Limpa o cache da aplicação (endpoint administrativo)"""
    cache.clear()
    return {"message": "Cache limpo com sucesso", "timestamp": datetime.now()}

@app.get("/api/v1/stats", tags=["Admin"])
async def estatisticas_sistema(conn: asyncpg.Connection = Depends(get_db)):
    """Retorna estatísticas do sistema"""
    stats_query = """
    SELECT 
        (SELECT COUNT(*) FROM "Clientes_Unificados") as total_clientes,
        (SELECT COUNT(*) FROM "Pedidos_Unificados") as total_pedidos,
        (SELECT SUM(valor_total) FROM "Pedidos_Unificados") as faturamento_total,
        (SELECT COUNT(DISTINCT cliente_id) FROM "Pedidos_Unificados" 
         WHERE created_at > NOW() - INTERVAL '30 days') as clientes_ativos_30d
    """
    
    try:
        result = await conn.fetchrow(stats_query)
        return {
            "total_clientes": result['total_clientes'],
            "total_pedidos": result['total_pedidos'],
            "faturamento_total": float(result['faturamento_total'] or 0),
            "clientes_ativos_30d": result['clientes_ativos_30d'],
            "pool_size": db_pool._pool.get_size() if db_pool._pool else 0,
            "cache_items": len(cache._cache)
        }
    except Exception as e:
        logger.error(f"Erro ao buscar estatísticas: {e}")
        raise HTTPException(status_code=500, detail="Erro ao buscar estatísticas")

# ========================= EXCEPTION HANDLERS =========================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handler customizado para HTTPException"""
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handler para exceções não tratadas"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# ========================= MAIN =========================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Desabilite em produção
        log_level="info"
    )
