# Importa o FastAPI, nosso principal utensílio de cozinha.
from fastapi import FastAPI
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Carrega as variáveis de ambiente (segredos como a senha do banco)
load_dotenv()

# Cria a aplicação principal, o nosso "Chef"
app = FastAPI()

# Função para conectar ao banco de dados (a nossa "despensa")
def get_db_connection():
    try:
        # A MUDANÇA ESTÁ AQUI: vamos usar a string de conexão completa com a porta 6543
        conn_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:6543/{os.getenv('DB_NAME')}"
        conn = psycopg2.connect(conn_string)
        return conn
    except Exception as e:
        # Adicionamos um print para vermos o erro nos logs do EasyPanel
        print(f"ERRO DE CONEXÃO PELA PORTA 6543: {e}")
        raise e # Lança a exceção para que o FastAPI mostre o erro corretamente

# A primeira receita: um endpoint de teste.
# Quando alguém for na URL principal "/", esta função é executada.
@app.get("/")
def read_root():
    """
    Endpoint inicial para verificar se o Chef (API) está online.
    """
    return {"status": "Chef está na cozinha e pronto para trabalhar!"}

# Segunda receita: Listar os clientes.
# Quando alguém for na URL "/clientes", esta função é executada.
@app.get("/clientes")
def get_clientes():
    """
    Busca e retorna a lista de clientes da tabela Clientes_Unificados.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor) # Retorna como dicionário
        cursor.execute("SELECT id, nome, telefone, created_at FROM \"Clientes_Unificados\" ORDER BY created_at DESC LIMIT 10;")
        clientes = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"clientes": clientes}
    except Exception as e:
        # Se algo der errado, retorna uma mensagem de erro clara.

        return {"erro": f"Não foi possível conectar ou buscar no banco de dados: {e}"}

