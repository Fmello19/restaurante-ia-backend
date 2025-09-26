# 1. A Base: Começamos com uma imagem oficial do Python.
FROM python:3.10-slim

# 2. O Local de Trabalho: Criamos uma pasta dentro da cozinha para o Chef organizar as coisas.
WORKDIR /app

# 3. A Lista de Compras: Copiamos a lista de ingredientes para dentro da cozinha.
COPY requirements.txt .

# 4. Instalando os Ingredientes: Mandamos o Python instalar tudo da lista.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Trazendo as Receitas: Copiamos o resto dos nossos arquivos (o main.py) para a cozinha.
COPY . .

# 6. A Ordem Final: Dizemos ao Chef o que ele deve fazer quando a cozinha abrir.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]