#!/bin/bash
# Setup do ambiente virtual para o projeto

echo "Configurando ambiente virtual..."

# Criar venv
python -m venv .venv

# Ativar venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

echo "✅ Ambiente virtual configurado!"
echo ""
echo "Para ativar o venv:"
echo "  source .venv/bin/activate"
echo ""
echo "Para executar o projeto:"
echo "  python app.py"
