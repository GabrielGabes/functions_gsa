# functions_gsa

Este repositório foi criado com o objetivo de facilitar o meu dia a dia, centralizando funções que utilizo frequentemente em Python para automações e análises. Acredito que essas funções também podem ser úteis para outros usuários, por isso decidi tornar o repositório público.

## Como Usar

Você pode carregar e executar qualquer arquivo Python deste repositório diretamente no seu código, sem precisar clonar o repositório. Veja o exemplo abaixo de como fazer isso usando o módulo `requests`:

### Exemplo de Uso

```python
import requests

# Nome do arquivo Python que você deseja carregar (sem a extensão .py)
pasta_escolhida = 'ML_supervised_learning'

# URL para o arquivo raw no GitHub
url = "https://raw.githubusercontent.com/GabrielGabes/functions_gsa/main/" + pasta_escolhida + ".py"

# Baixando o conteúdo do arquivo
response = requests.get(url)
code = response.text

# Executando o código baixado
exec(code)
```