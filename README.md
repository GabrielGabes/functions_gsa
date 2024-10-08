# functions_gsa

Este repositório foi criado com o objetivo de facilitar o meu dia a dia, centralizando funções que utilizo frequentemente em Python para automações e análises. Acredito que essas funções também podem ser úteis para outros usuários, por isso decidi tornar o repositório público.

## Como Usar

Você pode carregar e executar qualquer arquivo Python deste repositório diretamente no seu código, sem precisar clonar o repositório. Veja o exemplo abaixo de como fazer isso usando o módulo `requests`:

### Exemplo de Uso

```python
import requests

# Nome do arquivo Python que você deseja carregar (sem a extensão .py)
arquivo = 'ML_supervised_learning'

# URL para o arquivo raw no GitHub
url = "https://raw.githubusercontent.com/GabrielGabes/functions_gsa/main/" + arquivo + ".py"

# Baixando o conteúdo do arquivo
response = requests.get(url)
code = response.text

# Executando o código baixado
exec(code)
```

Após carregar o código, você poderá utilizar as funções diretamente em seus scripts, desde que conheça previamente sua finalidade e os parâmetros necessários para sua execução.

Espero que essas funções contribuam para agilizar seus projetos de automação e análise. Sinta-se à vontade para explorar, adaptar, e expandir conforme suas necessidades. Se tiver alguma sugestão ou melhoria, sua contribuição será muito bem-vinda.

Caso queira usar mais de um arquivo, para que não seja desgaste ficar copiando, colando e modificando o código acima varias vezes segue uma função facilitadora:

```python
import requests

def carregando_funcoes(arquivo):
    url = "https://raw.githubusercontent.com/GabrielGabes/functions_gsa/main/" + arquivo + ".py"
    try:
        response = requests.get(url)
        code = response.text
        exec(code)
        return f'{arquivo} carregado com sucesso!'
    except Exception as e:
        print("Error:", e)

# Exemplo de uso:
carregando_funcoes('ML_supervised_learning')
carregando_funcoes('ML_base_titanic')
```

### Contact
For more information, please contact [gabriel_s_anjos@yahoo.com][(https://www.linkedin.com/feed/)].