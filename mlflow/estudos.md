# Vantagens do MLFlow:

- Ter um caminho dos parâmetros e resultados de diferentes reproduções.
- Código reusável para compartilhar com outros cientistas de dados.
- Deploy de um modelo de Machile Learning.
- Muito flexível e pouco rígido (feito para funcionar com qualquer etapa e qualquer modelo).

# Instalação:

```powershell
pip install mlflow
```

# Exemplos Tutorial:

- [mlflow/examples at master · mlflow/mlflow (github.com)](https://github.com/mlflow/mlflow/tree/master/examples)
- Com FastAi: [mlflow/examples/fastai at master · mlflow/mlflow (github.com)](https://github.com/mlflow/mlflow/tree/master/examples/fastai)

# Componentes MLFlow:

1. **Tracking**: log de parâmetros, versões, métricas e artefatos para a visualização de resultados depois.
2. **Projetos**: empacotar códigos reutilizáveis. Simplesmente um código ou repositório do git.
3. **Modelos**: convenção para empacotar modelos e deploy eles. Há diversas maneiras, uma delas é agir como uma função do python.
4. **Registro**: armazena todo o ciclo de vida de um modelo.

É possível rodar diferentes runs com diferentes parâmetros (Projetos) e ter o track (Tracking) de cada um deles. → Diferentes usuários trabalham em um mesmo problema e podem ter seus trabalhos salvos.

# Tracking:

- Cada run salva as seguintes informações:
    - Versão do código.
    - Tempo de início e término.
    - Fonte (arquivo que rodou).
    - Parâmetros (key e value).
    - Métricas (key e value).
    - Artefatos: outputs (imagens, modelos, dados).
- A princípio salva em uma pasta mlruns localmente → acessa com ‘mlflow ui’.
- Log de Parâmetros, Métricas e Artefato:

```python
## Log manual

log_param("param1", randint(0, 100))
log_metric("foo", random())
log_artifact()

## Log automático
mlflow.package.autlog() # package exemplo: sklearn
```

- Log_metric pode receber ‘step’ como parâmetro, sendo a etapa de um mesmo algoritmo ‘key’, mas diferente valor.
- Logar em um tracking server:

```python
import mlflow
remote_server_uri = "..." # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)

mlflow.set_experiment("/my-experiment")
with mlflow.start_run():
    mlflow.log_param("a", 1)
    mlflow.log_metric("b", 2)
```

# Projetos:

- Nome
- Ponto de Entrada (arquivo .py): comandos que serão rodados no projeto e informações sobre seus parâmetros.
- Ambiente (bibliotecas).
- Rodar projeto (pode passar parâmetros):

```python
mlflow run projeto # default: conda, caso queira ambiente da máquina: --no-conda -> powershell

mlflow.projects.run # Pode usar para rodar em paralelo e comparar diferentes resultados

# Diretório
# Repositório Git
# Commit hash ou nome da branch
# Entry Point -> src/file.py
```

- Pode-se adicionar um arquivo YAML como MLproject para explicar melhor e ter mais controle sobre o projeto a ser executado:
    - Entry Point.
    - Parâmetros (tipo e default value).

# Modelos:

- Empacotar modelos de aprendizagem de máquina para serem usados.
- Trabalha com ‘flavors’: diferentes formas de lidar com o deploy desse modelo. Eles são descritos  dentro de um arquivo MLmodel YAML.
- É importante a especificação do conda.YAML ou do virtualenv + requirements.txt para que o modelo instale todas as dependências.
- **Model Signature**: descrição dos inputs (name e type) e outputs (type) do modelo. Formato JSON.

```python
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_schema = Schema([
  ColSpec("double", "sepal length (cm)"),
  ColSpec("double", "sepal width (cm)"),
  ColSpec("double", "petal length (cm)"),
  ColSpec("double", "petal width (cm)"),
])
output_schema = Schema([ColSpec("long")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# log_model(classifier, "nome", signature=signature)
```

- **Input Example**: exemplo de entrada que deve ser no formato JSON e adicionados no ‘log_model’ (pode ser pandasDF ou Dict também):

```python
input_example = {
  "sepal length (cm)": 5.1,
  "sepal width (cm)": 3.5,
  "petal length (cm)": 1.4,
  "petal width (cm)": 0.2
}
mlflow.sklearn.log_model(..., input_example=input_example)
```

- **Flavors**: tipos de armazenamento/exportação para o modelo (save_model e log_model):
    - Python function
    - Sklearn
    - TensorFlow
    - FastAi
- **Evaluation**: ‘mlflow.evaluate()’:

```python
with mlflow.start_run() as run:
    model_info = mlflow.sklearn.log_model(model, "model")
    result = mlflow.evaluate(
        model_info.model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        dataset_name="adult",
        evaluators=["default"], # custom_metrics: função para outros tipos de métricas
    )
```

# Armazenamento:

- Modelo centralizado para gerenciar o ciclo de um modelo no MLFlow.
    - Modelo vem de um experimento ou run que é salvo com log_model com FLAVOR: só assim para ser salvo/criado modelo.
    - Tem como encontrar diferentes versões para um mesmo modelo
- Adicionar uma run específica no storage:

```python
result = mlflow.register_model(
    "runs:/d16076a3ec534311817565e6527539c0/sklearn-model",
    "sk-learn-random-forest-reg"
)
```

- Para carregar um modelo já salvo:

```python
import mlflow.pyfunc

model_name = "sk-learn-random-forest-reg-model"
model_version = 1 # Pode ser stage também

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}" 
)

model.predict(data)
```

- Adicionar descrição ou renomear ao modelo:

```python
client = MlflowClient()
client.update_model_version( 
    name="sk-learn-random-forest-reg-model",
    version=1,
    description="This model version is a scikit-learn random forest containing 100 decision trees"
)
client.rename_registered_model(
    name="sk-learn-random-forest-reg-model",
    new_name="sk-learn-random-forest-reg-model-100"
)
```

- Printar todos os modelos salvos:

```python
client = MlflowClient()
for rm in client.list_registered_models(): #search_model_versions para as versões de um modelo
    pprint(dict(rm), indent=4)
```

- Deletar um modelo:
```python
client.delete_registered_model(name="sk-learn-random-forest-reg-model")
```