# Pré-processamento de Dados

[Jupyter Notebook com Códigos Práticos](https://gitlab.com/victorleaoo/pre-processing-studies/-/blob/main/preprocessing_study.ipynb)

## Visão Geral de Linguagem Natural (PNL)

### Conceitos Iniciais

A partir da mineração de dados, se é obtido uma grande quantidade de documentos e o objetivo é **filtrar/extrair** aqueles que serão **úteis**.<br />
Os textos crus são dados não estruturados. Dessa forma, deve-se buscar uma forma de deixá-los estruturados para análises dos tipos:
- **Classificação:** Sobre o que o texto é. Feito por supervisão, assim, entrega-se documentos de treino com classes pré-definidas.
- **Clusterização:** Fazer uma conexão entre assuntos. Não supervisionado.
Entretanto, antes de partir diretamente para a análise, alguns passos são importantes: **entendimento do negócio (problema)**, **entendimento dos dados** e, assim, o **pré-processamento** deles é realizado.<br />
De forma generalizada e prática para **classificação**, o **Corpus** é um conjunto de pares de documentos e suas classes, que após serem pré-processados, são treinados e, assim, é criado um modelo para a máquina realizar seu trabalho.

### Pré-Processamento

O processo de **limpeza (cleaning)** e **padronização (standardization)** ajuda muito no aumento de **acurácia** de modelos.<br />
Inicialmente, é possível extrair dados de treino por: ```(dataset='train')``` <br />
Além disso, para acessar um único documento: ```dataset.data[num]``` <br />

#### 1 - Tokenizar (Biblioteca nltk)

A partir de uma regra, o texto é **quebrado/divido** em **Tokens** que podem ser sentenças ou palavras.<br />

##### 1.1 - Sentence Tokenization (Tokenizar Sentenças)

**Segmentar** o texto em sentenças com significados úteis.<br />

1. 1.1.1 - Sent Tokenize:<br />
A função ```st = nltk.sent_tokenize``` é a mais **geral** para tokenização e leva em conta, basicamente, pontuação. Para aplicar: ```ss = st(text=texto, language='lang')```. Sendo ```ss``` os **tokens gerados**.<br />
Com o parâmetro **language**, usa-se o modelo pré-treinado do **nltk**, mas também é possível algoritmos próprios com ```nltk.data.load(resource_url='url')```.<br />

1. 1.1.2 - RegexpTokenizer:<br />
É possível criar um **padrão de tokenização com Regex**: ```st = nltk.tokenize.RegexpTokenizer(pattern=r’Regex’, gaps=True)```. Assim, ```ss = st.tokenize(texto)```, **ss** retorna os tokens com o regex definido como regra.<br />
**Regex para pontuação**: ``` SENTENCE_TOKENS_PATTERN = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s' ```

##### 1.2 - Word Tokenization (Tokenizar Palavras)

**Segmentar** o texto em palavras, sendo mais usado quando é importante com operações de **stemming e lemmatization**.

1. 1.2.1 - Word Tokenize:<br />
A função ```wt = nltk.word_tokenize```, usada da seguinte forma: ```w = wt(sentença)```, separa a sentença em **palavras** de forma geral.<br />
De forma geral, segue algumas regras bem similares à função ```nltk.TreebankWordTokenizer()```, já que usam o mesmo **mecanismo de tokenização**.

1. 1.2.2 - RegexpTokenizer: <br />
É possível criar um **padrão de tokenização com Regex**: ```st = nltk.RegexpTokenizer(pattern=r’Regex’, gaps=False)```. Assim, ```ss = st.tokenize(texto)```, **ss** retorna os tokens com o regex definido como regra.<br />
**Regex para palavras**: ``` TOKEN_PATTERN = r'\w+' ```. Palavras em geral, separa "don", "t".<br />
**Regex para palavras com gap**: ``` GAP_PATTERN = r'\s+' ```. Palavras completas, *gaps=True*. Não separa "don't".

1. 1.2.3 - Outras Funções (Punctuation e White Space): <br />
**Punctuation - Separa pontuação**: ``` wt = nltk.WordPunctTokenizer() ```. Separa da seguinte forma: "don", "'", "t".<br />
**WhiteSpace - Separa Espaço**: ``` wt = nltk.WhitespaceTokenizer()```. Separa espaços.

#### 2 - Normalização (Biblioteca re)

No processo de pré-processamento, **remover caracteres e pontuação** não úteis pode ser uma boa estratégia, uma vez que os textos ficam limpos e padronizados para serem usados como entradas em diversos algoritmos.<br />

##### 2.1 Cleaning Text

**Remover tokens e caracteres inúteis**. Algumas vezes, **tags** de extração (HTML, json) precisam ser retiradas, uma função de exemplo do nltk: ```clean_html()```.<br />
A **tokenização** pode acontecer antes ou depois, depende do que se está resolvendo. 

1. 2.1.1 - Removendo Caracteres Especiais:<br />
Normalmente, **caracteres especiais (simbolos e pontuação)** não fazem tanta diferença nos significados finais. Podem ser feitas **depois da tokenização:**<br />
``` pattern = re.compile('[{}]'.format(string.punctuation)) ```: O que vai ser retirado, sendo ```string.punctuation``` todos os caracteres de pontuação.<br />
``` filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens]) ```: Filtra os tokens.<br />
Também pode-se substituir da seguinte forma:<br />
```pattern.sub(' ', dataset.data[num])```: Na função de sub, se trocará todas as pontuações por ' '.<br />
Para fazer a limpeza **antes da tokenização**, deve-se aplicar um padrão regex para realizara a limpeza.<br />

*Uma dica:* Caso queira **remover acento**, só passar o texto para **ASCII**. 

1. 2.1.2 - Expandindo contrações:<br />
Em inglês é muito comum as **contrações**. Assim, pode ser interessante **expandir as contrações** para usar as palavras corretas.

1. 2.1.3 - Case Conversion:<br />
Pode-se converter um texto todo para **letras minúsculas** (```text.lower()```) ou **maiúsculas** (```text.upper()```).

#### 2.2 - Stop Words (Biblioteca nltk)

Algumas **palavras são insignificantes** e podem ser retiradas de um texto em ordem a ter um texto mais compacto e significativo.<br />
A biblioteca nltk tem uma lista com **stopwords:**  ```nltk.corpus.stopwords.words(language)``` <br />
Então, é feita uma função para remover as stopwords dos tokens e retornar esses tokens filtrados: <br />
``` filtered_tokens = [ token for token in tokens if token not in stop_words ] ```<br />
Algumas stopwords, como as **palavras de negação**, são retiradas, mas podem ser importantes. Então, fique atento!

#### 2.3 - Correção ortográfica

Os erros ortográficos podem ser aqueles relacionados a **spelling mistakes** ou **letras repetitidas** (finalmmente ao invés de finalmente).<br />
Pode ser importante realizar a correção ortográfica de um texto quando ele for **curto e poucas palavras podem fazer muitas mudanças** na análise.<br />

1. 2.3.1 - Removendo Repetição de Caracteres:<br />
Pode ser feito um algoritmo de **remoção de letras repetidas e comparação com um banco de dados de palavras** para que, quando a palavra seja encontrada, o algoritmo pare e retorne a palavra correta.

1. 2.3.2 - Corrigir Escritas Erradas:<br />
Humanos ou corretores podem **escrever palavras de forma errônea** e, portanto, é importante corrigí-las para uma melhor padronização do texto.<br />
A forma de correção que será usada é aquela em que se procurará **palavras parecidas/próximas** e substituirá. Dessa forma, é necessário usar **um banco de palavras candidatas** para comparar e encontrar palavras próximas: **Cálculo da distância de edição**.<br />

*Alguns pontos a serem considerados após usar uma função de **cálculo da distância de edição***: Caso sejam dadas várias palavras de substituição, procura-se a com **menor diferença da errada**. Além disso, caso haja um empate nesse quesito, busca-se a com **maior frequência no banco de palavras**.

#### 2.4 - Stemming (Biblioteca nltk)

**Redução/corte de palavras parecidas** (verbos, por exemplo), a partir do radical delas.<br />
Há diversos **algoritmos** já prontos (sendo o **Porter Stemmer** o mais famoso) para fazer stemming. Mas também é possível criar o próprio algoritmo com **RegexpStemmer**.<br />
Com isso, a palavra "perde" o significado caso queira voltar, mas como o objetivo é fazer um modelo de Machine Learning que vai entregar a classe, isso não importa muito.

#### 2.5 - Lemmatization

Stemming melhor trabalhado (também mais lento), deixando as palavras com **melhor significado visível**. Basicamente, ao invés de pegar o radical em si, pega a **palavra raiz** (sempre presente no dicionário).<br />
Basicamente, a melhor forma de Lemmatização é usando um **banco de palavras** que serão buscadas e retornar a **lemma** da palavra.

#### 2.6 - Biblioteca Spacy

Carrega os dados **pré-treinados** (importante). <br />
A partir de um documento: ```doc = sp(texto)```, o Spacy já faz a **divisão em tokens** e se é possível analisar diversas características desses tokens, como: <br />
Também há outras funções interessantes, como:<br />
```displacy.render[doc, style='dep']```: mostra a **dependência das palavras** no texto a partir de suas classes gramaticais.<br />
```doc1.similarity(doc2)```: a partir da distância entre os vetores de cada palavra, calcula a **similaridade** entre dois documentos.<br />

*Uma dica:* a busca de **entidade** do displacy **não** é tão eficaz.

#### 2.7 - Word2Vec (Biblioteca gensim)

Treinar palavras como vetores, bom para, por exemplo, buscar a **similaridade entre palavras**.<br />
```gensim.model.Word2Vec(articles, min_count=10)```: min_count para encontrar palavras próximas.<br />
A partir de um modelo com palavras chaves (exemplo: ```d = gensim.models.KeyedVectors.load_word2vec_format('doc.txt')```), podemos usar **funções para encontrar similaridades:**<br />
```d.wv.similarity('palavra1', 'palavra2')```: Similaridade entre 2 palavras.<br />
```d.wv.most_similar_to_given('palavra1', ['palavra2', 'palavra3'])```: Mostra a palavra na lista mais próxima à palavra 1.

### Conceitos em Machine Learning

1. **Preparação de Dados:** pré-processamento de dados antes de extrair os importantes.
1. **Extração de Features:** extrair features importantes dos dados para serem treinados.
1. **Features:** atributos importantes dos dados.
1. **Separar Dados:** separa dados em treino (treina o modelo) e teste (vê efetividade do modelo).
1. **Modelo:** combinação de dados/features para criação do modelo de machine learning.
1. **Acurácia:** quão bom o modelo é.

## Scikit-Learn

### Conceitos Iniciais

#### Estimator: Fit

O processo de **fit** é necessário para ajustar/validar os dados de entrada/atributos e calcular algumas métricas antes do **transform**. Um dos trabalhos do **fit** também é interpretar os **parâmetros** passados.<br />
Normalmente, é feito da seguinte forma: ```model.fit(X, Y)```.<br />
Quando um fit é chamado, qualquer outro **fit antes dele é ignorado** e sobrescrito (a não ser que warm_start=True).

#### Transformer: Transform

Já o **transform**, recebe o **dado bruto X** e retorna ele **modificado X'**. Só pode ser realizado **depois do fit**.<br />
Pode ser chamado sozinho: ```model.transform(X)```.<br />
Ou pode ser feito o **fit junto com o transform** (em geral, mais eficiente): ```model.fit_transform(X)```.

### Desenvolvendo estimators

[Referência](https://scikit-learn.org/stable/developers/develop.html)

#### Estimator

Responsável pelo **fit** dos dados.<br />
Os **parâmetros** definidos no ```__init__``` devem ser **constantes com valores de default** e não os dados de treino. Dessa forma, o **__init__** deve ser da seguinte forma (não se pode alterar os parâmetros, só atribuir):<br />
```def __init__(self, param1=1, param2=2):```<br />
```    self.param1 = param1```<br />
```    self.param2 = param2```<br /><br />

Na criação de um **Template** próprio, deve-se criar uma classe que tem como **herança BaseEstimator e os Mixin**. Exemplo: ```class TemplateClassifier(BaseEstimator, ClassifierMixin):```. Assim, dentro dessa classe há o **__init__(self, params)**, **fit(self, X, Y)** e o que for adicionado de Mixin (seta parâmetros automaticamente).<br />
**TransformerMixin**: traz como herança o método **fit_transform**.

### Pipeline

Ao invés de fazer cada etapa de **fit e transform** linha por linha, é possível criar um **Pipeline** que executa **de uma vez** todas as **funções das classes *transformers*** (sendo necessária um estimator final para realizar o fit) presentes nele.<br />
O exemplo de criação e execução de um pipeline é a seguinte:<br />
```pipe = Pipeline([```<br />
```       ('clean', Cleaner()),```<br />
```       ('tfidf', TfidfTransformer())```<br />
```])```<br />
```dado_t = pipe.fit_transform(dado_bruto)```<br /><br />

Não é o ponto ainda, mas o Pipeline pode ser usado para **algoritmos de machine learning (predict)**, que, juntamente com paramêtros, pode ser usado em um **GridSearchCV**.