# Metaflow

## Vantagens
1. Organização do código em etapas, o que ajuda a encontrar erros de forma mais específica.
2. É possível executar o código a partir da etapa onde uma possível falha apareça, assim, não é necessário rodar o algoritmo desde o início.
3. Criação de um versionamento para cada execução, tornando acessível o encontro e comparação de diferentes resultados no algoritmo.
4. É possível criar cards (como se fossem notebooks) para analisar resultados obtidos em um flow, o que ajuda na visualização e entendimento do que foi desenvolvido.

## Desvantagens
1. Não é possível a execução da biblioteca em Windows (No module named 'fcntl’). Sendo assim, é necessária a execução de código em outro SO. No meu caso, usei o WSL.