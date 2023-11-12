#Equipe Pandora
- Iago Magalhães
- Vanessa Carvalho

## Introdução


## Objetivos
### Objetivo Geral
- Analisar o uso de redes convolucionais no auxílio de leitura de cartões respostas;
### Objtivos especificos
- Criar um detector de respostas com Deep Learning;
- Analisar desempenho de arquiteturas de CNN;
- Desenvolver uma API para analisar imagens de provas diversas.

## Materiais e Metódos
### Dataset
O dataset utilizado para o treino das CNN's foi disponibilizado pela equipe organizadora do Hackathon. Para fins da utilização, tratamento e treino das redes, foi optado por realizar o download dos arquivos no formato 'Pascal VOC XML'. Ao todo são 209 imagens divididas em três conjuntos, sendo treino, teste e validação.

Após obter o banco de dados foi feito o recorte das imagens, selecionado como a região de interesse (ROI) apenas à área já demacarda pela equipe do Hackathon. Logo, as imagens de entrada da rede utilizadas para o desenvolvimento do classificador foram apenas os recortes das regiões em que o usuário/aluno teria marcado na prova.

### Pré-processamento
O pré-processamento é uma das etapas fundamentais da Visão Computacional que tem como objetivo melhorar a imagem em aspectos relevantes em função do que será feito com ela, seja corrigindo defeitos oriundos da aquisição ou adaptando para o problema. Neste trabalho foram aplicados 10 técnicas de pré-processamento para realizar o agusamento/melhoria da imagem que seria utilizada no treinamento das redes convolucionais. Sendo eles:
1. Dilatação
2. Filtro Laplaciano
3. Filtro Gaussiano
4. Filtro de Média
5. Filtro de Mediana
6. Filtro de Mediana + Binarização
7. Filtro Gaussiano + Binarização
8. Filtro de Mediana + Filtro Gaussiano
9. Filtro de Mediana + Filtro Sobel
10. Filtro Gaussiano + Canny

### Métricas de avaliação
Para avaliar os resultados obtidos pelas redes convolucionais, são utilizadas métricas estatísticas comumente utilizadas pela comunidade. Neste trabalho as métricas de acurácia e precisão foram utilizadas com fator de seleção do melhor modelo de classificação.

Após plotar a matriz de confusão obtemos os valores de verdadeiros positivos e negativos e os falsos postivos e negativos. Com esses valores é possível calcular à acurácia e precisão através das seguintes equações:

![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/8bbe0f13-2709-4e41-8449-8a96859654c1)


### Redes utilizadas

## Resultados

## Conclusão

## Referências
