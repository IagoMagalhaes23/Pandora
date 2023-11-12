# Equipe Pandora
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
Para se desenvolver o classificador, optamos por utilizar redes conhecidas pela comunidade, visando a facilidade de implementação, rapidez no desenvolvimento e utilizando seus resultados anteriores como base para o desenvolvimento de uma solução para o problema.

As redes utilizadas foram:
- AlexNet
- EfficientNetB0
- InceptionV3
- LeNet

## Metodologia
A meotodologia abordada foi baseada em trabalhos acadêmicos sobre treino de redes convolucionais. O primeiro passo foi organizar o banco de dados em treino, teste e validação visando problemas clássicos como sobreajuste e subajuste. Além disso, foi optado por realizar o pré-processamento das imagens para realizar o agusamento de características da imagem. Logo depois, foi realizado o treinamento das quatro redes CNN's, seguindo os mesmos padrões, tais como, 1000 epócas de treinamento e otimizador Adam.
Para seleção do melhor modelo foi levado em conta as métricas de acurácia e precisão.

## Resultados
Os resultados obtidos para cada rede podem ser visualizados a seguir. Vale ressaltar que entre os 10 pré-processamentos utilizados, o melhor em todos os modelos foi 'Filtro Gaussiano + Binarização'.

- AlexNet
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/d77bc69b-21b6-4d19-b69a-0af213ab1fc1)
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/3dcd9574-a717-48fd-9240-46a338591888)
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/62b95f74-d941-44fa-84c6-bb7c713feda8)
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/815f0c2e-b3c9-48bb-b57e-cee770c62066)
  Acurácia: 0.65%
  Recall: 0.6633333333333333%
  Precisão: 0.7%
  
- EfficientNetB0
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/f02ddef0-5ef8-4dd8-9840-033de2330b00)
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/f5e9cb58-73d6-4731-bd36-3bd05708280a)
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/50d173d7-8ad7-4982-a653-8931b60f3617)
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/fc356d7a-838f-4957-b883-a4e0f930c663)
  Acurácia: 0.9%
  Recall: 0.9099999999999999%
  Precisão: 0.9333333333333332%

-InceptionV3
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/ab821541-96dd-48ef-a3ba-f6f27ff28917)
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/69fb17cc-7f0c-4f49-a93d-17984a997c7a)
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/9e46f4e0-3d56-4311-92e2-9ae51d03a5b3)
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/b0525a88-e412-4474-8c99-f96d5d4cf15e)
  Acurácia: 0.2%
  Recall: 0.2%
  Precisão: 0.04%

- LeNet
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/e2b3a6e5-6eca-4f3b-b6ce-0fb63d692220)
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/0917cd7a-30b8-4c4f-81c5-c48a4cbae889)
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/a1fea653-af16-4da1-a3dc-6cb3cd73fd2a)
  ![image](https://github.com/IagoMagalhaes23/Pandora/assets/65053026/7048c224-1145-4abf-87ef-9a0413666972)
  Acurácia: 0.55%
  Recall: 0.5433333333333332%
  Precisão: 0.5599999999999999%

## Conclusão

## Referências
