################################################################################################
#
# MODELAGEM PREDITIVA - MBA Business Analytics e Big Data
#
# CASE AULA 3 - INVESTIMENT SUBSCRIBING
#
################################################################################################
# LENDO OS DADOS
path <- "D:/AULAS MBA FGV/2. MODELAGEM PREDITIVA/1. Turma 1 - Berrini [1o. sem 19]/AULA 3/"

base <- read.csv(paste(path,"investiment_subscribe.csv",sep=""), 
                 sep=",",header = T,stringsAsFactors = T)[,-1] # deletando a primeira coluna

################################################################################################
# ANALISANDO AS VARI?VEIS DA BASE DE DADOS
# ANALISE UNIVARIADA
summary(base)

library("VIM")
matrixplot(base)
aggr(base)

# n?o encontramos aqui vari?veis com valor NA, parece que a base j? foi tratada com 
# Others ou Unknown
# estrat?gia adotada

# ANALISE BIVARIADA
# Vari?veis quantitativas 
boxplot(base$x1_age            ~ base$y_subscribe)
boxplot(base$x11_duration      ~ base$y_subscribe)
boxplot(base$x12_campaign      ~ base$y_subscribe)
boxplot(base$x13_pdays         ~ base$y_subscribe)
boxplot(base$x16_empvar.rate   ~ base$y_subscribe)
boxplot(base$x17_consprice.idx ~ base$y_subscribe)
boxplot(base$x18_consconf.idx  ~ base$y_subscribe)
boxplot(base$x19_euribor3m     ~ base$y_subscribe)
boxplot(base$x20_nr.employed   ~ base$y_subscribe)

#Vari?veis quantitativas e quali
prop.table(table(base$y_subscribe))
prop.table(table(base$x2_job,       base$y_subscribe),1)
prop.table(table(base$x3_marital,   base$y_subscribe),1)
prop.table(table(base$x4_education, base$y_subscribe),1)
prop.table(table(base$x5_default,   base$y_subscribe),1)
prop.table(table(base$x6_housing,   base$y_subscribe),1)
prop.table(table(base$x7_loan,      base$y_subscribe),1)
prop.table(table(base$x8_contact,   base$y_subscribe),1)
prop.table(table(base$x9_month,     base$y_subscribe),1)
prop.table(table(base$x10_weekday,  base$y_subscribe),1)
prop.table(table(base$x14_previous, base$y_subscribe),1)
prop.table(table(base$x15_poutcome, base$y_subscribe),1)

################################################################################################
# AMOSTRAGEM DO DADOS
library(caret)

set.seed(12345)
index <- createDataPartition(base$y_subscribe, p= 0.7,list = F)

data.train <- base[index, ] # base de desenvolvimento: 70%
data.test  <- base[-index,] # base de teste: 30%

# Checando se as propor??es das amostras s?o pr?ximas ? base original
prop.table(table(base$y_subscribe))
prop.table(table(data.train$y_subscribe))
prop.table(table(data.test$y_subscribe))

################################################################################################
# MODELAGEM DOS DADOS - REGRESS?O LOGISTICA

# Avaliando multicolinearidade - vars quantitativas
library(mctest)
vars.quant <- data.train[,c(1,11,12,13,14,16,17,18,19,20)]

omcdiag(vars.quant,data.train$y_subscribe)
# 5 metodos distintos detectaram algum nivel de correla??o entre as vari?veis

imcdiag(vars.quant,data.train$y_subscribe)
# olhando o VIF observamos valores bem elevados > 10 indicam correla??o entre as vari?veis
# macroeconomicas

library(ppcor)
pcor(vars.quant, method = "pearson")
# as tres vari?veis macro possuem correla??o acima de 50%, o resto a correl?o ? menor

# estrat?gia: vou remover arbitrariamente x18 e x19 e modelar (testem outras combina??es)
# e testar novamente se ainda persiste alguma correl?ao forte entre as vari?veis


vars.quant2 <- data.train[,c(1,11,12,13,14,16,17,19)]

omcdiag(vars.quant2,data.train$y_subscribe)
imcdiag(vars.quant2,data.train$y_subscribe)
pcor(vars.quant2, method = "pearson")
# ainda ? possivel observar rela??o entre x16 e x19, remover x16 tamb?m 

vars.quant3 <- data.train[,c(1,11,12,13,14,17,19)]

omcdiag(vars.quant3,data.train$y_subscribe)
imcdiag(vars.quant3,data.train$y_subscribe)
pcor(vars.quant3, method = "pearson")

data.train2 <- data.train[,-c(16,18,20)]

names  <- names(data.train2) # salva o nome de todas as vari?veis e escreve a f?rmula
f_full <- as.formula(paste("y_subscribe ~",
                           paste(names[!names %in% "y_subscribe"], collapse = " + ")))

glm.full <- glm(f_full, data= data.train2, family= binomial(link='logit'))
summary(glm.full)
# observam-se vari?veis n?o significantes, podemos remover uma de cada vez e testar, ou
# usar o m?todo stepwise que escolhe as vari?veis que minimizem o AIC

# sele??o de vari?veis
glm.step <- stepAIC(glm.full,direction = 'both', trace = TRUE)
summary(glm.step)
# O m?todo manteve apenas vari?veis que minimizaram o AIC

# Aplicando o modelo nas amostras  e determinando as probabilidades
glm.prob.train <- predict(glm.step,type = "response")

glm.prob.test <- predict(glm.step, newdata = data.test, type= "response")

# Verificando a ader?ncia do ajuste log?stico
library(rms)
val.prob(glm.prob.train, data.train$y_subscribe, smooth = F)
# p valor > 5%, n?o podemos rejeitar a hipotese nula

# Comportamento da saida do modelo
hist(glm.prob.test, breaks = 25, col = "lightblue",xlab= "Probabilidades",
     ylab= "Frequ?ncia",main= "Regress?o Log?stica")

boxplot(glm.prob.test ~ data.test$y_subscribe,col= c("red", "green"), horizontal= T)


################################################################################################
# AVALIANDO A PERFORMANCE

# Matricas de discrimina??o para ambos modelos
library(hmeasure) 

glm.train <- HMeasure(data.train$y_subscribe,glm.prob.train)
glm.test  <- HMeasure(data.test$y_subscribe, glm.prob.test)
summary(glm.train)
summary(glm.test)


library(pROC)
roc1 <- roc(data.test$y_subscribe,glm.prob.test)
y1 <- roc1$sensitivities
x1 <- 1-roc1$specificities

plot(x1,y1, type="n",
     xlab = "1 - Especificidade", 
     ylab= "Sensitividade")
lines(x1, y1,lwd=3,lty=1, col="purple") 

################################################################################################
################################################################################################

