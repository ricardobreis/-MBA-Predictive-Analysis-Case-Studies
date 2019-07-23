################################################################################################
#
# MODELAGEM PREDITIVA - MBA Business Analytics e Big Data
#
# CASE AULA 5 - EMPLOYEE TURNOVER
#
################################################################################################
# LENDO OS DADOS

path <- "C:/Users/AutoLogon.ACAD/Desktop/AULA 5"

base <- read.csv(paste(path,"employee_turnover.csv",sep=""), 
                 sep=",",header = T,stringsAsFactors = T)[,-1] # deletando a primeira coluna

################################################################################################
# ANALISANDO AS VARI?VEIS DA BASE DE DADOS
# ANALISE UNIVARIADA
summary(base)

library("VIM")
matrixplot(base)
aggr(base)

# ANALISE BIVARIADA
# Vari?veis quantitativas 
boxplot(base$x3_timecompany ~ base$y_empleft)
boxplot(base$x6_workhours   ~ base$y_empleft)
boxplot(base$x7_nprojects   ~ base$y_empleft)
boxplot(base$x8_lasteval    ~ base$y_empleft)
boxplot(base$x9_satisflevel ~ base$y_empleft)

#Vari?veis qualitativas
prop.table(table(base$y_empleft))
prop.table(table(base$x1_department,   base$y_empleft),1)
prop.table(table(base$x2_salary,       base$y_empleft),1)
prop.table(table(base$x4_promotion5yr, base$y_empleft),1)
prop.table(table(base$x5_workaccident, base$y_empleft),1)

################################################################################################
# AMOSTRAGEM DO DADOS
library(caret)

set.seed(12345)
index <- createDataPartition(base$y_empleft, p= 0.7,list = F)

data.train <- base[index, ] # base de desenvolvimento: 70%
data.test  <- base[-index,] # base de teste: 30%

# Checando se as propor??es das amostras s?o pr?ximas ? base original
prop.table(table(base$y_empleft))
prop.table(table(data.train$y_empleft))
prop.table(table(data.test$y_empleft))

# Algoritmos de arvore necessitam que a vari?vel resposta num problema de classifica??o seja 
# um factor; convertendo aqui nas amostras de desenvolvimento e teste
data.train$y_empleft <- as.factor(data.train$y_empleft)
data.test$y_empleft  <- as.factor(data.test$y_empleft)

################################################################################################
# MODELAGEM DOS DADOS - M?TODOS DE ENSEMBLE

names  <- names(data.train) # salva o nome de todas as vari?veis e escreve a f?rmula
f_full <- as.formula(paste("y_empleft ~",
                           paste(names[!names %in% "y_empleft"], collapse = " + ")))

# a) Random Forest
library(randomForest)
# Aqui come?amos a construir um modelo de random forest usando sqrt(n var) | mtry = default
# Construimos 500 ?rvores, e permitimos n?s finais com no m?nimo 50 elementos
rndfor <- randomForest(f_full,data= data.train,importance = T, nodesize =50, ntree = 500)
rndfor

# Avaliando a evolu??o do erro com o aumento do n?mero de ?rvores no ensemble
plot(rndfor, main= "Mensura??o do erro")
legend("topright", c('Out-of-bag',"1","0"), lty=1, col=c("black","green","red"))

# Uma avalia??o objetiva indica que a partir de ~30 ?rvores n?o mais ganhos expressivos
rndfor2 <- randomForest(f_full,data= data.train,importance = T, nodesize =50, ntree = 30)
rndfor2

# Import?ncia das vari?veis
varImpPlot(rndfor2, sort= T, main = "Import?ncia das Vari?veis")

# Aplicando o modelo nas amostras  e determinando as probabilidades
rndfor2.prob.train <- predict(rndfor2, type = "prob")[,2]
rndfor2.prob.test  <- predict(rndfor2,newdata = data.test, type = "prob")[,2]

# Comportamento da saida do modelo
hist(rndfor2.prob.test, breaks = 25, col = "lightblue",xlab= "Probabilidades",
     ylab= "Frequ?ncia",main= "Random Forest")

boxplot(rndfor2.prob.test ~ data.test$y_empleft,col= c("green", "red"), horizontal= T)

#-------------------------------------------------------------------------------------------
# b) Boosted trees
library(adabag)
# Aqui construimos inicialmente um modelo boosting com 1000 itera??es, profundidade 1
# e minbucket 50, os pesos das ?rvores ser? dado pelo algortimo de Freund
boost <- boosting(f_full, data= data.train, mfinal= 200, 
                  coeflearn = "Freund", 
                  control = rpart.control(minbucket= 50,maxdepth = 1))

# Avaliando a evolu??o do erro conforme o n?mero de itera??es aumenta
plot(errorevol(boost, data.train))

# podemos manter em 200 itera??es

# Import?ncia das vari?veis
var_importance <- boost$importance[order(boost$importance,decreasing = T)]
var_importance
importanceplot(boost)

# Aplicando o modelo na amostra de teste e determinando as probabilidades
boost.prob.train <- predict.boosting(boost, data.train)$prob[,2]
boost.prob.test  <- predict.boosting(boost, data.test)$prob[,2]

# Comportamento da saida do modelo
hist(boost.prob.test, breaks = 25, col = "lightblue",xlab= "Probabilidades",
     ylab= "Frequ?ncia",main= "Boosting")

boxplot(boost.prob.test ~ data.test$y_empleft,col= c("green", "red"), horizontal= T)

################################################################################################
# AVALIANDO A PERFORMANCE

# Matricas de discrimina??o para ambos modelos
library(hmeasure) 

rndfor.train  <- HMeasure(data.train$y_empleft,rndfor2.prob.train)
rndfor.test  <- HMeasure(data.test$y_empleft,rndfor2.prob.test)
summary(rndfor.train)
summary(rndfor.test)

boost.train <- HMeasure(data.train$y_empleft,boost.prob.train)
boost.test  <- HMeasure(data.test$y_empleft,boost.prob.test)
summary(boost.train)
summary(boost.test)


library(pROC)
roc1 <- roc(data.test$y_empleft,rndfor2.prob.test)
y1 <- roc1$sensitivities
x1 <- 1-roc1$specificities

roc2 <- roc(data.test$y_empleft,boost.prob.test)
y2 <- roc2$sensitivities
x2 <- 1-roc2$specificities


plot(x1,y1, type="n",
     xlab = "1 - Especificidade", 
     ylab= "Sensitividade")
lines(x1, y1,lwd=3,lty=1, col="purple") 
lines(x2, y2,lwd=3,lty=1, col="blue") 
legend("topright", c('Random Forest',"Boosting"), lty=1, col=c("purple","blue"))

################################################################################################
################################################################################################

