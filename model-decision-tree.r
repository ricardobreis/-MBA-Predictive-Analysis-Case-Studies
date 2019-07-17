################################################################################################
#
# MODELAGEM PREDITIVA - MBA Business Analytics e Big Data
#
# CASE AULA 4 - TELCO CHURNING
#
################################################################################################
# LENDO OS DADOS
path <- "D:/AULAS MBA FGV/2. MODELAGEM PREDITIVA/1. Turma 1 - Berrini [1o. sem 19]/AULA 2/"

base <- read.csv(paste(path,"telco_churning.csv",sep=""), 
                 sep=",",header = T,stringsAsFactors = T)[,-1] # deletando a primeira coluna

################################################################################################
# ANALISANDO AS VARI?VEIS DA BASE DE DADOS
# ANALISE UNIVARIADA
summary(base)

library("VIM")
matrixplot(base)
aggr(base)
# Aqui temos 11 observa??es com uma vari?vel missing, podemos optar em manter ou deletar essas
# linhas

# ANALISE BIVARIADA
# Vari?veis quantitativas 
boxplot(base$x5_tenure        ~ base$y_churn)
boxplot(base$x9_mthlycharges  ~ base$y_churn)
boxplot(base$x10_totalcharges ~ base$y_churn)

#Vari?veis qualitativas
prop.table(table(base$y_churn))
prop.table(table(base$x1_gender,        base$y_churn),1)
prop.table(table(base$x2_seniorcitizen, base$y_churn),1)
prop.table(table(base$x3_partner,       base$y_churn),1)
prop.table(table(base$x4_dependents,    base$y_churn),1)
prop.table(table(base$x6_contract,      base$y_churn),1)
prop.table(table(base$x7_paperlessbill, base$y_churn),1)
prop.table(table(base$x8_paymntmethod,  base$y_churn),1)
prop.table(table(base$x11_phone,        base$y_churn),1)
prop.table(table(base$x12_multipleline, base$y_churn),1)
prop.table(table(base$x13_internet,     base$y_churn),1)
prop.table(table(base$x14_onlinesecur,  base$y_churn),1)
prop.table(table(base$x15_onlinebckp,   base$y_churn),1)
prop.table(table(base$x16_devprotect,   base$y_churn),1)
prop.table(table(base$x17_techsupport,  base$y_churn),1)
prop.table(table(base$x18_streamtv,     base$y_churn),1)
prop.table(table(base$x19_streammovie,  base$y_churn),1)


################################################################################################
# AMOSTRAGEM DO DADOS
library(caret)

set.seed(12345)
index <- createDataPartition(base$y_churn, p= 0.7,list = F)

data.train <- base[index, ] # base de desenvolvimento: 70%
data.test  <- base[-index,] # base de teste: 30%

# Checando se as propor??es das amostras s?o pr?ximas ? base original
prop.table(table(base$y_churn))
prop.table(table(data.train$y_churn))
prop.table(table(data.test$y_churn))

# Algoritmos de arvore necessitam que a vari?vel resposta num problema de classifica??o seja 
# um factor; convertendo aqui nas amostras de desenvolvimento e teste
data.train$y_churn <- as.factor(data.train$y_churn)
data.test$y_churn  <- as.factor(data.test$y_churn)

################################################################################################
# MODELAGEM DOS DADOS - ?RVORE DE CLASSIFICA??O

names  <- names(data.train) # salva o nome de todas as vari?veis e escreve a f?rmula
f_full <- as.formula(paste("y_churn ~",
                           paste(names[!names %in% "y_churn"], collapse = " + ")))

library(rpart)
# Aqui come?amos a construir uma ?rvore completa e permitimos apenas que as parti??es
# tenham ao menos 50 observa??es
tree.full <- rpart(data= data.train, f_full,
                   control = rpart.control(minbucket=50),
                   method = "class")

# saida da ?rvore
tree.full
summary(tree.full)

# Import?ncia das vari?veis
round(tree.full$variable.importance, 3)

# AValiando a necessidade de poda da ?rvore
printcp(tree.full)
plotcp(tree.full)

# Aqui conseguimos podar a ?rvore
tree.prune <- prune(tree.full, cp= tree.full$cptable[which.min(tree.full$cptable[,"xerror"]),"CP"])

# Plotando a ?rvore

library(rpart.plot)
rpart.plot(tree.full, cex = 1.3,type=0,
           extra=104, box.palette= "BuRd",
           branch.lty=3, shadow.col="gray", nn=TRUE, main="?rvore de Classifica??o")


rpart.plot(tree.prune, cex = 1.3,type=0,
           extra=104, box.palette= "BuRd",
           branch.lty=3, shadow.col="gray", nn=TRUE, main="?rvore de Classifica??o")

# Aplicando o modelo nas amostras  e determinando as probabilidades
tree.prob.train <- predict(tree.prune, type = "prob")[,2]

tree.prob.test  <- predict(tree.prune, newdata = data.test, type = "prob")[,2]

# Comportamento da saida do modelo
hist(tree.prob.test, breaks = 25, col = "lightblue",xlab= "Probabilidades",
     ylab= "Frequ?ncia",main= "?rvore de Classifica??o")

boxplot(tree.prob.test ~ data.test$y_churn,col= c("green", "red"), horizontal= T)


################################################################################################
# AVALIANDO A PERFORMANCE

# Matricas de discrimina??o para ambos modelos
library(hmeasure) 

tree.train <- HMeasure(data.train$y_churn,tree.prob.train)
tree.test  <- HMeasure(data.test$y_churn, tree.prob.test)
summary(tree.train)
summary(tree.test)


library(pROC)
roc1 <- roc(data.test$y_churn,tree.prob.test)
y1 <- roc1$sensitivities
x1 <- 1-roc1$specificities

plot(x1,y1, type="n",
     xlab = "1 - Especificidade", 
     ylab= "Sensitividade")
lines(x1, y1,lwd=3,lty=1, col="purple") 

################################################################################################
################################################################################################

