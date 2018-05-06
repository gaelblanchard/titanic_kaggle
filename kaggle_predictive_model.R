#Gael Blanchard
#Titanic 
#The libraries we'll be using
# caret allows us to construct our predictive model
#doSNOW allows us to construct the model faster
library(caret)
library(doSNOW)
#The working directory "/???/??/??" format
setwd("/file/directory/for/Titanic_data")
#loading data from our csv into 2 variables trainSet and testSet
trainSet <- read.table("train.csv",sep=",",header = TRUE)
testSet <- read.table("test.csv",sep = ",", header = TRUE)
#Setting factors from the train set for consideration in out predictive model
#is age missing accounts for any missing age data
trainSet$Survived <- factor(trainSet$Survived)
trainSet$is_age_missing <- ifelse(is.na(trainSet$Age),1,0)
trainSet$travelers <- trainSet$SibSp + trainSet$Parch + 1
trainSet$Pclass <- factor(trainSet$Pclass)
trainSet$is_age_missing <- factor(trainSet$is_age_missing)
train_test <- subset(trainSet,select = c(Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,is_age_missing,travelers))
dummy <- dummyVars(~.,data = train_test[,-1])
dummy_train <- predict(dummy,train_test[,-1])
#Our pre.process variable allows us to replace missing data with a substituted value
#Imputation
#Tested methods:
#knnImpute(3rd best), medianImpute(best so far), bagImpute (2nd best)
pre.process <- preProcess(dummy_train,method="medianImpute")
imputed.data <- predict(pre.process,dummy_train)
train_test$Age <- imputed.data[,6]
set.seed(123)
#partition our data after forming imputations
partition_indexes <- createDataPartition(trainSet$Survived, times=1,p=0.7,list=FALSE)
titanic.train <- train_test[partition_indexes,]
titanic.test <- train_test[-partition_indexes,]
train_test.control <- trainControl(method="repeatedcv",number=10,repeats=3,search="grid")
tune.grid <- expand.grid(eta = c(0.05,0.075,0.1),nrounds=c(50,75,100),max_depth = 6:8,min_child_weight = c(2.0,2.25,2.5),colsample_bytree=c(0.3,0.4,0.5),gamma=0,subsample=1)
cl <- makeCluster(3,type = "SOCK")
registerDoSNOW(cl)
#More predictive models to try:
# kknn,native (k-nearest neighbors)[Provided a slight accuracy boost],
# nueralnet,nnet (nueral net),
# nb, nbDiscrete (Naive Bayes), klaR regularized discriminant analysis
caret.cv <- train(Survived~., data = titanic.train, method="xgbTree",tuneGrid=tune.grid,trControl=train_test.control)
stopCluster(cl)
preds <- predict(caret.cv, titanic.test)
#A visual representation of the results from our model using 
confusionMatrix(preds,titanic.test$Survived)