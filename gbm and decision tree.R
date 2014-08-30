setwd("C:/Users/zhengkya/Desktop/Crash&Repair Data/CSV/")
library(gbm)
library(caret)
library(rpart)
library(corrgram)
library(party)
library(tree)
library(MESS)
library(rpart.plot)

data <- read.csv("Mapping_recent_crash_repair.csv",header = TRUE)
data <- data[,-c(1,2,19)] #Test, rowname,IMEI varables are deleted

#Split data set 
row_train <- sample(seq(1,nrow(data),by=1),0.6*nrow(data),replace=FALSE)
data_train <- data[row_train,]
data_test <- data[-row_train,]


#########################################
########Decision Tree###################
data_train$repair_id[which(data_train$repair_id == 1)] <- "repaired"
data_train$repair_id[which(data_train$repair_id == 0)] <- "Norepair"
control = tree.control(nobs=nrow(data_train), mincut = 20, minsize = 40, mindev = 0.0005)
tree.fit <- rpart(repair_id~.,data_train,control=rpart.control(minsplit = 20,minbucket = 40,cp=0.01),method="class")

#Visualize the tree
plot(tree.fit)
text(tree.fit, use.n=TRUE)
prp(tree.fit, extra=6, uniform=T, branch=0.5, yesno=T, border.col=1, type=4,xsep="/",box.col="green",branch.col="red")


#Predict true values/false values
pred <-predict(tree.fit, newdata=data_test, type="prob")[,2]
obs = data_test$repair_id

#Calculate ROC
threshold = 0.5
roc.curve= function(threshold,print = FALSE){
  Ps = (pred>threshold)*1
  FP = sum((Ps==1)*(obs==0))/sum(obs==0)
  TP = sum((Ps==1)*(obs == 1))/sum(obs==1)
  if(print==TRUE){
    print(table(Observed = obs,Predicted=Ps))
  }
  vect = c(FP,TP)
  names(vect) = c("FPR","TPR")
  return(vect)
}
roc.curve(threshold,print=TRUE)

ROC.curve <- Vectorize(roc.curve)
Grid.ROC <- ROC.curve(seq(0,1,by=0.01))

plot(Grid.ROC[1,],Grid.ROC[2,], lwd = 3, col = "red", xlab = "False Positive Rate", ylab = "True Positive Rate",lty=2)

seg_mat <- Grid.ROC[,c(1,16,47,98)]

for(i in c(1:4)){
  if(i!=4){
    segments(seg_mat[1,i],seg_mat[2,i],seg_mat[1,i+1],seg_mat[2,i+1],col="red",lwd = 3,lty=1)
  }
}


lines(seq(0,1,by=0.01),seq(0,1,by=0.01),col= "blue",lwd=3)

#Area Under curve
auc(Grid.ROC[1,],Grid.ROC[2,],type = 'spline')



##########################################
#########Boosted Tree#####################
##########################################

fitControl <- trainControl(method="repeatedcv",number=10, repeats=1,classProbs = TRUE,
                            summaryFunction = twoClassSummary)
tuneGrid = expand.grid(n.trees = c(100,200,300), interaction.depth=c(3,4,5),shrinkage=c(0.05,0.1,0.15))
boostedTr <- train(as.factor(repair_id)~.,data=data_train,tuneGrid = tuneGrid, trControl=fitControl,method="gbm")

#Best parameters: n.trees = 300, interaction.depth = 5 and shrinkage = 0.1
data_train$repair_id[which(data_train$repair_id == "repaired")] <- 1
data_train$repair_id[which(data_train$repair_id == "Norepair")] <- 0
boostedTr <- gbm(repair_id~.,data=data_train,distribution="bernoulli", n.trees = 300, interaction.depth= 5, shrinkage=0.1)

pred <- predict(boostedTr, newdata=data_test,n.trees=300,type="response")
obs <- data_test$repair_id


#Plot the relative importance of predictors
x <- data.frame(summary(boostedTr),row.names=NULL)
barplot(x$rel.inf, beside=T, xlab = "Bucket Area", ylab = "Relative Importance", main="Boosted Tree Variable Importance",horiz=FALSE, names.arg = row.names(x),las=1, cex.names=0.7,col="blue")


#Calculate ROC curve
threshold = 0.5
roc.curve= function(threshold,print = FALSE){
  Ps = (pred>threshold)*1
  FP = sum((Ps==1)*(obs==0))/sum(obs==0)
  TP = sum((Ps==1)*(obs == 1))/sum(obs==1)
  if(print==TRUE){
    print(table(Observed = obs,Predicted=Ps))
  }
  vect = c(FP,TP)
  names(vect) = c("FPR","TPR")
  return(vect)
}
roc.curve(threshold,print=TRUE)

ROC.curve = Vectorize(roc.curve)

M.ROC=ROC.curve(seq(0,1,by=.01))
plot(M.ROC[1,],M.ROC[2,],lwd=3,col="red",type="l",xlab = "False Positive Rate", ylab="True Positive Rate")

lines(seq(0,1,by=0.01),seq(0,1,by=0.01),col="blue",lwd=3)


#Area Under curve
auc(M.ROC[1,],M.ROC[2,],type = 'spline')




