library(caret)
library(gclus)
library(ROCR)
library(party)
library(randomForest)

setwd("C:/Users/Sameer/Desktop/Study/Practical Machine Learning/Project")

PML_Training<-read.csv("pml_training.csv",header=T)

PML_Testing<-read.csv("pml_testing.csv",header=T)

summary(PML_Training)

columns<-which(names(PML_Training) %in% c("classe",
                                          "num_window",
                                          "roll_belt",
                                          "pitch_belt",
                                          "yaw_belt",
                                          "total_accel_belt",
                                          "gyros_belt_x",
                                          "gyros_belt_y",
                                          "gyros_belt_z",
                                          "accel_belt_x",
                                          "accel_belt_y",
                                          "accel_belt_z",
                                          "magnet_belt_x",
                                          "magnet_belt_y",
                                          "magnet_belt_z",
                                          "roll_arm",
                                          "pitch_arm",
                                          "yaw_arm",
                                          "total_accel_arm",
                                          "gyros_arm_x",
                                          "gyros_arm_y",
                                          "gyros_arm_z",
                                          "accel_arm_x",
                                          "accel_arm_y",
                                          "accel_arm_z",
                                          "magnet_arm_x",
                                          "magnet_arm_y",
                                          "magnet_arm_z",
                                          "roll_dumbbell",
                                          "pitch_dumbbell",
                                          "yaw_dumbbell"))

## Use relevant columns
PML_Training_Subset<-PML_Training[,columns]

inTrain<-createDataPartition(y=PML_Training_Subset$classe,p=0.75,list=FALSE)
training<-PML_Training_Subset[inTrain,]
testing<-PML_Training_Subset[-inTrain,]





## Factor Reduction
prComp<-prcomp(training[,-31],center=T,scale=T)
prComp<-prcomp(training[,-31],center=T,scale=T)

summary(prComp)

#Eigen Values - show that we should use 10 components
prComp$sdev^2

#Screeplot shows that we should use 9 components
screeplot(prComp,main="Scree Plot",xlab="Components")
screeplot(prComp,type="line",main="Scree  Plot")

#We will go with Conservative approach and use 9 components
preProc<-preProcess(training[,-31],method="pca",pcaComp=9)
trainPC<-predict(preProc,training[,-31])
testPC<-predict(preProc,testing[,-31])

r = randomForest(training$classe ~., data=trainPC, importance=TRUE, do.trace=100)
summary(r)

pred<-predict(r,testPC)
confusionMatrix(testing$classe,pred)

######Simple Random Forest##########



r2 = randomForest(classe ~., data=training, importance=TRUE, do.trace=100)
pred2<-predict(r2,testing[,-31])
confusionMatrix(testing$classe,pred2)


PML_Testing

columns_PML_TEST<-which(names(PML_Testing) %in% c("num_window",
                                          "roll_belt",
                                          "pitch_belt",
                                          "yaw_belt",
                                          "total_accel_belt",
                                          "gyros_belt_x",
                                          "gyros_belt_y",
                                          "gyros_belt_z",
                                          "accel_belt_x",
                                          "accel_belt_y",
                                          "accel_belt_z",
                                          "magnet_belt_x",
                                          "magnet_belt_y",
                                          "magnet_belt_z",
                                          "roll_arm",
                                          "pitch_arm",
                                          "yaw_arm",
                                          "total_accel_arm",
                                          "gyros_arm_x",
                                          "gyros_arm_y",
                                          "gyros_arm_z",
                                          "accel_arm_x",
                                          "accel_arm_y",
                                          "accel_arm_z",
                                          "magnet_arm_x",
                                          "magnet_arm_y",
                                          "magnet_arm_z",
                                          "roll_dumbbell",
                                          "pitch_dumbbell",
                                          "yaw_dumbbell"))

pred_final<-predict(r2,PML_Testing[,columns_PML_TEST])




######Random Forest With Bagging##########
predictors<-PML_Training_Subset[,-31]
Classe<-PML_Training_Subset[,31]
treeBag<-bag(predictors,Classe,B=10,
             bagControl=bagControl(fit=ctreeBag$fit,
                                   predict=ctreeBag$pred,
                                   aggregate=ctreeBag$aggregate))


pred3<-predict(treeBag,predictors)

confusionMatrix(Classe,pred3)


#################################



r,r2,treeBag

plot(r2, log="y",main="Simple Random Forrest")
legend("topright", colnames(r2$err.rate),col=1:6,cex=0.8,fill=1:6)
varImpPlot(r2,main="Variable Importance")

VariableUsed<-varUsed(r2, by.tree=FALSE, count=TRUE)
which(VariableUsed==max(VariableUsed))
which(VariableUsed==min(VariableUsed))


#############################################################################
tree <- getTree(r,k=2,labelVar=TRUE)
d <- to.dendrogram(tree)
str(d)
plot(d,center=TRUE,leaflab='none',edgePar=list(t.cex=1,p.col=NA,p.lty=0))

to.dendrogram <- function(dfrep,rownum=1,height.increment=0.1){
  
  if(dfrep[rownum,'status'] == -1){
    rval <- list()
    
    attr(rval,"members") <- 1
    attr(rval,"height") <- 0.0
    attr(rval,"label") <- dfrep[rownum,'prediction']
    attr(rval,"leaf") <- TRUE
    
  }else{##note the change "to.dendrogram" and not "to.dendogram"
    left <- to.dendrogram(dfrep,dfrep[rownum,'left daughter'],height.increment)
    right <- to.dendrogram(dfrep,dfrep[rownum,'right daughter'],height.increment)
    rval <- list(left,right)
    
    attr(rval,"members") <- attr(left,"members") + attr(right,"members")
    attr(rval,"height") <- max(attr(left,"height"),attr(right,"height")) + height.increment
    attr(rval,"leaf") <- FALSE
    attr(rval,"edgetext") <- dfrep[rownum,'split var']
  }
  
  class(rval) <- "dendrogram"
  
  return(rval)
}

##reference 
##http://www.alanfielding.co.uk/multivar/rf.htm
##http://www.r-bloggers.com/a-brief-tour-of-the-trees-and-forests/