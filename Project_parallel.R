#### LIBRARIES AND FUNCTIONS ####

# misclassification error function
misc<-function(yhat,y){
  mean(yhat!=y)
}

# libraries
library(MASS)
library(e1071) # svm
library(tree) # decision trees
library(randomForest) # random forest
library(doParallel) # parallelization
library(profvis) # profiler

# Specify the number of observation and variables
n <- 500  # n. obs
p <- 100  # n. predictive variables

# train sample to split the dataset
set.seed(123)
train<-sample(1:n,ceiling(n/2))

#### SIMULATION ####

# Generates the X matrix with random variables from the multivariate normal distribution
mean_vector <- rep(0, p)  # Mean vectors
cov_matriX <- diag(p)     # Covariance matrix (identity)

log_tuning = matrix(NA,0,ncol = 4)
colnames(log_tuning) = c('cost','kernel','degree','gamma')
cir_tuning = matrix(NA,0,ncol = 4)
colnames(cir_tuning) = c('cost','kernel','degree','gamma')

indexes = c(2,103,104)

cl = makeCluster(10)
registerDoParallel(cl)
#### Tuning for SVM
tuning = foreach (i = 1:20, .combine = rbind) %dopar%{
  set.seed(i)
  library(MASS)
  library(e1071)
  X <- mvrnorm(n, mu = mean_vector, Sigma = cov_matriX)
  # Coefficients for the logistic regression model (random)
  beta <- rnorm(p)
  # Calculates the weighted sum of the predictor variables
  linear_combination <- X %*% beta + rnorm(500, 0, 5)
  
  # Calculate probabilities using the logistic function
  probabilities <- 1 / (1 + exp(-linear_combination))
  
  # Generates the probability-based binary response variable y
  y1 <- ifelse(runif(n) < probabilities, 1, 0)
  ac = princomp(X)
  cs = numeric(500)
  
  for (k in 1:500) {
    cs[k] = ac$scores[k,1]^2 + ac$scores[k,2]^2
  }
  
  y2 = as.numeric(cs>1)
  ds <- data.frame(y1=as.factor(y1), y2=as.factor(y2), X, ac$scores[,1],ac$scores[,2])
  

  best.mod_log<-tune(METHOD=svm,y1~.,data=ds[train,-2],
                 ranges=list(cost=c(0.001,0.01,0.1,1),
                             kernel=c("linear","radial","polynomial"),
                             degree=c(1:2),
                             gamma=1:2))
  best.mod_cir<-tune(METHOD=svm,y2~.,data=ds[train,c(2,103,104)],
                 ranges=list(cost=c(0.001,0.01,0.1,1),
                             kernel=c("linear","radial","polynomial"),
                             degree=c(1:2),
                             gamma=1:2))
  cbind(best.mod_log$best.parameters,best.mod_cir$best.parameters)
}
stopCluster(cl)

logchoice = c(median(tuning[,1]),'linear',median(tuning[,3]),median(tuning[,4]))
circhoice = c(median(tuning[,5]), 'polynomial',median(tuning[,7]),median(tuning[,8]))

################################################################################

#### GENERATION ####

generation = function(seed){
  
  set.seed(seed)
  
  X <- mvrnorm(n, mu = mean_vector, Sigma = cov_matriX) 
  
  # Coefficients for the logistic regression model (random)
  beta <- rnorm(p)
  beta
  # Calculates the weighted sum of the predictor variables
  linear_combination <- X %*% beta + rnorm(500, 0, 5)
  
  # Calculate probabilities using the logistic function
  probabilities <- 1 / (1 + exp(-linear_combination))
  
  # Generates the probability-based binary response variable y
  y1 <- ifelse(runif(n) < probabilities, 1, 0)

  ac = princomp(X)
  cs = numeric(500)
  
  for (i in 1:500) {
    cs[i] = ac$scores[i,1]^2 + ac$scores[i,2]^2
  }
  
  y2 = as.numeric(cs>1)
  ds <- data.frame(y1=as.factor(y1), y2=as.factor(y2), X, PC1 = ac$scores[,1],PC2 = ac$scores[,2])
  
  return(ds)
}

#### SVM ####

SVMfunction <- function(ds, method){
  if (!method %in% c('log', 'cir')) {
    stop("Invalid method. Use 'log' or 'cir'.")
  }
  # Find the overall best SVM model 
  if (method =='log'){
    ds_log = ds[ , -c(2,103,104)]
    ds_logtrain <- ds_log[train,]
    ds_logtest <- ds_log[-train,]
    svm.out <- svm(y1 ~.,data = ds_logtrain, kernel = logchoice[2], cost = as.numeric(logchoice[1]))
    yhat <- predict(svm.out, newdata = ds_logtest)
    return (misc(yhat, ds_logtest$y1))
    
  }else if(method =='cir'){
    ds_cir = ds[ , c(2,103,104)]
    ds_cirtrain <- ds_cir[train,]
    ds_cirtest <- ds_cir[-train,]
    svm.out <- svm(y2 ~.,data = ds_cirtrain, kernel = circhoice[2], cost = as.numeric(circhoice[1]))
    yhat <- predict(svm.out, newdata = ds_cirtest)
    return (misc(yhat, ds_cirtest$y2))
  }
}

#### CLASSIFICATION TREES ####

treesfunction <- function(ds, method){
  if (!method %in% c('log', 'cir')) {
    stop("Invalid method. Use 'log' or 'cir'.")
  }
  if (method =='log'){
    ds_logtest <- ds[,-c(2,103,104)][-train,]
    tree.ds<-tree(y1~., data = ds[,-c(2,103,104)], subset=train)
    # Pruning the tree 
    set.seed(1234)
    cv.ds<-cv.tree(tree.ds,FUN=prune.misclass)
    
    best.size<-cv.ds$size[which.min(cv.ds$dev)]
    
    prune.ds<-prune.misclass(tree.ds, best=best.size)
    
    pruned.y<-predict(prune.ds, ds_logtest, type="class")
    return (misc(pruned.y, ds_logtest$y1))
    
  }else if(method =='cir'){
    ds_cirtest <- ds[,c(2,103,104)][-train,]
    tree.ds<-tree(y2~.,ds[,c(2,103,104)], subset=train)
    # Pruning the tree 
    set.seed(1234)
    cv.ds<-cv.tree(tree.ds, FUN=prune.misclass)
    best.size<-cv.ds$size[which.min(cv.ds$dev)]
    prune.ds<-prune.misclass(tree.ds, best=best.size)
    pruned.y<-predict(prune.ds, ds_cirtest, type="class")
    return (misc(pruned.y, ds_cirtest$y2))
  }
}

#### RANDOM FOREST ####

rdfunction <- function(ds, method){
  if (!method %in% c('log', 'cir')) {
    stop("Invalid method. Use 'log' or 'cir'.")
  }
  if (method == 'log'){
    ds_log = ds[ , -c(2,103,104)]
    ds_logtest <- ds_log[-train,]
    out.rf<-randomForest(y1~.,ds_log,subset=train,
                     importance=T)
    # Test error estimate
    yhat.rf<-predict(out.rf,newdata=ds_logtest)
    misc(yhat.rf,ds_logtest$y1) # RF
    
  }else if (method == 'cir'){
    ds_cir = ds[ , c(2,103,104)]
    ds_cirtest <- ds_cir[-train,]
    out.rf<-randomForest(y2~.,ds_cir,subset=train,
                         importance=T)
    # Test error estimate
    yhat.rf<-predict(out.rf,newdata=ds_cirtest)
    misc(yhat.rf,ds_cirtest$y2) # RF
  }
}

##### PARALLELIZATION ####

indexes = c(2,103,104)

cl = makeCluster(10)
registerDoParallel(cl)

#pb <- txtProgressBar(min = 1, max = 1000, style = 3)
profvis({
parallelo = foreach (i = 20:1020, .combine = rbind) %dopar%{
  
  library(MASS)
  library(e1071) # svm
  library(tree) # decision trees
  library(randomForest) # random forest
  library(doParallel) # parallelization
  
  ds = generation(i)
  
  svm_cir = SVMfunction(ds, method = 'cir')
  svm_log = SVMfunction(ds, method = 'log')
  #tree_cir = treesfunction(generation(i), method = 'cir')
  #tree_log = treesfunction(generation(i), method = 'log')
  rd_cir = rdfunction(ds, method = "cir")
  rd_log = rdfunction(ds, method = "log")
  #setTxtProgressBar(pb, i)
  
  cbind(rbind(c(svm_log, rd_log, svm_cir, rd_cir)))
}
stopCluster(cl)
})


##### Compare the computational time between the methods ####

ds = generation(20)
ds

system.time(SVMfunction(ds, method = 'cir'))

system.time(rdfunction(ds, method = "cir"))

system.time(SVMfunction(ds, method = 'log'))

system.time(rdfunction(ds, method = "log"))

library(bench)
library(ggbeeswarm)
(lbcir <- bench::mark(
  SVMfunction(ds, method = 'cir'),
  rdfunction(ds, method = "cir"),
  check = FALSE
))

plot(lbcir)

(lblog <- bench::mark(
  SVMfunction(ds, method = 'log'),
  rdfunction(ds, method = "log"),
  check = FALSE
))

plot(lblog)




