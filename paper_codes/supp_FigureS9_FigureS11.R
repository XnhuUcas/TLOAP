rm(list=ls())
gc()
source("Performance_Eva.R")
source("translasso_func.R")
library(glmtrans)
library(Matrix)
library(MASS)
library(glmnet)
library(Rsolnp)
library(caret)
library(ncvreg)
library(splines)
library(ARTtransfer)
source("ART.R")

eq=function(w){sum(w)-1}

obj_logit = function(w){
  alpha1 = sum((theta_hat%*%w)*data.train$data.y[[1]])
  alpha2 = theta_hat%*%w
  if(family=='binomial'){
    alpha2=sum(log(1+exp(alpha2)))
  }else if(family=='poisson'){
    alpha2=sum(exp(alpha2))
  }else if(family=='gaussian'){
    alpha2=sum(alpha2^2/2)
  }
  return(alpha2-alpha1)
}

############### Load Data

data.raw <- read.csv('diabetes.csv', header = T, encoding = "UTF-8")
x.colnames <- colnames(data.raw)[-c(8,9)]
z.colnames <- "Age"
data.raw$age_stage <- "null"
data.raw$age_stage <- ifelse(data.raw$Age >= 20 & data.raw$Age <= 30, "20-30", data.raw$age_stage)
data.raw$age_stage <- ifelse(data.raw$Age > 30 & data.raw$Age <= 50, "30-50", data.raw$age_stage)
data.raw$age_stage <- ifelse(data.raw$Age > 50, "50+", data.raw$age_stage)
data.raw$Age <- (data.raw$Age-min(data.raw$Age))/(max(data.raw$Age)-min(data.raw$Age))
source.ind <- sort(unique(data.raw$age_stage))

## add noise
noise.ind <- which(data.raw$age_stage==source.ind[2])
data.raw[noise.ind, x.colnames] <- data.raw[noise.ind, x.colnames] + matrix(rnorm(length(noise.ind)*length(x.colnames), mean=1.5, sd = 1), nrow = length(noise.ind), ncol = length(x.colnames))
data.raw[noise.ind, z.colnames] <- data.raw[noise.ind, z.colnames] + matrix(rnorm(length(noise.ind)*length(z.colnames), mean=1.5, sd = 1), nrow = length(noise.ind), ncol = length(z.colnames))

p <- 7
s.cnt <- length(source.ind)
qz <- rep(1,s.cnt)
family <- "binomial"
nfold <- 5
non.ind <- 1

rep <- 500
brea_error10=0
brea_conve10=0
brea_error20=0
brea_conve20=0
brea_errorjcv=0
brea_convejcv=0
brea_abn=0
options(warn = -1)

kl.loss.train <- array(NA,dim=c(s.cnt, rep, 8))
kl.loss.test <- array(NA,dim=c(s.cnt, rep, 8))
er.train <- array(NA,dim=c(s.cnt, rep, 8))
er.test <- array(NA,dim=c(s.cnt, rep, 8))

############## Running 

for(h0 in 1:s.cnt){
  
  data.x.list <- vector(mode = 'list', length = length(source.ind))
  data.x.list[[1]] <- as.matrix(data.raw[which(data.raw$age_stage==source.ind[h0]), x.colnames])
  data.u.list <- vector(mode = 'list', length = length(source.ind))
  data.u.list[[1]] <- as.matrix(data.raw[which(data.raw$age_stage==source.ind[h0]), z.colnames])
  data.y.list <- vector(mode = 'list', length = length(source.ind))
  data.y.list[[1]] <- as.matrix(data.raw[which(data.raw$age_stage==source.ind[h0]), 'Outcome'])
  source.ind.del <- source.ind[-which(source.ind==source.ind[h0])]
  for(i in 1:length(source.ind.del)){
    row.ind <- which(data.raw$age_stage==source.ind.del[i])
    data.x.list[[i+1]] <- as.matrix(data.raw[row.ind, x.colnames])
    data.u.list[[i+1]] <- as.matrix(data.raw[row.ind, z.colnames])
    data.y.list[[i+1]] <- as.matrix(data.raw[row.ind, 'Outcome'])
  }
  data.model.list <- list(data.y=data.y.list, data.x=data.x.list, data.u=data.u.list)
  size <- sapply(data.model.list$data.x, function(x) nrow(x))
  ndg <- 3
  nknots <- ceiling(size[1]^(1/5))
  ndf <- ndg+nknots
  
  for(r in 1:rep){
    
    set.seed(20230327+100*r+200*h0)
    cat('---------------------------------\n')
    cat('Now target is', source.ind[h0], 'in the', r, 'th replicate..\n')
    cat('---------------------------------\n')
    
    split.all.ind <- createFolds(data.model.list$data.y[[1]], nfold)
    data.train <- data.model.list
    data.train$data.y[[1]] <- as.matrix(data.train$data.y[[1]][-split.all.ind[[1]]])
    data.train$data.x[[1]] <- as.matrix(data.train$data.x[[1]][-split.all.ind[[1]],])
    data.train$data.u[[1]] <- as.matrix(data.train$data.u[[1]][-split.all.ind[[1]],])
    size.train <- sapply(data.train$data.x, function(x) nrow(x))
    
    data.test.y <- as.matrix(data.model.list$data.y[[1]][split.all.ind[[1]]])
    data.test.x <- as.matrix(data.model.list$data.x[[1]][split.all.ind[[1]],])
    data.test.u <- as.matrix(data.model.list$data.u[[1]][split.all.ind[[1]],])
    data.test <- list(data.y=data.test.y, data.x=data.test.x, data.u=data.test.u)
    size.test <- length(split.all.ind[[1]])
    
    ###### TLOAP cv10
    bsz.tar.te=array(0,dim=c(size.test,ndf*qz[1]))
    for(h in 1:s.cnt){
      bsz.tar <- array(0,dim=c(size.train[h], ndf*qz[h]))
      if(h==1){
        bsz.tar[,1:ndf] <- data.train$data.x[[h]][,non.ind]*bs(data.train$data.u[[h]], df=ndf, degree = ndg)
        bsz.tar.te[,1:ndf] <- data.test$data.x[,non.ind]*predict(bs(data.train$data.u[[h]], df=ndf, degree = ndg), data.test$data.u)
      }else{
        bsz.tar[,1:ndf] <- data.train$data.x[[h]][,non.ind]*bs(data.train$data.u[[h]], df=ndf, degree = ndg)
      }
      data.train$data.merge[[h]] <- cbind(data.train$data.y[[h]], bsz.tar, data.train$data.x[[h]][,-non.ind])
    }
    data.test$data.merge <- cbind(data.test$data.y, bsz.tar.te, data.test$data.x[,-non.ind])
    
    n_group_10 <- {if((size.train[1]%%10)== 0) size.train[1]%/%10 else size.train[1]%/%10+1} 
    theta_hat <- array(0,dim=c(size.train[1],s.cnt))
    tar.cv <- cbind(1,data.train$data.merge[[1]][,-1])
    for(nf in 1:n_group_10){
      split.ind <- (1:size.train[1])[((nf-1)*10+1):min(nf*10,size.train[1])]
      train.data.cv <- data.train
      train.data.cv$data.merge[[1]] <- train.data.cv$data.merge[[1]][-split.ind,]
      est.beta.cv <- matrix(NA, nrow=s.cnt, ncol=p-qz[1])
      glm.tr <- vector(mode='list', length=s.cnt)
      for(j in 1:s.cnt){
        dataglm <- as.data.frame(train.data.cv$data.merge[[j]]); colnames(dataglm) <- c('respon',paste('z',1:(ndf*qz[j]),sep = ''),paste('x',1:(p-qz[j]),sep = ''))
        glm.tr[[j]] <- glm(respon~., data = dataglm, family=family, control=list(maxit=1000))$coefficients
        est.beta.cv[j,] <- glm.tr[[j]][(ndf*qz[j]+2):length(glm.tr[[j]])]
      }
      for(j in 1:s.cnt){
        beta.est.train.mat.cv <- as.matrix(c(glm.tr[[1]][1], glm.tr[[1]][2:(ndf*qz[j]+1)], est.beta.cv[j,]))
        theta_hat[split.ind,j] <-tar.cv[split.ind,]%*%beta.est.train.mat.cv
      }
    }
    solve.weight=try(solnp(rep(1/s.cnt,s.cnt), fun = obj_logit, eqfun=eq, eqB=0, LB=rep(0,s.cnt), control=list(trace=0)),silent=TRUE)
    if ('try-error' %in% class(solve.weight)){
      brea_error10=brea_error10+1
      next
    }else{
      if(solve.weight$convergence!=0){
        brea_conve10=brea_conve10+1
        next
      } 
    }
    
    beta.est.train.mat <- matrix(NA, p-qz[1], s.cnt)
    gvcm.res <- vector(mode='list', length=s.cnt)
    for(k in 1:s.cnt){
      data.train.frame = as.data.frame(data.train$data.merge[[k]]); colnames(data.train.frame) <- c('respon',paste('z',1:(ndf*qz[k]),sep = ''),paste('x',1:(p-qz[k]),sep = ''))
      gvcm.res[[k]] <- glm(respon~., data = data.train.frame, family=family, control=list(maxit=1000))
      beta.est.train.mat[,k] <- gvcm.res[[k]]$coefficients[(ndf*qz[k]+2):length(gvcm.res[[k]]$coefficients)]
    }
    weight.est <- solve.weight$par
    beta.est.train <- beta.est.train.mat%*%weight.est
    para.ma <- c(gvcm.res[[1]]$coefficients[1:(ndf*qz[1]+1)], beta.est.train)
    
    pred.ma.train <- cbind(matrix(rep(1,size.train[1]),size.train[1],1), data.train$data.merge[[1]][,-1])%*%para.ma
    kl.loss.train[h0,r,1] <- -2*sum(pred.ma.train*data.train$data.y[[1]]-log(1+exp(pred.ma.train)))/size.train[1]
    pred.ma.test <- cbind(matrix(rep(1,size.test[1]),size.test[1],1), data.test$data.merge[,-1])%*%para.ma
    kl.loss.test[h0,r,1] <- -2*sum(pred.ma.test*data.test$data.y-log(1+exp(pred.ma.test)))/size.test[1]
    
    y.sign.train <- ifelse(exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.ma)/(1+exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.ma))>0.5,1,0)
    er.train[h0,r,1] <-mean(data.train$data.y[[1]]!=y.sign.train)
    y.sign.test <- ifelse(exp(cbind(1,data.test$data.merge[,-1])%*%para.ma)/(1+exp(cbind(1,data.test$data.merge[,-1])%*%para.ma))>0.5,1,0)
    er.test[h0,r,1] <-mean(data.test$data.y!=y.sign.test)
    
    
    ## TLOAP-cv5
    n_group_5 <- {if((size.train[1]%%5)== 0) size.train[1]%/%5 else size.train[1]%/%5+1} 
    theta_hat <- array(0,dim=c(size.train[1],s.cnt))
    for(nf in 1:n_group_5){
      split.ind <- (1:size.train[1])[((nf-1)*5+1):min(nf*5,size.train[1])]
      train.data.cv <- data.train
      train.data.cv$data.merge[[1]] <- train.data.cv$data.merge[[1]][-split.ind,]
      est.beta.cv <- matrix(NA, nrow=s.cnt, ncol=p-qz[1])
      glm.tr <- vector(mode='list', length=s.cnt)
      for(j in 1:s.cnt){
        dataglm <- as.data.frame(train.data.cv$data.merge[[j]]); colnames(dataglm) <- c('respon',paste('z',1:(ndf*qz[j]),sep = ''),paste('x',1:(p-qz[j]),sep = ''))
        glm.tr[[j]] <- glm(respon~., data = dataglm, family=family, control=list(maxit=1000))$coefficients
        est.beta.cv[j,] <- glm.tr[[j]][(ndf*qz[j]+2):length(glm.tr[[j]])]
      }
      for(j in 1:s.cnt){
        beta.est.train.mat.cv <- as.matrix(c(glm.tr[[1]][1], glm.tr[[1]][2:(ndf*qz[j]+1)], est.beta.cv[j,]))
        theta_hat[split.ind,j] <- tar.cv[split.ind,]%*%beta.est.train.mat.cv
      }
    }
    solve.weight20=try(solnp(rep(1/s.cnt,s.cnt), fun = obj_logit, eqfun=eq, eqB=0, LB=rep(0,s.cnt), control=list(trace=0)),silent=TRUE)
    if ('try-error' %in% class(solve.weight20)){
      brea_error20=brea_error20+1
      next
    }else{
      if(solve.weight20$convergence!=0){
        brea_conve20=brea_conve20+1
        next
      } 
    }
    weight.est20 <- solve.weight20$par
    beta.est.train20 <- beta.est.train.mat%*%weight.est20
    para.ma20 <- c(gvcm.res[[1]]$coefficients[1:(ndf*qz[1]+1)], beta.est.train20)
    
    pred.ma20.train <- cbind(matrix(rep(1,size.train[1]),size.train[1],1), data.train$data.merge[[1]][,-1])%*%para.ma20
    kl.loss.train[h0,r,2] <- -2*sum(pred.ma20.train*data.train$data.y[[1]]-log(1+exp(pred.ma20.train)))/size.train[1]
    pred.ma20.test <- cbind(matrix(rep(1,size.test[1]),size.test[1],1), data.test$data.merge[,-1])%*%para.ma20
    kl.loss.test[h0,r,2] <- -2*sum(pred.ma20.test*data.test$data.y-log(1+exp(pred.ma20.test)))/size.test[1]
    
    y.sign.train <- ifelse(exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.ma20)/(1+exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.ma20))>0.5,1,0)
    er.train[h0,r,2] <-mean(data.train$data.y[[1]]!=y.sign.train)
    y.sign.test <- ifelse(exp(cbind(1,data.test$data.merge[,-1])%*%para.ma20)/(1+exp(cbind(1,data.test$data.merge[,-1])%*%para.ma20))>0.5,1,0)
    er.test[h0,r,2] <-mean(data.test$data.y!=y.sign.test)
    
    
    ## TLOAP-loocv
    theta_hat <- array(0,dim=c(size.train[1],s.cnt))
    for(nf in 1:size.train[1]){
      train.data.cv <- data.train
      train.data.cv$data.merge[[1]] <- train.data.cv$data.merge[[1]][-nf,]
      est.beta.cv <- matrix(NA, nrow=s.cnt, ncol=p-qz[1])
      glm.tr <- vector(mode='list', length=s.cnt)
      for(j in 1:s.cnt){
        dataglm <- as.data.frame(train.data.cv$data.merge[[j]]); colnames(dataglm) <- c('respon',paste('z',1:(ndf*qz[j]),sep = ''),paste('x',1:(p-qz[j]),sep = ''))
        glm.tr[[j]] <- glm(respon~., data = dataglm, family=family, control=list(maxit=1000))$coefficients
        est.beta.cv[j,] <- glm.tr[[j]][(ndf*qz[j]+2):length(glm.tr[[j]])]
      }
      for(j in 1:s.cnt){
        beta.est.train.mat.cv <- as.matrix(c(glm.tr[[1]][1], glm.tr[[1]][2:(ndf*qz[j]+1)], est.beta.cv[j,]))
        theta_hat[nf,j] <-tar.cv[nf,]%*%beta.est.train.mat.cv
      }
    }
    solve.weightjcv=try(solnp(rep(1/s.cnt,s.cnt), fun = obj_logit, eqfun=eq, eqB=0, LB=rep(0,s.cnt), control=list(trace=0)),silent=TRUE)
    if ('try-error' %in% class(solve.weightjcv)){
      brea_errorjcv=brea_errorjcv+1
      next
    }else{
      if(solve.weightjcv$convergence!=0){
        brea_convejcv=brea_convejcv+1
        next
      } 
    }
    weight.estjcv <- solve.weightjcv$par
    beta.est.trainjcv <- beta.est.train.mat%*%weight.estjcv
    para.majcv <- c(gvcm.res[[1]]$coefficients[1:(ndf*qz[1]+1)], beta.est.trainjcv)
    
    pred.majcv.train <- cbind(matrix(rep(1,size.train[1]),size.train[1],1), data.train$data.merge[[1]][,-1])%*%para.majcv
    kl.loss.train[h0,r,3] <- -2*sum(pred.majcv.train*data.train$data.y[[1]]-log(1+exp(pred.majcv.train)))/size.train[1]
    pred.majcv.test <- cbind(matrix(rep(1,size.test[1]),size.test[1],1), data.test$data.merge[,-1])%*%para.majcv
    kl.loss.test[h0,r,3] <- -2*sum(pred.majcv.test*data.test$data.y-log(1+exp(pred.majcv.test)))/size.test[1]
    
    y.sign.train <- ifelse(exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.majcv)/(1+exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.majcv))>0.5,1,0)
    er.train[h0,r,3] <-mean(data.train$data.y[[1]]!=y.sign.train)
    y.sign.test <- ifelse(exp(cbind(1,data.test$data.merge[,-1])%*%para.majcv)/(1+exp(cbind(1,data.test$data.merge[,-1])%*%para.majcv))>0.5,1,0)
    er.test[h0,r,3] <-mean(data.test$data.y!=y.sign.test)
    
    
    ####### TLAP-EW #######
    beta.simp <- beta.est.train.mat%*%matrix(1/s.cnt,s.cnt,1)
    para.simp <- c(gvcm.res[[1]]$coefficients[1:(ndf*qz[1]+1)], beta.simp)
    pred.sim.train <- cbind(matrix(rep(1,size.train[1]),size.train[1],1), data.train$data.merge[[1]][,-1])%*%para.simp
    kl.loss.train[h0,r,4] <- -2*sum(pred.sim.train*data.train$data.y[[1]]-log(1+exp(pred.sim.train)))/size.train[1]
    pred.sim.test <- cbind(matrix(rep(1,size.test[1]),size.test[1],1), data.test$data.merge[,-1])%*%para.simp
    kl.loss.test[h0,r,4] <- -2*sum(pred.sim.test*data.test$data.y-log(1+exp(pred.sim.test)))/size.test[1]
    
    y.sign.train <- ifelse(exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.simp)/(1+exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.simp))>0.5,1,0)
    er.train[h0,r,4] <-mean(data.train$data.y[[1]]!=y.sign.train)
    y.sign.test <- ifelse(exp(cbind(1,data.test$data.merge[,-1])%*%para.simp)/(1+exp(cbind(1,data.test$data.merge[,-1])%*%para.simp))>0.5,1,0)
    er.test[h0,r,4] <-mean(data.test$data.y!=y.sign.test)
    
    
    ###### MLE-Target #######
    para.tar <- gvcm.res[[1]]$coefficients
    pred.tar.train <- cbind(matrix(rep(1,size.train[1]),size.train[1],1), data.train$data.merge[[1]][,-1])%*%para.tar
    kl.loss.train[h0,r,5] <- -2*sum(pred.tar.train*data.train$data.y[[1]]-log(1+exp(pred.tar.train)))/size.train[1]
    pred.tar.test <- cbind(matrix(rep(1,size.test[1]),size.test[1],1), data.test$data.merge[,-1])%*%para.tar
    kl.loss.test[h0,r,5] <- -2*sum(pred.tar.test*data.test$data.y-log(1+exp(pred.tar.test)))/size.test[1]
    
    y.sign.train <- ifelse(exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.tar)/(1+exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.tar))>0.5,1,0)
    er.train[h0,r,5] <-mean(data.train$data.y[[1]]!=y.sign.train)
    y.sign.test <- ifelse(exp(cbind(1,data.test$data.merge[,-1])%*%para.tar)/(1+exp(cbind(1,data.test$data.merge[,-1])%*%para.tar))>0.5,1,0)
    er.test[h0,r,5] <-mean(data.test$data.y!=y.sign.test)
    
    
    ###### MLE-Pooled #######
    datax.all <- do.call(rbind, data.train$data.x)
    datau.all <- do.call(rbind, data.train$data.u)
    bsz.tar.te.all <-matrix(NA,size.test, ndf*ncol(datau.all))
    bsz.tar.all <- matrix(NA, sum(size.train), ndf*ncol(datau.all))
    bsz.tar.all[,1:ndf] <- datax.all[,non.ind]*bs(datau.all, df=ndf, degree = ndg)
    bsz.tar.te.all[,1:ndf] <- data.test$data.x[,non.ind]*predict(bs(datau.all, df=ndf, degree = ndg), data.test$data.u)
    datay.all <- as.vector(do.call(rbind, data.train$data.y))
    data.merge.all <- as.data.frame(cbind(datay.all, bsz.tar.all, datax.all[,-non.ind])); colnames(data.merge.all) <- c('respon',paste('bz',1:ncol(bsz.tar.all),sep = ''), paste('x',1:(p-ncol(datau.all)),sep = ''))
    gam.res.all <- glm(respon~., data=data.merge.all, family=family, control=list(maxit=1000))
    beta.est.all <- gam.res.all$coefficients[(2+ndf*ncol(datau.all)):length(gam.res.all$coefficients)]
    para.all <- gam.res.all$coefficients
    
    pred.all.train <- cbind(matrix(rep(1,size.train[1]),size.train[1],1), data.train$data.merge[[1]][,-1])%*%para.all
    kl.loss.train[h0,r,6] <- -2*sum(pred.all.train*data.train$data.y[[1]]-log(1+exp(pred.all.train)))/size.train[1]
    pred.all.test <- cbind(matrix(rep(1,size.test[1]),size.test[1],1), data.test$data.merge[,-1])%*%para.all
    kl.loss.test[h0,r,6] <- -2*sum(pred.all.test*data.test$data.y-log(1+exp(pred.all.test)))/size.test[1]
    
    y.sign.train <- ifelse(exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.all)/(1+exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.all))>0.5,1,0)
    er.train[h0,r,6] <-mean(data.train$data.y[[1]]!=y.sign.train)
    y.sign.test <- ifelse(exp(cbind(1,data.test$data.merge[,-1])%*%para.all)/(1+exp(cbind(1,data.test$data.merge[,-1])%*%para.all))>0.5,1,0)
    er.test[h0,r,6] <-mean(data.test$data.y!=y.sign.test)
    
    
    ###### Trans-GLM #######
    target.data <- list(x=cbind(data.train$data.x[[1]],data.train$data.u[[1]]), y=data.train$data.y[[1]])
    source.data <- vector(mode='list', length=s.cnt-1)
    for(j in 1:(s.cnt-1)){
      source.data[[j]] <- list(x=cbind(data.train$data.x[[j+1]],data.train$data.u[[j+1]]), y=data.train$data.y[[j+1]])
    }
    data.train.adj <- data.train
    data.train.adj$data.merge[[1]] <- cbind(data.train.adj$data.y[[1]], data.train.adj$data.x[[1]], data.train.adj$data.u[[1]])
    data.test.adj <- data.test
    data.test.adj$data.merge <- cbind(data.test.adj$data.y, data.test.adj$data.x, data.test.adj$data.u)
    glmtrans.tian <- glmtrans(target = target.data, source = source.data, family = family, transfer.source.id = "auto", detection.info = FALSE)
    para.tian <- glmtrans.tian$beta
    
    pred.tian.train <- cbind(matrix(rep(1,size.train[1]),size.train[1],1), data.train.adj$data.merge[[1]][,-1])%*%para.tian
    kl.loss.train[h0,r,7] <- -2*sum(pred.tian.train*data.train.adj$data.y[[1]]-log(1+exp(pred.tian.train)))/size.train[1]
    pred.tian.test <- cbind(matrix(rep(1,size.test[1]),size.test[1],1), data.test.adj$data.merge[,-1])%*%para.tian
    kl.loss.test[h0,r,7] <- -2*sum(pred.tian.test*data.test.adj$data.y-log(1+exp(pred.tian.test)))/size.test[1]
    
    y.sign.train <- ifelse(exp(cbind(1,data.train.adj$data.merge[[1]][,-1])%*%para.tian)/(1+exp(cbind(1,data.train.adj$data.merge[[1]][,-1])%*%para.tian))>0.5,1,0)
    er.train[h0,r,7] <-mean(data.train.adj$data.y[[1]]!=y.sign.train)
    y.sign.test <- ifelse(exp(cbind(1,data.test.adj$data.merge[,-1])%*%para.tian)/(1+exp(cbind(1,data.test.adj$data.merge[,-1])%*%para.tian))>0.5,1,0)
    er.test[h0,r,7] <-mean(data.test.adj$data.y!=y.sign.test)
    
    
    ###### ART #######
    source.data.x <- vector(mode='list', length=s.cnt-1)
    source.data.y <- vector(mode='list', length=s.cnt-1)
    for(j in 1:(s.cnt-1)){
      source.data.x[[j]] <- cbind(data.train$data.x[[j+1]], data.train$data.u[[j+1]])
      source.data.y[[j]] <- data.train$data.y[[j+1]]
    }
    fit.art.log <- ART(target.data$x, target.data$y, source.data.x, source.data.y, cbind(data.test$data.x,data.test$data.u), func=fit_logit, type="classification")
    beta.art.log <- fit.art.log$coef_ART[(qz[1]+2):(p+1)]
    para.all.artlog <- fit.art.log$coef_ART
    
    pred.art.train <- cbind(matrix(rep(1,size.train[1]),size.train[1],1), data.train.adj$data.merge[[1]][,-1])%*%para.all.artlog
    kl.loss.train[h0,r,8] <- -2*sum(pred.art.train*data.train.adj$data.y[[1]]-log(1+exp(pred.art.train)))/size.train[1]
    pred.art.test <- cbind(matrix(rep(1,size.test[1]),size.test[1],1), data.test.adj$data.merge[,-1])%*%para.all.artlog
    kl.loss.test[h0,r,8] <- -2*sum(pred.art.test*data.test.adj$data.y-log(1+exp(pred.art.test)))/size.test[1]
    
    y.sign.train <- ifelse(exp(cbind(1,data.train.adj$data.merge[[1]][,-1])%*%para.all.artlog)/(1+exp(cbind(1,data.train.adj$data.merge[[1]][,-1])%*%para.all.artlog))>0.5,1,0)
    er.train[h0,r,8] <-mean(data.train.adj$data.y[[1]]!=y.sign.train)
    y.sign.test <- ifelse(exp(cbind(1,data.test.adj$data.merge[,-1])%*%para.all.artlog)/(1+exp(cbind(1,data.test.adj$data.merge[,-1])%*%para.all.artlog))>0.5,1,0)
    er.test[h0,r,8] <-mean(data.test.adj$data.y!=y.sign.test)
    
    
    if(any(kl.loss.train[h0,r,]==Inf, na.rm = TRUE)){
      # stop("Check...")
      brea_abn=brea_abn+1
      next
    }
    
  }
  
}

###### MPE
kl_loss_fig_train <- NULL
kl_loss_fig_test <- NULL
er_fig_train <- NULL
er_fig_test <- NULL

for(hh in 1:length(source.ind)){
  kl_loss_fig_train <- c(kl_loss_fig_train, as.vector(kl.loss.train[hh,,]))
  kl_loss_fig_test <- c(kl_loss_fig_test, as.vector(kl.loss.test[hh,,]))
  er_fig_train <- c(er_fig_train, as.vector(er.train[hh,,]))
  er_fig_test <- c(er_fig_test, as.vector(er.test[hh,,]))
}

method_name = rep(c(rep('TLOAP-CV10', rep), rep('TLOAP-CV5', rep),
                    rep('TLOAP-LooCV', rep),rep('TLAP-EW', rep), 
                    rep('MLE-Target', rep), rep('MLE-Pooled', rep),
                    rep('Trans-GLM', rep), rep('ART', rep)), length(source.ind))
legend_name = c(rep("20-30",rep*8), rep("30-50",rep*8), rep("50+",rep*8))

data_total_kl_train = data.frame(kl_loss=kl_loss_fig_train, Methods=method_name, Target = legend_name)
data_total_kl_train$Methods <- factor(data_total_kl_train$Methods, levels = c("TLOAP-CV10", "TLOAP-CV5", "TLOAP-LooCV", "TLAP-EW", "MLE-Target", "MLE-Pooled", "Trans-GLM","ART"))
data_total_kl_train$Target <- factor(data_total_kl_train$Target, levels = c("20-30", "30-50", "50+"))

data_total_kl_test = data.frame(kl_loss=kl_loss_fig_test, Methods=method_name, Target = legend_name)
data_total_kl_test$Methods <- factor(data_total_kl_test$Methods, levels = c("TLOAP-CV10", "TLOAP-CV5", "TLOAP-LooCV", "TLAP-EW", "MLE-Target", "MLE-Pooled", "Trans-GLM","ART"))
data_total_kl_test$Target <- factor(data_total_kl_test$Target, levels = c("20-30", "30-50", "50+"))

data_total_er_train = data.frame(er=er_fig_train, Methods=method_name, Target = legend_name)
data_total_er_train$Methods <- factor(data_total_er_train$Methods, levels = c("TLOAP-CV10", "TLOAP-CV5", "TLOAP-LooCV", "TLAP-EW", "MLE-Target", "MLE-Pooled", "Trans-GLM","ART"))
data_total_er_train$Target <- factor(data_total_er_train$Target, levels = c("20-30", "30-50", "50+"))

data_total_er_test = data.frame(er=er_fig_test, Methods=method_name, Target = legend_name)
data_total_er_test$Methods <- factor(data_total_er_test$Methods, levels = c("TLOAP-CV10", "TLOAP-CV5", "TLOAP-LooCV", "TLAP-EW", "MLE-Target", "MLE-Pooled", "Trans-GLM","ART"))
data_total_er_test$Target <- factor(data_total_er_test$Target, levels = c("20-30", "30-50", "50+"))

p1 <- ggplot(data=data_total_kl_train, aes(x=Methods, y=kl_loss,color=Methods)) + 
  stat_boxplot(geom = "errorbar",
               width=0.5,
               position = position_dodge(0.6),
               size=1.5) +
  geom_boxplot(width=0.5, position = position_dodge(0.6), outlier.shape = NA, outlier.size = 2, size=1.5, notch=TRUE) +
  ylab('Mean of KL Loss (in sample)') +
  theme_set(theme_bw()) + 
  theme(axis.ticks.x = element_blank(), 
        axis.text.x = element_blank(),
        panel.grid =element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size=40),
        axis.text.y = element_text(size=40),
        legend.text = element_text(size = 40),
        legend.title = element_blank(),
        legend.key.size = unit(1, "inches"),
        legend.position = "bottom" ,
        legend.box = "horizontal",
        legend.text.align = 0) +
  facet_wrap(~ Target, scales='free_y', nrow = 3) +
  theme(strip.text.x = element_text(size = 40),
        strip.text.y = element_text(size = 40),
        plot.title = element_text(size = 40)) +
  facetted_pos_scales(y = list(Target == "50+" ~
                                 scale_y_continuous(limits=c(0.5,2.5),breaks=seq(0.5,2.5,0.5)),
                               Target == "30-50" ~
                                 scale_y_continuous(limits=c(1,1.8),breaks=seq(1,1.8,0.3))))

p2 <- ggplot(data=data_total_kl_test, aes(x=Methods, y=kl_loss,color=Methods)) + 
  stat_boxplot(geom = "errorbar",
               width=0.5,
               position = position_dodge(0.6),
               size=1.5) +
  geom_boxplot(width=0.5, position = position_dodge(0.6), outlier.shape = NA, outlier.size = 2, size=1.5, notch=TRUE) +
  ylab('Mean of KL Loss (out of sample)') +
  theme_set(theme_bw()) + 
  theme(axis.ticks.x = element_blank(), 
        axis.text.x = element_blank(),
        panel.grid =element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size=40),
        axis.text.y = element_text(size=40),
        legend.text = element_text(size = 40),
        legend.title = element_blank(),
        legend.key.size = unit(1, "inches"),
        legend.position = "bottom" ,
        legend.box = "horizontal",
        legend.text.align = 0) +
  facet_wrap(~ Target, scales='free_y', nrow = 3) +
  theme(strip.text.x = element_text(size = 40),
        strip.text.y = element_text(size = 40),
        plot.title = element_text(size = 40)) +
  facetted_pos_scales(y = list(Target == "50+" ~
                                 scale_y_continuous(limits=c(0,6),breaks=seq(0,6,1)),
                               Target == "30-50" ~
                                 scale_y_continuous(limits=c(0.8,3.5),breaks=seq(0.8,3.5,0.5))))

library(patchwork)
p3 <- p1 + p2 + plot_layout(guides = 'collect') & theme(legend.position = 'bottom')


pp1 <- ggplot(data=data_total_er_train, aes(x=Methods, y=er, color=Methods)) +
  stat_boxplot(geom = "errorbar",
               width=0.5,
               position = position_dodge(0.6),
               size=1.5) +
  geom_boxplot(width=0.5, position = position_dodge(0.6), outlier.shape = NA, outlier.size = 2, size=1.5, notch=TRUE) +
  ylab('Mean of CER (in sample)') +
  theme_set(theme_bw()) +
  theme(axis.ticks.x = element_blank(), 
        axis.text.x = element_blank(),
        panel.grid =element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size=40),
        axis.text.y = element_text(size=40),
        legend.text = element_text(size = 40),
        legend.title = element_blank(),
        legend.key.size = unit(1, "inches"),
        legend.position = "bottom" ,
        legend.box = "horizontal",
        legend.text.align = 0) +
  facet_wrap(~ Target, scales='free_y', nrow = 3) +
  theme(strip.text.x = element_text(size = 40),
        strip.text.y = element_text(size = 40),
        plot.title = element_text(size = 40)) 

pp2 <- ggplot(data=data_total_er_test, aes(x=Methods, y=er, color=Methods)) +
  stat_boxplot(geom = "errorbar",
               width=0.5,
               position = position_dodge(0.6),
               size=1.5) +
  geom_boxplot(width=0.5, position = position_dodge(0.6), outlier.shape = NA, outlier.size = 2, size=1.5, notch=TRUE) +
  ylab('Mean of CER (out of sample)') +
  theme_set(theme_bw()) +
  theme(axis.ticks.x = element_blank(), 
        axis.text.x = element_blank(),
        panel.grid =element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size=40),
        axis.text.y = element_text(size=40),
        legend.text = element_text(size = 40),
        legend.title = element_blank(),
        legend.key.size = unit(1, "inches"),
        legend.position = "bottom" ,
        legend.box = "horizontal",
        legend.text.align = 0) +
  facet_wrap(~ Target, scales='free_y', nrow = 3) +
  theme(strip.text.x = element_text(size = 40),
        strip.text.y = element_text(size = 40),
        plot.title = element_text(size = 40)) 

library(patchwork)
pp3 <- pp1 + pp2 + plot_layout(guides = 'collect') & theme(legend.position = 'bottom')
