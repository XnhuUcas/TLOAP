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

data.raw <- read.csv('train.csv', header = T, encoding = "UTF-8")
data.raw <- data.raw[,-1]
data.raw$person_home_ownership[data.raw$person_home_ownership=="OWN"] = "OTHER" 

var.dummy.names <- c("person_home_ownership", "loan_intent", "cb_person_default_on_file")
var.response <- "loan_status"
var.conti.names <- setdiff(colnames(data.raw),c(var.dummy.names, var.response, "loan_grade"))
data.cont <- data.raw[,var.conti.names]
data.cont$person_age <- (data.cont$person_age-min(data.cont$person_age))/(max(data.cont$person_age)-min(data.cont$person_age))
data.cont[,-1] <- as.data.frame(scale(data.cont[,-1]))

data.dum <- dummyVars(~., data = data.raw[,var.dummy.names], fullRank = T)
data.dum.mat <- data.frame(predict(data.dum, data.raw[,var.dummy.names]))

data.analysis.final <- cbind(data.raw$loan_grade, data.raw$loan_status, data.dum.mat, data.cont)
colnames(data.analysis.final)[1:2] <- c("loan_grade", "response")
x.colnames <- colnames(data.analysis.final)[-c(1,2,11)]
z.colnames <- "person_age"

p <- length(x.colnames)
source.ind <- unique(data.analysis.final$loan_grade)
s.cnt <- length(source.ind)
qz <- rep(1,s.cnt)
family <- 'binomial' 
nfold=5
non.ind <- which(x.colnames=="loan_percent_income") 

## add noise
noise.ind <- which(data.analysis.final$loan_grade==source.ind[6])
data.analysis.final[noise.ind, x.colnames] <- data.analysis.final[noise.ind, x.colnames] + matrix(rnorm(length(noise.ind)*length(x.colnames), mean=1.5, sd = 1), nrow = length(noise.ind), ncol = length(x.colnames))
data.analysis.final[noise.ind, z.colnames] <- data.analysis.final[noise.ind, z.colnames] + matrix(rnorm(length(noise.ind)*length(z.colnames), mean=1.5, sd = 1), nrow = length(noise.ind), ncol = length(z.colnames))

rep <- 100
brea_error10=0
brea_conve10=0
brea_error20=0
brea_conve20=0
brea_errorjcv=0
brea_convejcv=0
brea_abn=0
options(warn = -1)

kl.loss.train <- array(NA,dim=c(s.cnt, rep, 7))
kl.loss.test <- array(NA,dim=c(s.cnt, rep, 7))
er.train <- array(NA,dim=c(s.cnt, rep, 7))
er.test <- array(NA,dim=c(s.cnt, rep, 7))

############## Running 

for(h0 in 5:6){ 
  
  data.x.list <- vector(mode = 'list', length = length(source.ind))
  data.x.list[[1]] <- as.matrix(data.analysis.final[which(data.analysis.final$loan_grade==source.ind[h0]), x.colnames])
  data.u.list <- vector(mode = 'list', length = length(source.ind))
  data.u.list[[1]] <- as.matrix(data.analysis.final[which(data.analysis.final$loan_grade==source.ind[h0]), z.colnames])
  data.y.list <- vector(mode = 'list', length = length(source.ind))
  data.y.list[[1]] <- as.matrix(data.analysis.final[which(data.analysis.final$loan_grade==source.ind[h0]), 'response'])
  source.ind.del <- source.ind[-which(source.ind==source.ind[h0])]
  for(i in 1:length(source.ind.del)){
    row.ind <- which(data.analysis.final$loan_grade==source.ind.del[i])
    data.x.list[[i+1]] <- as.matrix(data.analysis.final[row.ind, x.colnames])
    data.u.list[[i+1]] <- as.matrix(data.analysis.final[row.ind, z.colnames])
    data.y.list[[i+1]] <- as.matrix(data.analysis.final[row.ind, 'response'])
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
    
    
    ####### TLAP-EW #######
    beta.simp <- beta.est.train.mat%*%matrix(1/s.cnt,s.cnt,1)
    para.simp <- c(gvcm.res[[1]]$coefficients[1:(ndf*qz[1]+1)], beta.simp)
    pred.sim.train <- cbind(matrix(rep(1,size.train[1]),size.train[1],1), data.train$data.merge[[1]][,-1])%*%para.simp
    kl.loss.train[h0,r,3] <- -2*sum(pred.sim.train*data.train$data.y[[1]]-log(1+exp(pred.sim.train)))/size.train[1]
    pred.sim.test <- cbind(matrix(rep(1,size.test[1]),size.test[1],1), data.test$data.merge[,-1])%*%para.simp
    kl.loss.test[h0,r,3] <- -2*sum(pred.sim.test*data.test$data.y-log(1+exp(pred.sim.test)))/size.test[1]
    
    y.sign.train <- ifelse(exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.simp)/(1+exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.simp))>0.5,1,0)
    er.train[h0,r,3] <-mean(data.train$data.y[[1]]!=y.sign.train)
    y.sign.test <- ifelse(exp(cbind(1,data.test$data.merge[,-1])%*%para.simp)/(1+exp(cbind(1,data.test$data.merge[,-1])%*%para.simp))>0.5,1,0)
    er.test[h0,r,3] <-mean(data.test$data.y!=y.sign.test)
    
    
    ###### MLE-Target #######
    para.tar <- gvcm.res[[1]]$coefficients
    pred.tar.train <- cbind(matrix(rep(1,size.train[1]),size.train[1],1), data.train$data.merge[[1]][,-1])%*%para.tar
    kl.loss.train[h0,r,4] <- -2*sum(pred.tar.train*data.train$data.y[[1]]-log(1+exp(pred.tar.train)))/size.train[1]
    pred.tar.test <- cbind(matrix(rep(1,size.test[1]),size.test[1],1), data.test$data.merge[,-1])%*%para.tar
    kl.loss.test[h0,r,4] <- -2*sum(pred.tar.test*data.test$data.y-log(1+exp(pred.tar.test)))/size.test[1]
    
    y.sign.train <- ifelse(exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.tar)/(1+exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.tar))>0.5,1,0)
    er.train[h0,r,4] <-mean(data.train$data.y[[1]]!=y.sign.train)
    y.sign.test <- ifelse(exp(cbind(1,data.test$data.merge[,-1])%*%para.tar)/(1+exp(cbind(1,data.test$data.merge[,-1])%*%para.tar))>0.5,1,0)
    er.test[h0,r,4] <-mean(data.test$data.y!=y.sign.test)
    
    
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
    kl.loss.train[h0,r,5] <- -2*sum(pred.all.train*data.train$data.y[[1]]-log(1+exp(pred.all.train)))/size.train[1]
    pred.all.test <- cbind(matrix(rep(1,size.test[1]),size.test[1],1), data.test$data.merge[,-1])%*%para.all
    kl.loss.test[h0,r,5] <- -2*sum(pred.all.test*data.test$data.y-log(1+exp(pred.all.test)))/size.test[1]
    
    y.sign.train <- ifelse(exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.all)/(1+exp(cbind(1,data.train$data.merge[[1]][,-1])%*%para.all))>0.5,1,0)
    er.train[h0,r,5] <-mean(data.train$data.y[[1]]!=y.sign.train)
    y.sign.test <- ifelse(exp(cbind(1,data.test$data.merge[,-1])%*%para.all)/(1+exp(cbind(1,data.test$data.merge[,-1])%*%para.all))>0.5,1,0)
    er.test[h0,r,5] <-mean(data.test$data.y!=y.sign.test)
    
    
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
    kl.loss.train[h0,r,6] <- -2*sum(pred.tian.train*data.train.adj$data.y[[1]]-log(1+exp(pred.tian.train)))/size.train[1]
    pred.tian.test <- cbind(matrix(rep(1,size.test[1]),size.test[1],1), data.test.adj$data.merge[,-1])%*%para.tian
    kl.loss.test[h0,r,6] <- -2*sum(pred.tian.test*data.test.adj$data.y-log(1+exp(pred.tian.test)))/size.test[1]
    
    y.sign.train <- ifelse(exp(cbind(1,data.train.adj$data.merge[[1]][,-1])%*%para.tian)/(1+exp(cbind(1,data.train.adj$data.merge[[1]][,-1])%*%para.tian))>0.5,1,0)
    er.train[h0,r,6] <-mean(data.train.adj$data.y[[1]]!=y.sign.train)
    y.sign.test <- ifelse(exp(cbind(1,data.test.adj$data.merge[,-1])%*%para.tian)/(1+exp(cbind(1,data.test.adj$data.merge[,-1])%*%para.tian))>0.5,1,0)
    er.test[h0,r,6] <-mean(data.test.adj$data.y!=y.sign.test)
    
    
    ###### ART #######
    source.data.x <- vector(mode='list', length=s.cnt-1)
    source.data.y <- vector(mode='list', length=s.cnt-1)
    for(j in 1:(s.cnt-1)){
      source.data.x[[j]] <- cbind(data.train$data.x[[j+1]], data.train$data.u[[j+1]])
      source.data.y[[j]] <- data.train$data.y[[j+1]]
    }
    fit.art.log <- try(ART(target.data$x, target.data$y, source.data.x, source.data.y, cbind(data.test$data.x,data.test$data.u), func=fit_logit, type="classification"),silent=TRUE)
    if ('try-error' %in% class(fit.art.log)){
      next
    }else{
      beta.art.log <- fit.art.log$coef_ART[(qz[1]+2):(p+1)]
      para.all.artlog <- fit.art.log$coef_ART
      pred.art.train <- cbind(matrix(rep(1,size.train[1]),size.train[1],1), data.train.adj$data.merge[[1]][,-1])%*%para.all.artlog
      kl.loss.train[h0,r,7] <- -2*sum(pred.art.train*data.train.adj$data.y[[1]]-log(1+exp(pred.art.train)))/size.train[1]
      pred.art.test <- cbind(matrix(rep(1,size.test[1]),size.test[1],1), data.test.adj$data.merge[,-1])%*%para.all.artlog
      kl.loss.test[h0,r,7] <- -2*sum(pred.art.test*data.test.adj$data.y-log(1+exp(pred.art.test)))/size.test[1]
      
      y.sign.train <- ifelse(exp(cbind(1,data.train.adj$data.merge[[1]][,-1])%*%para.all.artlog)/(1+exp(cbind(1,data.train.adj$data.merge[[1]][,-1])%*%para.all.artlog))>0.5,1,0)
      er.train[h0,r,7] <-mean(data.train.adj$data.y[[1]]!=y.sign.train)
      y.sign.test <- ifelse(exp(cbind(1,data.test.adj$data.merge[,-1])%*%para.all.artlog)/(1+exp(cbind(1,data.test.adj$data.merge[,-1])%*%para.all.artlog))>0.5,1,0)
      er.test[h0,r,7] <-mean(data.test.adj$data.y!=y.sign.test)
    }
  }
  
}


