rm(list=ls())
gc()
source("Data_Gen.R")
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
  if(family.opt=='binomial'){
    alpha2=sum(log(1+exp(alpha2)))
  }else if(family.opt=='poisson'){
    alpha2=sum(exp(alpha2))
  }else if(family.opt=='gaussian'){
    alpha2=sum(alpha2^2/2)
  }
  return(alpha2-alpha1)
}

s.cnt <- 7
qz <- rep(2,s.cnt)
size.cnd <- c(100, 300, 800)

smooth.fun.true1 <- function(u){
  return(cbind(cos(2*pi*u[,1]), sin(2*pi*u[,2])))
}
smooth.fun.true2 <- function(u){
  return(cbind(cos(2*pi*u[,1]), sin(4*pi*u[,2])))
}
smooth.fun.true3 <- function(u){
  return(cbind(sin(2*pi*u[,2]), cos(4*pi*u[,2])))
}
smooth.fun.true4 <- function(u){
  return(cbind(sin(6*pi*u[,1]), sin(2*pi*u[,2])))
}
smooth.fun.true5 <- function(u){
  return(cbind(cos(4*pi*u[,1]),cos(2*pi*u[,2])))
}
smooth.fun.true6 <- function(u){
  return(cbind(cos(6*pi*u[,1]), sin(6*pi*u[,2])))
}
smooth.fun.true7 <- function(u){
  return(cbind(sin(6*pi*u[,1]), sin(4*pi*u[,2])))
}

err.sigma <- 0.5
rho <- 0.5 
family <- c("gaussian", "binomial", "poisson")
rep <- 500
size.test <- 100
theta.src <- 1.8
theta.tar <- 1.2
mis.source.index <- c(2,3,4)
brea_error10=0
brea_conve10=0
brea_error20=0
brea_conve20=0
brea_errorjcv=0
brea_convejcv=0
brea_abn=0
options(warn = -1)

oos.kl.loss <- array(NA,dim=c(length(size.cnd), length(family), rep, 9))
est.error <- array(NA,dim=c(length(size.cnd),length(family), rep, 9))
me <- array(NA,dim=c(length(size.cnd),length(family), rep, 9))
me.bias <- array(NA,dim=c(length(size.cnd),length(family), rep, 9))
me.var <- array(NA,dim=c(length(size.cnd),length(family), rep, 9))
mape.is <- array(NA,dim=c(length(size.cnd),length(family), rep, 9))
mape.oos <- array(NA,dim=c(length(size.cnd),length(family), rep, 9))
er.is <- array(NA,dim=c(length(size.cnd),length(family), rep, 9))
er.oos <- array(NA,dim=c(length(size.cnd),length(family), rep, 9))
weight.mat <- array(NA,dim=c(length(size.cnd), length(family), rep, s.cnt))
weight.mat20 <- array(NA,dim=c(length(size.cnd), length(family), rep, s.cnt))
weight.matjcv <- array(NA,dim=c(length(size.cnd), length(family), rep, s.cnt))
time.consume <- array(NA,dim=c(length(size.cnd),length(family), rep, 9))

for(i0 in 1:length(size.cnd)){
  size <- rep(size.cnd[i0], s.cnt)
  if(size.cnd[i0]==100){
    para.true <- cbind(as.matrix(c(0.5,0.2,-0.2)),
                       as.matrix(c(0.5,0.2,-0.2)+0.02),
                       as.matrix(c(0.5,0.2,-0.2)+0.02),
                       as.matrix(c(0.5,0.2,-0.2)+0.3),
                       as.matrix(c(0.5,0.2,-0.2)),
                       as.matrix(c(0.5,0.2,-0.2)+0.02),
                       as.matrix(c(0.5,0.2,-0.2)+0.3))
  }
  if(size.cnd[i0]==300){
    para.true <- cbind(as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)),
                       as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)+0.02),
                       as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)+0.02),
                       as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)+0.3),
                       as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)),
                       as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)+0.02),
                       as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)+0.3))
  }
  if(size.cnd[i0]==800){
    para.true <- cbind(as.matrix(c(0.5,0.2,-0.2,-0.3,0.5,0.5,0.2)),
                       as.matrix(c(0.5,0.2,-0.2,-0.3,0.5,0.5,0.2)+0.02),
                       as.matrix(c(0.5,0.2,-0.2,-0.3,0.5,0.5,0.2)+0.02),
                       as.matrix(c(0.5,0.2,-0.2,-0.3,0.5,0.5,0.2)+0.3),
                       as.matrix(c(0.5,0.2,-0.2,-0.3,0.5,0.5,0.2)),
                       as.matrix(c(0.5,0.2,-0.2,-0.3,0.5,0.5,0.2)+0.02),
                       as.matrix(c(0.5,0.2,-0.2,-0.3,0.5,0.5,0.2)+0.3))
  }
  p <- floor(size[1]^0.3)+qz[1]
  ndg <- 3
  nknots <- ceiling(size[1]^(1/5))
  ndf <- ndg+nknots
  
  for(i in 1:length(family)){
    family.opt <- family[i]
    for(r in 1:rep){
      
      set.seed(20230327+50*i0+100*r+200*i)
      cat('---------------------------------\n')
      cat('Now sample size is', size.cnd[i0], ', family is', family[i],'in the', r, 'th replicate..\n')
      cat('---------------------------------\n')
      
      ## generate all datasets
      data.all <- Data_Gen(family = family[i], 
                           type = 'mis', 
                           s.cnt = s.cnt, 
                           size = size, 
                           size.test = size.test, 
                           para.true = para.true, 
                           theta.tar = theta.tar,
                           theta.src = theta.src, 
                           err.sigma = err.sigma, 
                           p = p, 
                           qz = qz, 
                           mis.source.index = mis.source.index)
      data.train <- data.all$train_data
      data.test <- data.all$test_data
      
      ###### TLOAP cv10
      timeSp=Sys.time()
      bsz.tar.te=array(0,dim=c(size.test,ndf*qz[1]))
      for(h in 1:s.cnt){
        bsz.tar <- array(0,dim=c(size[h], ndf*qz[h]))
        if(h==1){
          for(j in 1:qz[h]){
            bsz.tar[,((j-1)*ndf+1):(j*ndf)] <- data.train$data.x[[h]][,j]*bs(data.train$data.u[[h]][,j], df=ndf, degree = ndg)
            bsz.tar.te[,((j-1)*ndf+1):(j*ndf)] <- data.test$data.x[,j]*predict(bs(data.train$data.u[[h]][,j], df=ndf, degree = ndg), data.test$data.u[,j])
          }
        }else{
          for(j in 1:qz[h]){
            bsz.tar[,((j-1)*ndf+1):(j*ndf)] <- data.train$data.x[[h]][,j]*bs(data.train$data.u[[h]][,j], df=ndf, degree = ndg)
          }
        }
        data.train$data.merge[[h]] <- cbind(data.train$data.y[[h]], bsz.tar, data.train$data.x[[h]][,(qz[h]+1):p])
      }
      data.test$data.merge <- cbind(data.test$data.y, bsz.tar.te, data.test$data.x[,(qz[1]+1):p])
      time.consume.sp=as.double(difftime(Sys.time(), timeSp, units="secs"))
      
      timeES=Sys.time()
      n_group_10 <- {if((size[1]%%10)== 0) size[1]%/%10 else size[1]%/%10+1} 
      theta_hat <- array(0,dim=c(size[1],s.cnt))
      tar.cv <- cbind(1,data.train$data.merge[[1]][,-1])
      for(nf in 1:n_group_10){
        split.ind <- (1:size[1])[((nf-1)*10+1):min(nf*10,size[1])]
        train.data.cv <- data.train
        train.data.cv$data.merge[[1]] <- train.data.cv$data.merge[[1]][-split.ind,]
        est.beta.cv <- matrix(NA, nrow=s.cnt, ncol=p-qz[j])
        glm.tr <- vector(mode='list', length=s.cnt)
        for(j in 1:s.cnt){
          dataglm <- as.data.frame(train.data.cv$data.merge[[j]]); colnames(dataglm) <- c('respon',paste('z',1:(ndf*qz[j]),sep = ''),paste('x',1:(p-qz[j]),sep = ''))
          glm.tr[[j]] <- glm(respon~., data = dataglm, family=family[i], control=list(maxit=1000))$coefficients
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
      time.consume[i0,i,r,1]=as.double(difftime(Sys.time(), timeES, units="secs"))+time.consume.sp
      
      beta.est.train.mat <- matrix(NA, p-qz[1], s.cnt)
      gvcm.res <- vector(mode='list', length=s.cnt)
      for(k in 1:s.cnt){
        data.train.frame = as.data.frame(data.train$data.merge[[k]]); colnames(data.train.frame) <- c('respon',paste('z',1:(ndf*qz[k]),sep = ''),paste('x',1:(p-qz[k]),sep = ''))
        gvcm.res[[k]] <- glm(respon~., data = data.train.frame, family=family[i], control=list(maxit=1000))
        beta.est.train.mat[,k] <- gvcm.res[[k]]$coefficients[(ndf*qz[k]+2):length(gvcm.res[[k]]$coefficients)]
      }
      weight.est <- solve.weight$par
      beta.est.train <- beta.est.train.mat%*%weight.est
      para.all <- c(gvcm.res[[1]]$coefficients[1:(ndf*qz[1]+1)], beta.est.train)
      res.summary <- Performance_Eva(train.data = data.train, test.data = data.test,  beta.est.train = beta.est.train, para.all = para.all, family = family[i])
      oos.kl.loss[i0,i,r,1] <- res.summary$oos.kl.loss
      est.error[i0,i,r,1] <- res.summary$est.error
      weight.mat[i0,i,r,] <- weight.est
      me[i0,i,r,1] <- res.summary$model.error
      me.bias[i0,i,r,1] <- res.summary$bias.me
      me.var[i0,i,r,1] <- res.summary$var.me
      mape.is[i0,i,r,1] <- res.summary$mape.is
      mape.oos[i0,i,r,1] <- res.summary$mape.oos
      er.is[i0,i,r,1] <- res.summary$er.is
      er.oos[i0,i,r,1] <- res.summary$er.oos
      
      
      ## TLOAP-cv5
      timeES20=Sys.time()
      n_group_5 <- {if((size[1]%%5)== 0) size[1]%/%5 else size[1]%/%5+1} 
      theta_hat <- array(0,dim=c(size[1],s.cnt))
      for(nf in 1:n_group_5){
        split.ind <- (1:size[1])[((nf-1)*5+1):min(nf*5,size[1])]
        train.data.cv <- data.train
        train.data.cv$data.merge[[1]] <- train.data.cv$data.merge[[1]][-split.ind,]
        est.beta.cv <- matrix(NA, nrow=s.cnt, ncol=p-qz[1])
        glm.tr <- vector(mode='list', length=s.cnt)
        for(j in 1:s.cnt){
          dataglm <- as.data.frame(train.data.cv$data.merge[[j]]); colnames(dataglm) <- c('respon',paste('z',1:(ndf*qz[j]),sep = ''),paste('x',1:(p-qz[j]),sep = ''))
          glm.tr[[j]] <- glm(respon~., data = dataglm, family=family[i], control=list(maxit=1000))$coefficients
          est.beta.cv[j,] <- glm.tr[[j]][(ndf*qz[j]+2):length(glm.tr[[j]])]
        }
        for(j in 1:s.cnt){
          beta.est.train.mat.cv <- as.matrix(c(glm.tr[[1]][1], glm.tr[[1]][2:(ndf*qz[j]+1)], est.beta.cv[j,]))
          theta_hat[split.ind,j] <-tar.cv[split.ind,]%*%beta.est.train.mat.cv
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
      time.consume[i0,i,r,2]=as.double(difftime(Sys.time(), timeES20, units="secs"))+time.consume.sp
      weight.est20 <- solve.weight20$par
      beta.est.train20 <- beta.est.train.mat%*%weight.est20
      para.all.ma20 <- c(gvcm.res[[1]]$coefficients[1:(ndf*qz[1]+1)], beta.est.train20)
      res.summary20 <- Performance_Eva(train.data = data.train, test.data = data.test,  beta.est.train = beta.est.train20, para.all = para.all.ma20, family = family[i])
      oos.kl.loss[i0,i,r,2] <- res.summary20$oos.kl.loss
      est.error[i0,i,r,2] <- res.summary20$est.error
      weight.mat20[i0,i,r,] <- weight.est20
      me[i0,i,r,2] <- res.summary20$model.error
      me.bias[i0,i,r,2] <- res.summary20$bias.me
      me.var[i0,i,r,2] <- res.summary20$var.me
      mape.is[i0,i,r,2] <- res.summary20$mape.is
      mape.oos[i0,i,r,2] <- res.summary20$mape.oos
      er.is[i0,i,r,2] <- res.summary20$er.is
      er.oos[i0,i,r,2] <- res.summary20$er.oos
      
      
      ## TLOAP-loocv
      timeESn=Sys.time()
      theta_hat <- array(0,dim=c(size[1],s.cnt))
      for(nf in 1:size[1]){
        train.data.cv <- data.train
        train.data.cv$data.merge[[1]] <- train.data.cv$data.merge[[1]][-nf,]
        est.beta.cv <- matrix(NA, nrow=s.cnt, ncol=p-qz[1])
        glm.tr <- vector(mode='list', length=s.cnt)
        for(j in 1:s.cnt){
          dataglm <- as.data.frame(train.data.cv$data.merge[[j]]); colnames(dataglm) <- c('respon',paste('z',1:(ndf*qz[j]),sep = ''),paste('x',1:(p-qz[j]),sep = ''))
          glm.tr[[j]] <- glm(respon~., data = dataglm, family=family[i], control=list(maxit=1000))$coefficients
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
      time.consume[i0,i,r,3]=as.double(difftime(Sys.time(), timeESn, units="secs"))+time.consume.sp
      weight.estjcv <- solve.weightjcv$par
      beta.est.trainjcv <- beta.est.train.mat%*%weight.estjcv
      para.all.majcv <- c(gvcm.res[[1]]$coefficients[1:(ndf*qz[1]+1)], beta.est.trainjcv)
      res.summaryjcv <- Performance_Eva(train.data = data.train, test.data = data.test,  beta.est.train = beta.est.trainjcv, para.all = para.all.majcv, family = family[i])
      oos.kl.loss[i0,i,r,3] <- res.summaryjcv$oos.kl.loss
      est.error[i0,i,r,3] <- res.summaryjcv$est.error
      weight.matjcv[i0,i,r,] <- weight.estjcv
      me[i0,i,r,3] <- res.summaryjcv$model.error
      me.bias[i0,i,r,3] <- res.summaryjcv$bias.me
      me.var[i0,i,r,3] <- res.summaryjcv$var.me
      mape.is[i0,i,r,3] <- res.summaryjcv$mape.is
      mape.oos[i0,i,r,3] <- res.summaryjcv$mape.oos
      er.is[i0,i,r,3] <- res.summaryjcv$er.is
      er.oos[i0,i,r,3] <- res.summaryjcv$er.oos
      
      
      ####### TLAP-EW #######
      timeSA=Sys.time()
      ##
      beta.est.train.mat.sa <- matrix(NA, p-qz[1], s.cnt)
      gvcm.res.sa <- vector(mode='list', length=s.cnt)
      for(k in 1:s.cnt){
        data.train.frame.sa = as.data.frame(data.train$data.merge[[k]]); colnames(data.train.frame.sa) <- c('respon',paste('z',1:(ndf*qz[k]),sep = ''),paste('x',1:(p-qz[k]),sep = ''))
        gvcm.res.sa[[k]] <- glm(respon~., data = data.train.frame.sa, family=family[i], control=list(maxit=1000))
        beta.est.train.mat.sa[,k] <- gvcm.res.sa[[k]]$coefficients[(ndf*qz[k]+2):length(gvcm.res.sa[[k]]$coefficients)]
      }
      ##
      beta.simp <- beta.est.train.mat%*%matrix(1/s.cnt,s.cnt,1)
      para.all.simp <- c(gvcm.res[[1]]$coefficients[1:(ndf*qz[1]+1)], beta.simp)
      time.consume[i0,i,r,4]=as.double(difftime(Sys.time(), timeSA, units="secs"))
      res.simp <- Performance_Eva(train.data = data.train, test.data = data.test,  beta.est.train = beta.simp, para.all = para.all.simp, family = family[i])
      oos.kl.loss[i0,i,r,4] <- res.simp$oos.kl.loss
      est.error[i0,i,r,4] <- res.simp$est.error
      me[i0,i,r,4] <- res.simp$model.error
      me.bias[i0,i,r,4] <- res.simp$bias.me
      me.var[i0,i,r,4] <- res.simp$var.me
      mape.is[i0,i,r,4] <- res.simp$mape.is
      mape.oos[i0,i,r,4] <- res.simp$mape.oos
      er.is[i0,i,r,4] <- res.simp$er.is
      er.oos[i0,i,r,4] <- res.simp$er.oos
      
      
      ###### MLE-Target #######
      timeTAR=Sys.time()
      ##
      data.train.frame.tar = as.data.frame(data.train$data.merge[[1]]); colnames(data.train.frame.tar) <- c('respon',paste('z',1:(ndf*qz[1]),sep = ''),paste('x',1:(p-qz[1]),sep = ''))
      gvcm.res.tar <- glm(respon~., data = data.train.frame.tar, family=family[i], control=list(maxit=1000))
      ##
      beta.tar <- beta.est.train.mat[,1]
      para.all.tar <- gvcm.res[[1]]$coefficients
      time.consume[i0,i,r,5]=as.double(difftime(Sys.time(), timeTAR, units="secs"))
      res.tar <- Performance_Eva(train.data = data.train, test.data = data.test,  beta.est.train = beta.tar, para.all = para.all.tar, family = family[i])
      oos.kl.loss[i0,i,r,5] <- res.tar$oos.kl.loss
      est.error[i0,i,r,5] <- res.tar$est.error
      me[i0,i,r,5] <- res.tar$model.error
      me.bias[i0,i,r,5] <- res.tar$bias.me
      me.var[i0,i,r,5] <- res.tar$var.me
      mape.is[i0,i,r,5] <- res.tar$mape.is
      mape.oos[i0,i,r,5] <- res.tar$mape.oos
      er.is[i0,i,r,5] <- res.tar$er.is
      er.oos[i0,i,r,5] <- res.tar$er.oos
      
      
      ###### MLE-Pooled #######
      timePMLE=Sys.time()
      data.train.adj <- data.train
      mis.source.index.new <- c(1,mis.source.index)
      for(ii in 1:length(mis.source.index.new)){
        data.train.adj$data.x[[mis.source.index.new[ii]]] <- data.train.adj$data.x[[mis.source.index.new[ii]]][,-(p+1)]
      }
      datax.all <- do.call(rbind, data.train.adj$data.x)
      datau.all <- do.call(rbind, data.train.adj$data.u)
      bsz.tar.te.all <-matrix(NA,size.test, ndf*ncol(datau.all))
      bsz.tar.all <- matrix(NA, sum(size), ndf*ncol(datau.all))
      for(j in 1:ncol(datau.all)){
        bsz.tar.all[,((j-1)*ndf+1):(j*ndf)] <- datax.all[,j]*bs(datau.all[,j], df=ndf, degree = ndg)
        bsz.tar.te.all[,((j-1)*ndf+1):(j*ndf)] <- data.test$data.x[,j]*predict(bs(datau.all[,j], df=ndf, degree = ndg), data.test$data.u[,j])
      }
      datay.all <- as.vector(do.call(cbind, data.train$data.y))
      data.merge.all <- as.data.frame(cbind(datay.all, bsz.tar.all, datax.all[,(ncol(datau.all)+1):p])); colnames(data.merge.all) <- c('respon',paste('bz',1:ncol(bsz.tar.all),sep = ''), paste('x',1:(p-ncol(datau.all)),sep = ''))
      gam.res.all <- glm(respon~., data=data.merge.all, family=family[i], control=list(maxit=1000))
      beta.est.all <- gam.res.all$coefficients[(2+ndf*ncol(datau.all)):length(gam.res.all$coefficients)]
      para.all.all <- gam.res.all$coefficients
      time.consume[i0,i,r,6]=as.double(difftime(Sys.time(), timePMLE, units="secs"))
      res.pooling <- Performance_Eva(train.data = data.train, test.data = data.test,  beta.est.train = beta.est.all, para.all = para.all.all, family = family[i])
      oos.kl.loss[i0,i,r,6] <- res.pooling$oos.kl.loss
      est.error[i0,i,r,6] <- res.pooling$est.error
      me[i0,i,r,6] <- res.pooling$model.error
      me.bias[i0,i,r,6] <- res.pooling$bias.me
      me.var[i0,i,r,6] <- res.pooling$var.me
      mape.is[i0,i,r,6] <- res.pooling$mape.is
      mape.oos[i0,i,r,6] <- res.pooling$mape.oos
      er.is[i0,i,r,6] <- res.pooling$er.is
      er.oos[i0,i,r,6] <- res.pooling$er.oos
      
      
      ###### Trans-GLM #######
      timeTransGLM=Sys.time()
      target.data <- list(x=data.train.adj$data.x[[1]], y=data.train.adj$data.y[[1]])
      source.data <- vector(mode='list', length=s.cnt-1)
      for(j in 1:(s.cnt-1)){
        source.data[[j]] <- list(x=data.train.adj$data.x[[j+1]], y=data.train.adj$data.y[[j+1]])
      }
      data.test$data.merge <- cbind(data.test$data.y, data.test$data.x[,1:p])
      data.train$data.merge[[1]] <- cbind(data.train$data.y[[1]], data.train$data.x[[1]][,1:p])
      glmtrans.tian <- glmtrans(target = target.data, source = source.data, family = family[i], transfer.source.id = "auto", detection.info = FALSE)
      beta.glmtrans <- glmtrans.tian$beta[(qz[1]+2):(p+1)]
      para.all.tian <- glmtrans.tian$beta
      time.consume[i0,i,r,7]=as.double(difftime(Sys.time(), timeTransGLM, units="secs"))
      res.glmtrans <- Performance_Eva(train.data = data.train, test.data = data.test,  beta.est.train = beta.glmtrans, para.all = para.all.tian, family = family[i])
      oos.kl.loss[i0,i,r,7] <- res.glmtrans$oos.kl.loss
      est.error[i0,i,r,7] <- res.glmtrans$est.error
      me[i0,i,r,7] <- res.glmtrans$model.error
      me.bias[i0,i,r,7] <- res.glmtrans$bias.me
      me.var[i0,i,r,7] <- res.glmtrans$var.me
      mape.is[i0,i,r,7] <- res.glmtrans$mape.is
      mape.oos[i0,i,r,7] <- res.glmtrans$mape.oos
      er.is[i0,i,r,7] <- res.glmtrans$er.is
      er.oos[i0,i,r,7] <- res.glmtrans$er.oos
      
      
      ########### Trans-Lasso #############
      if(family[i] == 'gaussian'){
        timeTransL=Sys.time()
        X.comb <- do.call(rbind, data.train.adj$data.x)
        y.comb <- as.vector(do.call(cbind, data.train.adj$data.y))
        prop.re1 <- Trans.lasso(X.comb, y.comb, size, I.til = 1:(size[1]/2), l1 = T)
        prop.re2 <- Trans.lasso(X.comb, y.comb, size, I.til = (size[1]/2+1):size[1], l1=T)
        beta.translasso <- (prop.re1$beta.hat + prop.re2$beta.hat) / 2
        time.consume[i0,i,r,8]=as.double(difftime(Sys.time(), timeTransL, units="secs"))
        res.translasso <- Performance_Eva(train.data = data.train, test.data = data.test,  beta.est.train = beta.translasso[(qz[1]+2):(p+1)], para.all = beta.translasso, family = family[i])
        oos.kl.loss[i0,i,r,8] <- res.translasso$oos.kl.loss
        est.error[i0,i,r,8] <- res.translasso$est.error
        me[i0,i,r,8] <- res.translasso$model.error
        me.bias[i0,i,r,8] <- res.translasso$bias.me
        me.var[i0,i,r,8] <- res.translasso$var.me
        mape.is[i0,i,r,8] <- res.translasso$mape.is
        mape.oos[i0,i,r,8] <- res.translasso$mape.oos
        er.is[i0,i,r,8] <- res.translasso$er.is
        er.oos[i0,i,r,8] <- res.translasso$er.oos
      }
      
      
      ########### ART #############
      if(family[i] %in% c('gaussian','binomial')){
        timeART=Sys.time()
        source.data.x <- vector(mode='list', length=s.cnt-1)
        source.data.y <- vector(mode='list', length=s.cnt-1)
        for(j in 1:(s.cnt-1)){
          source.data.x[[j]] <- data.train.adj$data.x[[j+1]]
          source.data.y[[j]] <- data.train.adj$data.y[[j+1]]
        }
        if(family[i]=="binomial"){
          fit.art.log <- ART(target.data$x, target.data$y, source.data.x, source.data.y, data.test$data.x[,1:p], func=fit_logit, type="classification")
          beta.art.log <- fit.art.log$coef_ART[(qz[1]+2):(p+1)]
          para.all.artlog <- fit.art.log$coef_ART
          res.art <- Performance_Eva(train.data = data.train, test.data = data.test,  beta.est.train = beta.art.log, para.all = para.all.artlog, family = family[i])
        }else{
          fit.art.lm <- ART(target.data$x, target.data$y, source.data.x, source.data.y, data.test$data.x[,1:p], func=fit_lm, type="regression")
          beta.art.lm <- fit.art.lm$coef_ART[(qz[1]+2):(p+1)]
          para.all.artlm <- fit.art.lm$coef_ART
          res.art <- Performance_Eva(train.data = data.train, test.data = data.test,  beta.est.train = beta.art.lm, para.all = para.all.artlm, family = family[i])
        }
        time.consume[i0,i,r,9]=as.double(difftime(Sys.time(), timeART, units="secs"))
        oos.kl.loss[i0,i,r,9] <- res.art$oos.kl.loss
        est.error[i0,i,r,9] <- res.art$est.error
        me[i0,i,r,9] <- res.art$model.error
        me.bias[i0,i,r,9] <- res.art$bias.me
        me.var[i0,i,r,9] <- res.art$var.me
        mape.is[i0,i,r,9] <- res.art$mape.is
        mape.oos[i0,i,r,9] <- res.art$mape.oos
        er.is[i0,i,r,9] <- res.art$er.is
        er.oos[i0,i,r,9] <- res.art$er.oos
      }
      
      
      if(any(oos.kl.loss[i0,i,r,]==Inf, na.rm = TRUE)){
        # stop("Check...")
        brea_abn=brea_abn+1
        next
      }
      
    }
    
  }
}

