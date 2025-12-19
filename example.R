rm(list=ls())
gc()
source("Data_Gen.R")
source("Performance_Eva.R")

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

p <- 7
s.cnt <- 7
qz <- rep(2,s.cnt)
size <- rep(100, s.cnt)
para.true <- cbind(as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)),
                   as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)+0.02),
                   as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)+0.02), 
                   as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)+0.3),
                   as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)),
                   as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)+0.02),
                   as.matrix(c(0.5,0.2,-0.2,-0.3,0.5)+0.3)) 

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

ndg <- 3
nknots <- ceiling(size[1]^(1/5))
ndf <- ndg+nknots
rho <- 0.5 
family.opt <- "binomial"
err.sigma <- 0.5
size.test <- 100
theta.src <- 1.8
theta.tar <- 0.1
mis.source.index <- c(2,3,4)
options(warn = -1)

## generate all datasets
data.all <- Data_Gen(family = family.opt, 
                     type = 'corr', 
                     s.cnt = s.cnt, 
                     size = size, 
                     size.test = size.test, 
                     para.true = para.true, 
                     theta.src = theta.src, 
                     err.sigma = err.sigma, 
                     p = p, 
                     qz = qz, 
                     mis.source.index = mis.source.index)
data.train <- data.all$train_data
data.test <- data.all$test_data

###### TLOAP-CV10
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

n_group_10 <- {if((size[1]%%10)== 0) size[1]%/%10 else size[1]%/%10+1} # d=10/5/1
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
    glm.tr[[j]] <- glm(respon~., data = dataglm, family=family.opt, control=list(maxit=1000))$coefficients
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
  gvcm.res[[k]] <- glm(respon~., data = data.train.frame, family=family.opt, control=list(maxit=1000))
  beta.est.train.mat[,k] <- gvcm.res[[k]]$coefficients[(ndf*qz[k]+2):length(gvcm.res[[k]]$coefficients)]
}
weight.est <- solve.weight$par
beta.est.train <- beta.est.train.mat%*%weight.est
para.all <- c(gvcm.res[[1]]$coefficients[1:(ndf*qz[1]+1)], beta.est.train)
res.summary.cv10 <- Performance_Eva(train.data = data.train, test.data = data.test,  beta.est.train = beta.est.train, para.all = para.all, family = family.opt)

