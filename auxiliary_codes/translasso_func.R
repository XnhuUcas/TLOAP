
# library(glmnet)
agg.fun<- function(B, X.test,y.test, total.step=10, selection=F){
  if(sum(B==0)==ncol(B)*nrow(B)){
    return(rep(0,nrow(B)))
  }
  p<-nrow(B)
  K<-ncol(B)
  colnames(B)<-NULL
  if(selection){#select beta.hat with smallest prediction error
    khat<-which.min(colSums((y.test-X.test%*%B)^2))
    theta.hat<-rep(0, ncol(B))
    theta.hat[khat] <- 1
    beta=B[,khat]
    beta.ew=NULL
  }else{#Q-aggregation
    theta.hat<- exp(-colSums((y.test-X.test%*%B)^2)/2)
    theta.hat=theta.hat/sum(theta.hat)
    theta.old=theta.hat
    beta<-as.numeric(B%*%theta.hat)
    beta.ew<-beta
    # theta.old=theta.hat
    for(ss in 1:total.step){
      theta.hat<- exp(-colSums((y.test-X.test%*%B)^2)/2+colSums((as.vector(X.test%*%beta)-X.test%*%B)^2)/8)
      theta.hat<-theta.hat/sum(theta.hat)
      beta<- as.numeric(B%*%theta.hat*1/4+3/4*beta)
      if(sum(abs(theta.hat-theta.old))<10^(-3)){break}
      theta.old=theta.hat
    }
  }
  list(theta=theta.hat, beta=beta, beta.ew=beta.ew)
}


ind.set<- function(n.vec, k.vec){
  ind.re <- NULL
  for(k in k.vec){
    if(k==1){
      ind.re<-c(ind.re,1: n.vec[1])
    }else{
      ind.re<- c(ind.re, (sum(n.vec[1:(k-1)])+1): sum(n.vec[1:k]))
    }
  }
  ind.re
}
rep.col<-function(x,n){
  matrix(rep(x,each=n), ncol=n, byrow=TRUE)
}


###oracle Trans-Lasso
las.kA <- function(X, y, A0, n.vec, lam.const=NULL, l1=T){
  p<-ncol(X)
  size.A0<- length(A0)
  if(size.A0 > 0){
    ind.kA<- ind.set(n.vec, c(1, A0+1))
    ind.1<-1:n.vec[1]
    if(l1){
      y.A<-y[ind.kA]
    }else{ #the l0-method
      y.A<- y[ind.1]
      Sig.hat<-t(X)%*%X/nrow(X)
      for(k in 1:size.A0){
        ind.k<- ind.set(n.vec,k+1)
        lam.k <- sqrt(mean(y[ind.1]^2)/n.vec[1]+mean(y[ind.k]^2)/n.vec[k]) * sqrt(2*log(p))
        delta.hat.k<-lassoshooting(XtX=Sig.hat, 
                                   Xty=t(X[ind.k,])%*%y[ind.k]/n.vec[k+1]-t(X[1:n.vec[1],])%*%y[1:n.vec[1]]/n.vec[1],
                                   lambda=lam.k)$coef
        y.A<-c(y.A, y[ind.k]-X[ind.k,]%*%delta.hat.k)
      }
    }
    if(is.null(lam.const)){
      cv.init<-cv.glmnet(X[ind.kA,], y.A, nfolds=8, lambda=seq(1,0.1,length.out=10)*sqrt(2*log(p)/length(ind.kA)))
      lam.const <- cv.init$lambda.min/sqrt(2*log(p)/length(ind.kA))
    }
    w.kA.temp <- glmnet(X[ind.kA,], y.A, lambda=lam.const*sqrt(2*log(p)/length(ind.kA)))
    w.kA.beta <- as.numeric(w.kA.temp$beta)
    # w.kA <- as.numeric(glmnet(X[ind.kA,], y.A, lambda=lam.const*sqrt(2*log(p)/length(ind.kA)))$beta)
    w.kA<-c(as.numeric(w.kA.temp$a0), w.kA.beta*(abs(w.kA.beta)>=lam.const*sqrt(2*log(p)/length(ind.kA))))
    # cv.delta<-cv.glmnet(x=X[ind.1,],y=y[ind.1]-X[ind.1,]%*%w.kA, lambda=seq(1,0.1,length.out=10)*sqrt(2*log(p)/length(ind.1)))
    #delta.kA<-predict(cv.delta, s='lambda.min', type='coefficients')[-1]
    delta.kA.temp <- glmnet(x=X[ind.1,],y=y[ind.1]-(X[ind.1,] %*% w.kA[-1] + w.kA[1]), lambda=lam.const*sqrt(2*log(p)/length(ind.1)))
    delta.kA.beta <- as.numeric(delta.kA.temp$beta)
    # delta.kA <- as.numeric(glmnet(x=X[ind.1,],y=y[ind.1]-X[ind.1,]%*%w.kA, lambda=lam.const*sqrt(2*log(p)/length(ind.1)))$beta)
    delta.kA <- c(as.numeric(delta.kA.temp$a0), delta.kA.beta*(abs(delta.kA.beta)>=lam.const*sqrt(2*log(p)/length(ind.1))))
    beta.kA <- w.kA + delta.kA
    lam.const=NA
  }else{
    cv.init<-cv.glmnet(X[1:n.vec[1],], y[1:n.vec[1]], nfolds=8, lambda=seq(1,0.1,length.out=20)*sqrt(2*log(p)/n.vec[1]))
    lam.const<-cv.init$lambda.min/sqrt(2*log(p)/n.vec[1])
    beta.kA <- predict(cv.init, s='lambda.min', type='coefficients')
    # beta.kA <- predict(cv.init, s='lambda.min', type='coefficients')[-1]
    w.kA<-NA
  }
  list(beta.kA=as.numeric(beta.kA),w.kA=w.kA, lam.const=lam.const)
  
}


#Trans Lasso method
Trans.lasso <- function(X, y, n.vec, I.til, l1=T){
  M= length(n.vec)-1
  #step 1
  # X0.til<-X[I.til,] #used for aggregation
  X0.til<-cbind(matrix(1,length(I.til),1),X[I.til,])
  y0.til<-y[I.til]
  X<- X[-I.til,]
  y<-y[-I.til]
  #step 2
  Rhat <- rep(0, M+1)
  p<- ncol(X)
  n.vec[1]<- n.vec[1]-length(I.til)
  ind.1<-ind.set(n.vec,1)
  for(k in 2: (M+1)){
    ind.k<-ind.set(n.vec,k)
    Xty.k <- t(X[ind.k,])%*%y[ind.k]/n.vec[k] - t(X[ind.1,])%*%y[ind.1]/n.vec[1]
    margin.T<-sort(abs(Xty.k),decreasing=T)[1:round(n.vec[1]/3)]
    Rhat[k] <-  sum(margin.T^2)
  }
  Tset<- list()
  k0=0
  kk.list<-unique(rank(Rhat[-1]))
  #cat(rank(Rhat[-1]),'\n')
  for(kk in 1:length(kk.list)){#use Rhat as the selection rule
    Tset[[k0+kk]]<- which(rank(Rhat[-1]) <= kk.list[kk])
  }
  k0=length(Tset)
  Tset<- unique(Tset)
  #cat(length(Tset),'\n')
  
  beta.T<-list()
  init.re<-las.kA(X=X, y=y, A0=NULL, n.vec=n.vec, l1=l1)
  beta.T[[1]] <- init.re$beta.kA
  beta.pool.T<-beta.T ##another method for comparison
  for(kk in 1:length(Tset)){#use pi.hat as selection rule
    T.k <- Tset[[kk]]
    re.k<- las.kA(X=X, y=y, A0=T.k, n.vec=n.vec, l1=l1, lam.const=init.re$lam.const)
    beta.T[[kk+1]] <-re.k$beta.kA
    beta.pool.T[[kk+1]]<-re.k$w.kA
  }
  beta.T<-beta.T[!duplicated((beta.T))]
  beta.T<- as.matrix(as.data.frame(beta.T))
  agg.re1 <- agg.fun(B=beta.T, X.test=X0.til, y.test=y0.til)
  beta.pool.T<-beta.pool.T[!duplicated((beta.pool.T))]
  beta.pool.T<- as.matrix(as.data.frame(beta.pool.T))
  agg.re2<-agg.fun(B=beta.pool.T, X.test=X0.til, y.test=y0.til)
  
  return(list(beta.hat=agg.re1$beta, theta.hat=agg.re1$theta, rank.pi=rank(Rhat[-1]),
              beta.pool=agg.re2$beta, theta.pool=agg.re2$theta))
}


lasso.adapt.bic2 <- function(x,y){
  
  # adaptive lasso from lars with BIC stopping rule 
  # this one uses the "known variance" version of BIC with RSS/(full model mse)
  # must use a recent version of R so that normalize=FALSE can be used in lars
  
  require(lars)
  ok<-complete.cases(x,y)
  x<-x[ok,]                            # get rid of na's
  y<-y[ok]                             # since regsubsets can't handle na's
  m<-ncol(x)
  n<-nrow(x)
  x<-as.matrix(x)                      # in case x is not a matrix
  
  #  standardize variables like lars does 
  one <- rep(1, n)
  meanx <- drop(one %*% x)/n
  xc <- scale(x, meanx, FALSE)         # first subtracts mean
  normx <- sqrt(drop(one %*% (xc^2)))
  names(normx) <- NULL
  xs <- scale(xc, FALSE, normx)        # now rescales with norm (not sd)
  
  # out.ls=lm(y~xs)                      # ols fit on standardized
  # beta.ols=out.ls$coeff[2:(m+1)]       # ols except for intercept
  ## correct initial estimate
  beta.ols <- c()
  for(i in 1:m){
    beta.ols[i] <- sum(xs[,i]*y)/n
  }
  w=abs(beta.ols)                      # weights for adaptive lasso
  xs=scale(xs,center=FALSE,scale=1/w)  # xs times the weights
  object=lars(xs,y,type="lasso",normalize=FALSE)
  
  # get min BIC
  bic=log(n)*object$df+n*log(as.vector(object$RSS)/n)   # rss/n version
  # sig2f=summary(out.ls)$sigma^2        # full model mse
  # bic2=log(n)*object$df+as.vector(object$RSS)/sig2f       # Cp version
  step.bic2=which.min(bic)            # step with min BIC
  
  fit=predict.lars(object,xs,s=step.bic2,type="fit",mode="step")$fit
  coeff=predict.lars(object,xs,s=step.bic2,type="coef",mode="step")$coefficients
  coeff=coeff*w/normx                  # get back in right scale
  st=sum(coeff !=0)                    # number nonzero
  mse=sum((y-fit)^2)/(n-st-1)          # 1 for the intercept
  
  # this next line just finds the variable id of coeff. not equal 0
  if(st>0) x.ind<-as.vector(which(coeff !=0)) else x.ind<-0
  intercept=as.numeric(mean(y)-meanx%*%coeff)
  return(list(fit=fit,st=st,mse=mse,x.ind=x.ind,coeff=coeff,intercept=intercept,object=object,
              bic2=bic,step.bic2=step.bic2))
}


