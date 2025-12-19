Data_Gen_Sen <- function(family, 
                         type,
                         s.cnt, 
                         size, 
                         size.test, 
                         para.true, 
                         theta.tar = NULL, 
                         theta.src = NULL, 
                         err.sigma, 
                         p, 
                         qz,
                         mis.source.index = NULL,
                         noise.cov = NULL,
                         noise.ind = NULL)
{
  ########## target correct
  
  if(type == 'corr'){
    # train data
    respon <- vector(mode='list', length=s.cnt)
    beta.true <- vector(mode='list', length=s.cnt)
    datax <- vector(mode='list', length=s.cnt)
    datax.noise <- vector(mode='list', length=s.cnt)
    datau <- vector(mode='list', length=s.cnt)
    datau.noise <- vector(mode='list', length=s.cnt)
    vary.coef <- vector(mode='list', length=s.cnt)
    data.merge <- vector(mode='list', length=s.cnt)
    
    for(i in 1:s.cnt){
      datau[[i]] <- matrix(runif(size[i]*qz[i],0,1), nrow=size[i])
      fun.name <- get(paste('smooth.fun.true',i,sep = ''))
      vary.coef[[i]] <- fun.name(datau[[i]])
      if(is.null(noise.ind)){
        datau.noise[[i]] <- datau[[i]]
      }else{
        datau.noise[[i]] <- datau[[i]] + noise.ind
      }
    }
    
    for(k in 1:s.cnt){
      if(k %in% mis.source.index){
        beta.true[[k]] <- cbind(vary.coef[[k]], matrix(rep(c(para.true[,k],theta.src),each=size[k]),nrow=size[k]))
        xmat <- rho^abs(outer(1:(p+1),1:(p+1),"-"))
        datax[[k]] <- mvrnorm(size[k],rep(0,p+1),xmat)
      }else{
        beta.true[[k]] <- cbind(vary.coef[[k]], matrix(rep(para.true[,k],each=size[k]),nrow=size[k]))
        xmat <- rho^abs(outer(1:p,1:p,"-"))
        datax[[k]] <- mvrnorm(size[k],rep(0,p),xmat)
      }
      if(is.null(noise.cov)){
        datax.noise[[k]] <- datax[[k]]
      }else{
        datax.noise[[k]] <- datax[[k]] + noise.cov
      }
      if(family == "gaussian"){
        respon[[k]] <- as.numeric(rowSums(datax[[k]]*beta.true[[k]]) + rnorm(size[k],0,err.sigma))
      }else if(family == "binomial"){
        pr <- 1/(1+exp(-rowSums(datax[[k]]*beta.true[[k]])))
        respon[[k]] <- sapply(1:size[k], function(i){sample(0:1, size = 1, prob = c(1-pr[i], pr[i]))})
      }else if(family == "poisson"){
        lambda <- as.numeric(exp(rowSums(datax[[k]]*beta.true[[k]])))
        respon[[k]] <- rpois(size[k], lambda) 
      }
      data.merge[[k]] <- cbind(respon[[k]],datax[[k]])        
    }
    data.train <- list(data.y=respon, beta.true=beta.true, data.x=datax, data.u=datau, data.merge=data.merge)
    data.train.noise <- list(data.y=respon, beta.true=beta.true, data.x = datax.noise, data.u = datau.noise, data.merge=data.merge)
    
    ###### testing data
    xmat.te <- rho^abs(outer(1:p,1:p,"-"))
    datax.te <- mvrnorm(size.test,rep(0,p),xmat.te)
    datau.te <- matrix(runif(size.test*qz[1],0,1), nrow=size.test)
    vary.coef.te <- smooth.fun.true1(datau.te)
    beta.true.te <- cbind(vary.coef.te, matrix(rep(para.true[,1],each=size.test),nrow=size.test))
    if(family == "gaussian"){
      respon.te <- as.numeric(rowSums(datax.te*beta.true.te) + rnorm(size.test, 0, err.sigma))
    }else if(family == "binomial"){
      pr <- 1/(1+exp(-rowSums(datax.te*beta.true.te)))
      respon.te <- sapply(1:size.test, function(i){sample(0:1, size = 1, prob = c(1-pr[i], pr[i]))})
    }else if(family == "poisson"){
      lambda <- as.numeric(exp(rowSums(datax.te*beta.true.te)))
      respon.te <- rpois(size.test, lambda) 
    }
    data.merge.te <- cbind(respon.te, datax.te)
    data.test <- list(data.y=respon.te, beta.true=beta.true.te, data.x=datax.te, data.u=datau.te, data.merge=data.merge.te)
  }
  
  ######## target is misspecified
  
  else if(type == 'mis'){
    
    # train data
    respon <- vector(mode='list', length=s.cnt)
    beta.true <- vector(mode='list', length=s.cnt)
    datax <- vector(mode='list', length=s.cnt)
    datax.noise <- vector(mode='list', length=s.cnt)
    datau <- vector(mode='list', length=s.cnt)
    datau.noise <- vector(mode='list', length=s.cnt)
    vary.coef <- vector(mode='list', length=s.cnt)
    data.merge <- vector(mode='list', length=s.cnt)
    
    for(i in 1:s.cnt){
      datau[[i]] <- matrix(runif(size[i]*qz[i],0,1), nrow=size[i])
      fun.name <- get(paste('smooth.fun.true',i,sep = ''))
      vary.coef[[i]] <- fun.name(datau[[i]])
      if(is.null(noise.ind)){
        datau.noise[[i]] <- datau[[i]]
      }else{
        datau.noise[[i]] <- datau[[i]] + noise.ind
      }
    }
    
    # main model
    beta.true[[1]] <- cbind(vary.coef[[1]], matrix(rep(c(para.true[,1],theta.tar), each=size[1]),nrow=size[1]))
    xmat <- rho^abs(outer(1:(p+1),1:(p+1),"-"))
    datax[[1]] <- mvrnorm(size[1],rep(0,(p+1)),xmat)
    if(is.null(noise.cov)){
      datax.noise[[1]] <- datax[[1]]
    }else{
      datax.noise[[1]] <- datax[[1]] + noise.cov
    }
    if(family == "gaussian"){
      respon[[1]] <- as.numeric(rowSums(datax[[1]]*beta.true[[1]]) + rnorm(size[1],0,err.sigma))
    }else if(family == "binomial"){
      pr <- 1/(1+exp(-rowSums(datax[[1]]*beta.true[[1]])))
      respon[[1]] <- sapply(1:size[1], function(i){sample(0:1, size = 1, prob = c(1-pr[i], pr[i]))})
    }else if(family == "poisson"){
      lambda <- as.numeric(exp(rowSums(datax[[1]]*beta.true[[1]])))
      respon[[1]] <- rpois(size[1], lambda) 
    }
    data.merge[[1]] <- cbind(respon[[1]],datax[[1]])
    
    for(k in 2:s.cnt){
      if(k %in% mis.source.index){ 
        beta.true[[k]] <- cbind(vary.coef[[k]], matrix(rep(c(para.true[,k],theta.src), each=size[k]),nrow=size[k]))
        xmat <- rho^abs(outer(1:(p+1),1:(p+1),"-"))
        datax[[k]] <- mvrnorm(size[k],rep(0,(p+1)),xmat)
      }else{
        beta.true[[k]] <- cbind(vary.coef[[k]], matrix(rep(para.true[,k],each=size[k]),nrow=size[k]))
        xmat <- rho^abs(outer(1:p,1:p,"-"))
        datax[[k]] <- mvrnorm(size[k],rep(0,p),xmat)
      }
      if(is.null(noise.cov)){
        datax.noise[[k]] <- datax[[k]]
      }else{
        datax.noise[[k]] <- datax[[k]] + noise.cov
      }
      if(family == "gaussian"){
        respon[[k]] <- as.numeric(rowSums(datax[[k]]*beta.true[[k]]) + rnorm(size[k],0,err.sigma))
      }else if(family == "binomial"){
        pr <- 1/(1+exp(-rowSums(datax[[k]]*beta.true[[k]])))
        respon[[k]] <- sapply(1:size[k], function(i){sample(0:1, size = 1, prob = c(1-pr[i], pr[i]))})
      }else if(family == "poisson"){
        lambda <- as.numeric(exp(rowSums(datax[[k]]*beta.true[[k]])))
        respon[[k]] <- rpois(size[k], lambda) 
      }
      data.merge[[k]] <- cbind(respon[[k]],datax[[k]])
    }
    data.train <- list(data.y=respon, beta.true=beta.true, data.x=datax, data.u=datau, data.merge=data.merge)
    data.train.noise <- list(data.y=respon, beta.true=beta.true, data.x = datax.noise, data.u = datau.noise, data.merge=data.merge)
    
    # test data
    xmat.te <- rho^abs(outer(1:(p+1),1:(p+1),"-"))
    datax.te <- mvrnorm(size.test,rep(0,(p+1)),xmat.te)
    datau.te <- matrix(runif(size.test*qz[1],0,1), nrow=size.test)
    vary.coef.te <- smooth.fun.true1(datau.te)
    beta.true.te <- cbind(vary.coef.te, matrix(rep(c(para.true[,1],theta.tar), each=size.test),nrow=size.test))
    if(family == "gaussian"){
      respon.te <- as.numeric(rowSums(datax.te*beta.true.te) + rnorm(size.test, 0, err.sigma))
    }else if(family == "binomial"){
      pr <- 1/(1+exp(-rowSums(datax.te*beta.true.te)))
      respon.te <- sapply(1:size.test, function(i){sample(0:1, size = 1, prob = c(1-pr[i], pr[i]))})
    }else if(family == "poisson"){
      lambda <- as.numeric(exp(rowSums(datax.te*beta.true.te)))
      respon.te <- rpois(size.test, lambda) 
    }
    data.merge.te <- cbind(respon.te, datax.te)
    data.test <- list(data.y=respon.te, beta.true=beta.true.te, data.x=datax.te, data.u=datau.te, data.merge=data.merge.te)
  }
  
  return(list(train_data = data.train, train_data.noise = data.train.noise, test_data = data.test))
  
}

