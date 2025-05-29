pred.transgvcm <- function(train.data, 
                            test.data, 
                            beta.est.train,
                            para.all,
                            family)
{

	size.test <- nrow(test.data$data.x)

	## KL loss
	oos.pred.kl <- cbind(1,test.data$data.merge[,-1])%*%para.all
	if(family == 'gaussian'){
    temp1 <- rowSums(test.data$data.x*test.data$beta.true)*(rowSums(test.data$data.x*test.data$beta.true)-oos.pred.kl)
    temp2 <- (rowSums(test.data$data.x*test.data$beta.true))^2/2-oos.pred.kl^2/2
    oos.kl.loss <- sum(temp1-temp2)/size.test
    mape.is <- mean(abs(train.data$data.y[[1]]-cbind(1, train.data$data.merge[[1]][,-1])%*%para.all))
    mape.oos <- mean(abs(test.data$data.y-cbind(1,test.data$data.merge[,-1])%*%para.all))
    er.is <- NA
    er.oos <- NA
  }
  else if(family == 'binomial'){
    temp1 <- exp(rowSums(test.data$data.x*test.data$beta.true))/(1+exp(rowSums(test.data$data.x*test.data$beta.true)))*(rowSums(test.data$data.x*test.data$beta.true)-oos.pred.kl)
    temp2 <- log(1+exp(rowSums(test.data$data.x*test.data$beta.true)))-log(1+exp(oos.pred.kl))
    oos.kl.loss <- sum(temp1-temp2)/size.test
    mean.oos.pred <- exp(cbind(1,test.data$data.merge[,-1])%*%para.all)/(1+exp(cbind(1,test.data$data.merge[,-1])%*%para.all))
    mean.test.true <- exp(rowSums(test.data$data.x*test.data$beta.true))/(1+exp(rowSums(test.data$data.x*test.data$beta.true)))
    mean.is.pred <- exp(cbind(1,train.data$data.merge[[1]][,-1])%*%para.all)/(1+exp(cbind(1,train.data$data.merge[[1]][,-1])%*%para.all))
    mean.train.true <- exp(rowSums(train.data$data.x[[1]]*train.data$beta.true[[1]]))/(1+exp(rowSums(train.data$data.x[[1]]*train.data$beta.true[[1]])))
    mape.oos <- mean(abs(mean.oos.pred-mean.test.true))
    mape.is <- mean(abs(mean.is.pred-mean.train.true))
    y.sign.tr <- ifelse(exp(cbind(1,train.data$data.merge[[1]][,-1])%*%para.all)/(1+exp(cbind(1,train.data$data.merge[[1]][,-1])%*%para.all))>0.5,1,0)
    y.sign.te <- ifelse(exp(cbind(1,test.data$data.merge[,-1])%*%para.all)/(1+exp(cbind(1,test.data$data.merge[,-1])%*%para.all))>0.5,1,0)
    er.is <- mean(train.data$data.y[[1]]!=y.sign.tr)
    er.oos <-mean(test.data$data.y!=y.sign.te)
  }
  else if(family == 'poisson'){
    temp1 <- exp(rowSums(test.data$data.x*test.data$beta.true))*(rowSums(test.data$data.x*test.data$beta.true)-oos.pred.kl)
    temp2 <- exp(rowSums(test.data$data.x*test.data$beta.true))-exp(oos.pred.kl)
    oos.kl.loss <- sum(temp1-temp2)/size.test
    mape.oos <- mean(abs(exp(cbind(1,test.data$data.merge[,-1])%*%para.all)-exp(rowSums(test.data$data.x*test.data$beta.true))))
    mape.is <- mean(abs(exp(cbind(1,train.data$data.merge[[1]][,-1])%*%para.all)-exp(rowSums(train.data$data.x[[1]]*train.data$beta.true[[1]]))))
    er.is <- NA
    er.oos <- NA
  }
	## estimation MSE
	mse.est <- sum((beta.est.train-train.data$beta.true[[1]][1,(qz[1]+1):p])^2)
	## ME
	me <- mean((cbind(1,train.data$data.merge[[1]][,-1])%*%para.all-rowSums(train.data$data.x[[1]]*train.data$beta.true[[1]]))^2)

	return(list(oos.kl.loss=oos.kl.loss,
    					est.error=mse.est,
    					model.error=me,
    					mape.is=mape.is,
    					mape.oos=mape.oos,
    					er.is=er.is,
    					er.oos=er.oos))
}



