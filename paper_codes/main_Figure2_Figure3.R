rm(list=ls())
gc()
source("Data_Gen.R")
library(Matrix)
library(MASS)
library(glmnet)
library(Rsolnp)
library(caret)
library(ncvreg)
library(splines)

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

rho <- 0.5 
family <- c("gaussian", "binomial", "poisson")
err.sigma <- 0.5
rep <- 500
size.test <- 100
theta.src <- 1.8
mis.source.index <- c(2,3,4)
brea_error10=0
brea_conve10=0
brea_error20=0
brea_conve20=0
brea_errorjcv=0
brea_convejcv=0
brea_abn=0
options(warn = -1)
size.main <- seq(100,1000,100)
weight.all <- array(NA,dim=c(length(family), length(size.main), rep, s.cnt))

system.time({
  for(i0 in 1:length(family)){
    
    family.opt <- family[i0]
    
    for(i1 in 1:length(size.main)){
      
      size <- rep(size.main[i1],s.cnt)
      ndg <- 3
      nknots <- ceiling(size[1]^(1/5))
      ndf <- ndg+nknots
      
      for(r in 1:rep){

        set.seed(20230327+50*i0+100*r+200*i1)
        cat('---------------------------------\n')
        cat('Now family is', family[i0], ', size is', size.main[i1],'in the', r, 'th replicate..\n')
        cat('---------------------------------\n')

        ## generate all datasets
        data.all <- Data_Gen(family = family[i0], 
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
        
        for(h in 1:s.cnt){
          bsz.tar <- array(0,dim=c(size[h], ndf*qz[h]))
          for(j in 1:qz[h]){
            bsz.tar[,((j-1)*ndf+1):(j*ndf)] <- data.train$data.x[[h]][,j]*bs(data.train$data.u[[h]][,j], df=ndf, degree = ndg)
          }
          data.train$data.merge[[h]] <- cbind(data.train$data.y[[h]], bsz.tar, data.train$data.x[[h]][,(qz[h]+1):p])
        }
 
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
            glm.tr[[j]] <- glm(respon~., data = dataglm, family=family[i0], control=list(maxit=1000))$coefficients
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
        
        weight.all[i0,i1,r,] <- solve.weight$par
        
        rm(data.train, data.all)
        gc()
      }
    }
  }

})

############################## sum of weight #####################################

weight_sum <- matrix(NA, nrow=3, ncol=10)
for(i in 1:3){
  for(j in 1:10){
    weight_sum_tmp <- apply(weight.all[i,j,,],2,mean,na.rm = TRUE)
    weight_sum[i,j] <- round(sum(weight_sum_tmp[-c(2,3,4)]),3)
  }
}

weight_fig <- as.vector(t(weight_sum))
size_ind = rep(size.main,3)
family_name <- c(rep('Gaussian',10),rep('Binomial',10),rep('Poisson',10))
data_total = data.frame(weight_fig=weight_fig, size_ind=size_ind, family_name=family_name)
data_total$size_ind <- factor(data_total$size_ind,
                              levels = c("100", "200", "300", "400", "500",
                                         "600", "700", "800", "900", "1000"))

ggplot(data=data_total, 
       aes(x=size_ind, 
           y=weight_fig, 
           group = family_name)) + 
  geom_line(aes(color=family_name), size=1.5) +
  geom_point(aes(color=family_name, shape=family_name), size=8) +
  ylab('Mean of sum of weights') +
  xlab('Sample size') +
  theme_set(theme_bw()) +
  theme(panel.grid =element_blank(),
        axis.title.y = element_text(size=30),
        axis.text.y = element_text(size=30),
        axis.title.x = element_text(size=30),
        axis.text.x = element_text(size=30),
        legend.text = element_text(size = 30),
        legend.title = element_blank(),
        legend.key.size = unit(0.8, "inches")) 


############################## weight assignments #####################################

weight.all.sel <- weight.all[,c(1,3,6,10),,]
weight_assign <- NULL
for(i in 1:3){
  for(j in 1:4){
    weight_assign_tmp <- round(apply(weight.all.sel[i,j,,],2,mean,na.rm=TRUE),2)
    weight_assign <- c(weight_assign, weight_assign_tmp)
  }
}

size_ind = rep(rep(size.main[c(1,3,6,10)], each=7),3)
family_name <- c(rep('Gaussian',28),rep('Binomial',28),rep('Poisson',28))
model_ind <- rep(c("target model (k=0)", "source model (k=1)", "source model (k=2)",
               "source model (k=3)", "source model (k=4)", "source model (k=5)",
               "source model (k=6)"), 12)
data_total = data.frame(weight_assign=weight_assign, size_ind=size_ind, family_name=family_name, model_ind=model_ind)
data_total$size_ind <- factor(data_total$size_ind, levels = c("100", "300", "600", "1000"))

ggplot(data=data_total, 
       aes(x=size_ind, 
           y=weight_assign, 
           fill = model_ind,
           label = weight_assign)) + 
  geom_bar(stat = "identity",
           position = "fill", width =0.6) +
  ylab('Mean of weight estimates') +
  xlab('Sample size') +
  scale_fill_manual(values = c("target model (k=0)"="#98d09d","source model (k=1)"="#d7e698",
                               "source model (k=2)"="#dadada","source model (k=3)"="#fbf398",
                               "source model (k=4)"="#f7a895","source model (k=5)"="#e77381",
                               "source model (k=6)"="#9b8191"),
                    limits=c("target model (k=0)", "source model (k=1)", "source model (k=2)",
                             "source model (k=3)", "source model (k=4)", "source model (k=5)",
                             "source model (k=6)")) +
  scale_y_continuous(expand = expansion(mult=c(0,0.1))) +
  theme_set(theme_bw()) +
  theme(panel.grid =element_blank(),
        axis.line = element_line(),
        axis.title.y = element_text(size=30),
        axis.text.y = element_text(size=30),
        axis.title.x = element_text(size=30),
        axis.text.x = element_text(size=30),
        legend.text = element_text(size = 30),
        legend.title = element_blank(),
        legend.key.size = unit(0.8, "inches"),
        legend.position = "bottom") +
  facet_wrap(~ family_name, scales='free', nrow = 1) +
  theme(strip.text.x = element_text(size = 30)) + 
  theme(strip.text.y = element_text(size = 30)) 
dev.off()
