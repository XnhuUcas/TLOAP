# Model aggregation-assisted transfer learning framework for generalized semiparametric models

This repository is the implementation of "*Model aggregation-assisted transfer learning framework for generalized semiparametric models*". This repository provides a multi-source transfer learning procedure for prediction via frequentist model averaging under generalized partially linear varying-coefficient models. 

## Maintainer

Xiaonan Hu (<xiaonanhu@cnu.edu.cn>)

Any questions or comments, please don’t hesitate to contact with me any time.

**File Structure:**

- **`auxiliary_codes`**
  - ART.R: main code for the ART algorithm;
  - Data_Gen.R: code for simulated data generation;
  - Data_Gen_Sen.R: code for noise data generation;
  - Performance_Eva.R: code for all evaluation indicators;
  - translasso_func.R: all codes for the implementation of Trans-Lasso algorithm.

- **`dataset`**
  - diabetes.csv: the Pima Indians Diabetes dataset available at \url{https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data};
  - train.csv: the loan approval dataset from the “Playground Series-Season 4, Episode 10” competition publicly available at \url{https://www.kaggle.com/competitions/playground-series-s4e10/data}.

- **`paper_codes`**  
  codes for reproduction of all numerical results in our paper:
  - main_Figure2_Figure3.R: results of Fig.2 and Fig.3 in main text;
  - main_Figure4_supp_FigureS6_FigureS7_TableS12.R: results of Fig.4 in main text and Fig.S.6, Fig.S.7, and Table.S.12 in the Supplementary Materials;
  - main_Figure5_supp_FigureS8_TableS12.R: results of Fig.5 in main text and Fig.S.8 and Table.S.12 in the Supplementary Materials;
  - main_Table1_Table2_Table3_supp_FigureS2_TableS11.R: results of Table 1-3 in main text and Fig.S.2 and Table.S.11 in the Supplementary Materials;
  - supp_FigureS9_FigureS11.R: results of Fig.S.9 and Fig.S.11 in the Supplementary Materials;
  - supp_FigureS10_FigureS12.R: results of Fig.S.10 and Fig.S.12 in the Supplementary Materials;
  - supp_TableS1_TableS2_TableS3_FigureS4.R: results of Table S.1-S.3 and Fig.S.4 in the Supplementary Materials;
  - supp_TableS4_TableS5_TableS6_TableS11_FigureS3.R: results of Table S.4-S.6, Table S.11 and Fig.S.3 in the Supplementary Materials;
  - supp_TableS7_TableS8_TableS9_FigureS5.R: results of Table S.7-S.9 and Fig.S.5 in the Supplementary Materials;
  - supp_TableS10.R: results of Table S.10 in the Supplementary Materials.

- **`example.R`**
  an example code

- **`pred.tloap.R`**
  an example code
  
- **`simdata.gen.R`**
  an example code

## Usage

This is a simple example which shows users how to use the provided
functions to estimate weights and make predictions.

First, we generate simulation datasets (M=3) under the corrected target
model and homogeneous dimension settings.

``` r
library(matrans)

## generate simulation datasets (M=3)
size <- c(150, 200, 200, 150)
coeff0 <- cbind(
  as.matrix(c(1.4, -1.2, 1, -0.8, 0.65, 0.3)),
  as.matrix(c(1.4, -1.2, 1, -0.8, 0.65, 0.3) + 0.02),
  as.matrix(c(1.4, -1.2, 1, -0.8, 0.65, 0.3) + 0.3),
  as.matrix(c(1.4, -1.2, 1, -0.8, 0.65, 0.3))
)
px <- 6
err.sigma <- 0.5
rho <- 0.5
size.test <- 500

whole.data <- simdata.gen(
  px = px, num.source = 4, size = size, coeff0 = coeff0, coeff.mis = as.matrix(c(coeff0[, 2], 1.8)),
  err.sigma = err.sigma, rho = rho, size.test = size.test, sim.set = "homo", tar.spec = "cor",
  if.heter = FALSE
)
data.train <- whole.data$data.train
data.test <- whole.data$data.test
```

Then, we apply the functions to implement weights estimation and
out-of-sample predictions.

``` r
## optimal weights estimation
bs.para <- list(bs.df = rep(3, 3), bs.degree = rep(3, 3))
data.train$data.x[[2]] <- data.train$data.x[[2]][, -7]
fit.transsmap <- trans.smap(train.data = data.train, nfold = 5, bs.para = bs.para)
ma.weights <- fit.transsmap$weight.est
time.transsmap <- fit.transsmap$time.transsmap

## out-of-sample prediction results
pred.res <- pred.transsmap(object = fit.transsmap, newdata = data.test, bs.para = bs.para)
pred.val <- pred.res$predict.val
```

