#' ART: Adaptive and Robust Transfer Learning
#'
#' ART is a flexible framework for transfer learning that leverages information from auxiliary data sources to enhance model performance on primary tasks. 
#' It is designed to be robust against negative transfer by including the non-transfer model in the candidate pool, 
#' ensuring stable performance even when auxiliary datasets provide limited or no useful information. 
#' The ART framework supports both regression and classification tasks, aggregating predictions from multiple 
#' auxiliary models and the primary model using an adaptive exponential weighting mechanism to prevent negative transfer.
#' Variable importance is also provided to indicate the contribution of each variable in the final model.
#'
#' @param X       A matrix for the primary dataset (target domain) predictors.
#' @param y       A vector for the primary dataset (target domain) responses.
#' @param X_aux   A list of matrices for the auxiliary datasets (source domains) predictors.
#' @param y_aux   A list of vectors for the auxiliary datasets (source domains) responses.
#' @param X_test  A matrix for the test dataset predictors.
#' @param func    A function used to fit the model on each dataset. The function must have the following signature: 
#'                \code{func(X, y, X_val, y_val, X_test, min_prod = 1e-5, max_prod = 1-1e-5, ...)}. 
#'                The function should return a list with the following elements:
#'                \itemize{
#'                  \item{\code{dev}: The deviance (or loss) on the validation set if provided.}
#'                  \item{\code{pred}: The predictions on the test set if \code{X_test} is provided.}
#'                  \item{\code{coef} (optional): The model coefficients (only for regression models when \code{is_coef = TRUE}).}
#'                }
#'                Pre-built wrapper functions, such as \code{fit_lm}, \code{fit_logit}, \code{fit_glmnet_lm}, 
#'                \code{fit_glmnet_logit}, \code{fit_random_forest}, \code{fit_gbm}, and \code{fit_nnet}, can be used. 
#'                Users may also provide their own model-fitting functions, but the input and output structure must follow 
#'                the described signature and format.
#' @param lam     A regularization parameter for weighting the auxiliary models. Default is 1.
#' @param maxit   The maximum number of iterations for the model. Default is 5000.
#' @param eps     A convergence threshold for stopping the iterations. Default is 1e-6.
#' @param type    A string specifying the task type. Options are "regression" or "classification". Default is "regression".
#' @param is_coef Logical; if TRUE, coefficients from the model are returned. Default is TRUE.
#' @param importance Logical; if TRUE, variable importance is calculated. Only applicable if `is_coef` is TRUE. Default is TRUE.
#' @param ...     Additional arguments passed to the model-fitting function.
#'
#' @details
#' The ART function performs adaptive and robust transfer learning by iteratively 
#' combining predictions from the primary dataset and auxiliary datasets. It updates 
#' the weights of each dataset's predictions through an aggregation process, eventually 
#' yielding a final set of predictions based on weighted contributions from the source and 
#' target models.
#'
#' The auxiliary datasets (`X_aux` and `y_aux`) must be provided as lists, with each 
#' element corresponding to a dataset from a different source domain.
#'
#' @return
#' A list containing:
#' \item{pred_ART}{The predictions for the test dataset.}
#' \item{coef_ART}{The coefficients of the final model, if `is_coef` is TRUE.}
#' \item{W_ART}{The final weights for each dataset (including the primary dataset).}
#' \item{iter_ART}{The number of iterations performed until convergence.}
#' \item{VI_ART}{The variable importance, if `importance` is TRUE.}
#'
#' @examples
#' # Example usage
#' dat <- generate_data(n0=50, K=3, nk=50, K_noise=2, nk_noise=30, p=10, 
#'        mu_trgt=1, xi_aux=0.5, ro=0.5, err_sig=1)
#' fit <- ART(dat$X, dat$y, dat$X_aux, dat$y_aux, dat$X_test, func=fit_lm, lam=1, type="regression")
#'
#' @export
ART <- function(X, y, X_aux, y_aux, X_test, func, lam = 1, maxit = 5000L, 
  eps = 1e-6, type = c("regression", "classification"), 
  is_coef = TRUE, importance = TRUE, ...) {

  # model setting
  if (!is.list(X_aux) || !is.list(y_aux)) {
    stop("Auxiliary sets, X_aux and y_aux, must be lists.")
  }
  K <- length(y_aux)
  if (K != length(X_aux)) {
    stop("X_aux and y_aux don't have the same length.")
  }
  # if (is_coef && type == "classification") {
  #   stop("is_coef is TRUE only for the regression models.")
  # }
  if (importance) {
    if (!is_coef) stop("is_coef must be TRUE for variable importance")
  }
  n0 <- NROW(X)
  W_all <- rep(0, K + 1)
  iter  <- 1

  # Aggregation by mixing
  for (iter in seq(maxit)) {
    id0 <- sample(n0, n0 / 2)
    X_00 <- X[id0, ]
    y_00 <- y[id0]
    X_01 <- X[-id0, ]
    y_01 <- y[-id0]

    fit_0_agg <- func(X_00, y_00, X_val = X_01, y_val = y_01, X_test = NULL, ...)
    W_0 <- exp(-lam * fit_0_agg$dev)
    W_k <- rep(0, K)

    for (k in seq(K)) {
      X_k0_agg <- rbind(X_aux[[k]], X_00)
      y_k0_agg <- c(y_aux[[k]], y_00)
      fit_k_agg <- func(X_k0_agg, y_k0_agg, X_val = X_01, y_val = y_01, X_test = NULL, ...)
      W_k[k] <- exp(-lam * fit_k_agg$dev)
    }   
    sum_W_k <- W_0 + sum(W_k)
    W_0     <- W_0 / sum_W_k
    W_k     <- W_k / sum_W_k
    W_new   <- (W_all * (iter - 1) + c(W_0, W_k)) / iter
    dif     <- sum((W_new - W_all)^2)
    W_all   <- W_new
    if (dif < eps) break
  }

  # Final model
  fit_0 <- func(X, y, X_val = NULL, y_val = NULL, X_test = X_test, ...)
  pred_0 <- fit_0$pred
  pred_k <- matrix(NA, NROW(X_test), K)
  if (is_coef) {
    coef_0 <- fit_0$coef 
    coef_k <- matrix(NA, NCOL(X) + 1, K)
  }
  for (k in seq(K)) {
    X_k <- rbind(X_aux[[k]], X)
    y_k <- c(y_aux[[k]], y)
    fit_k <- func(X_k, y_k, X_val = NULL, y_val = NULL, X_test = X_test, ...)
    pred_k[, k] <- fit_k$pred
    if (is_coef) coef_k[, k] <- fit_k$coef
  }

  pred_ART <- as.vector(pred_k %*% matrix(W_k, nrow=K) + W_0 * pred_0)
  coef_ART <- NULL
  if (is_coef) {
    coef_ART <- as.vector(coef_k %*% matrix(W_k, nrow=K) + W_0 * coef_0)
  }
  VI_ART <- NULL
  if (importance) {
    VI_ART <- as.vector((coef_k != 0) %*% matrix(W_k, nrow=K) + 
      W_0 * (coef_0 != 0))
  }
  list(pred_ART = pred_ART, coef_ART = coef_ART, 
    W_ART = W_all, iter_ART = iter, VI_ART=VI_ART)
}
