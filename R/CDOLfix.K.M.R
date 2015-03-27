
# 2015.03.27
# CDOLfix: Concept Drift Online Learning (fix)
#===============================================
# Input:
#   Y: the vector of labels
#   K: precomputed kernel for all the example, i.e., K_{ij} = K(x_i, x_j)
#   id.list: a randomized ID list
#   options: a struct containing rho, sigma, C, n.label and t.tick
# Output:
#   err.count: total number of training errors
#   run.time: time consumed by this algorithm once
#   mistakes: a vector of mistake rate
#   mistake.idx: a vector of number, in which every number corresponds to a mistake rate in the vector above
#   SVs: a vector records the number of support vectors
#   size.SV: the size of final support set
#==========================================================

CDOLfix.K.M <- function(Y, K, options, id.list)
{
  # initialize parameters
  C <- options$C  # 1 by default
  P <_ options$Period
  t.tick <- options$t.tick
  ID <- id.list
  
  err.count <- 0
  mistakes <- vector()
  mistakes.idx <- vector()
  SVs <- vector()
  TMs <- vector()
  
  alpha.v <- vector()
  SV.v <- vector()
  
  alpha.w <- vector()
  SV.w <- vector()
  
  a.1t <- 0
  a.2t <- 1
  eta <- 1/2
  
  # loop
  
  t1 <- proc.time()
  
  for (t in 1 : length(ID))
  {
    id <- ID[t]
    y.t <- Y[id]
    
    if (length(alpha.v) == 0)
    {
      f.v <- 0
    }
    else
    {
      k.v <- K[id, SV.v]
      f.v <- alpha.v %*% k.v
    }
    
    if (length(alpha.w) == 0)
    {
      f.w <- 0
    }
    else
    {
      k.w <- K[id, SV.w]
      f.w <- alpha.w %*% k.w
    }
    
    hat.f.v <- max(0, min(1, (f.v+1)/2))
    hat.f.w <- max(0, min(1, (f.w+1)/2))
    f.t <- a.1t * hat.f.v + a.2t * hat.f.w - 1/2
    
    hat.y.t <- sign(f.t)
    if (hat.y.t == 0)
    {
      hat.y.t <- 1
    }
    
    if (hat.y.t != y.t)
    {
      err.count <- err.count + 1
    }
    
    loss.w <- max(0, 1-y.t*f.w)
    s.t <- K[id, id]
    if ( (loss.w > 0) && (s.t != 0) )  # 
    {
      gamma.t <- min(C, loss.w/s.t)
      alpha.w <- c(alpha.w, gamma.t*y.t)
      SV.w <- c(SV.w, id)
    }
    
    Pi.y <- (y.t+1) / 2
    ell1 <- (hat.f.v - Pi.y) ^ 2
    ell2 <- (hat.f.w - Pi.y) ^ 2
    
    a.1t <- a.1t * exp(-eta * ell1)
    a.2t <- a.2t * exp(-eta * ell2)
    sum.a <- a.1t + a.2t
    a.1t <- a.1t / sum.a
    a.2t <- a.2t / sum.a
    
    if ((t %% P) == 0)
    {
      if (a.2t > a.1t)
      {
        alpha.v <- alpha.w
        SV.v <- SV.w
      }
      alpha.w <- vector()
      SV.w <- vector()
      a.1t <- 1/2
      a.2t <- 1/2
    }
    
    t2 <- proc.time()
    run.time <- t2[3] - t1[3]
    
    if ((t %% t.tick) == 0)
    {
      mistakes <- c(mistakes, err.count/t)
      mistakes.idx <- c(mistakes.idx, t)
      SVs <- c(SVs, length(SV.w) + length(SV.v))
      TMs <- c(TMs, run.time)
    }
  }
  
  classifier <- list(SV.v = SV.v, alpha.v = alpha.v,
                     SV.w = SV.w, alpha.w = alpha.w)
  
  model <- list(classifier = classifier, err.count = err.count, mistakes = mistakes,
                mistakes.idx = mistakes.idx, SVs = SVs, TMs = TMs)
  
  print('The number of mistakes =', err.count)
  
  t3 <- proc.time()
  run.time <- t3[3] - t2[3]
  
  model$run.time <- run.time
  
  return(model)

}


