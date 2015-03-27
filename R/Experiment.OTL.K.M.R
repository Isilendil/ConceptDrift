
# 2015.03.27
# Experiment.OTL.K.M: the main function used to compare all the online algorithms
#================================================================================
# Input: 
#   dataset.name, name of the dataset, e.g. 'usenet1'
# Output:
#   a table containing the accuracies, the numbers of support vectors,
#   the running times of all the online learning algorithms on the inputed datasets
#   a figure for the online average accuracies of all the online learning algorithms
#   a figure for the online numbers of SVs of all the online algorithms
#   a figure for the online running time of all the online learning algorithms
#================================================================================

Experiment.OTL.Km.M <- function(dataset.name)
{
  # load dataset
  require(R.matlab)
  data.mat <- readmat(paste('../data/', dataset.name, sep = ''))
  
  size <- dim(data.mat)
  n <- size[1]
  d <- size[2]
  
  # set parameters
  options <- list()
  options$C <- 5  # penalty parameter for PA-I
  options$Period = 30  # the number of one period for CDOL
  options$lambda <- 10  # parameter lambda for ShiftPE
  options$t.tick <- round(n/20)  # 't.tick' (step size for plotting figures)
  options$sigma <- 8
  
  #options$number.Old <- n - m
  
  Y <- data.mat$data[, 1]
  X <- data.mat$data[, 2:d]
  
  print('Pre-computing kernel matrix...')
  require(kernlab)
  rbf <- rbfdot(sigma = 1/(2*options$sigma^2))
  K <- kernelMatrix(rbf, X)
  
  # run experiments:
  
  n.Group <- dim(data.mat$ID.ALL)[1]
  n.Column <- dim(data.mat$ID.ALL)[2] %/% options$t.tick
  
  for (i in 1 : n.Group)
  {
    print(paste('running on the ', i, '-th trial...', sep = ''))
    
    ID <- data.mat$ID.ALL[i, ]
    
    # 1. PE
    
    
    # 2. PA-I
    
    
    # 3. ShiftPE
    
    
    # 4. ModiPE
    
    
    # 5. CDOLfix
    
    
    # 6. CDOL
  }
  
  # print and plot results
  
  
}

setwd('~/Workspace/OnlineTransferLearning/ConceptDrift/R')



