import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape
  for i in range(0, N[0]):
    score = np.exp(X[i,:].dot(W))
    all_sum = np.sum(score)
    loss += (-np.log(score[y[i]] / all_sum))
    dW[:,y[i]] += X[i].T
    #TODO need more efficient
    #for j in range(W.shape[1]):
    # dW[:,j] -= (score[j] * X[i].T) / all_sum
    dW -= X[i].reshape((N[1],1))*score / all_sum
    
    
    
  loss /= N[0]
  loss += 0.5 * reg * np.sum( W * W )

  dW /= (-N[0])
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  scores = X.dot(W)
  scores = np.exp(scores)
  sum_all = np.sum(scores, axis=1)
  loss = np.sum(-np.log(scores[range(0,N),y] / sum_all))
  #dW -= np.sum(X.reshape((X.shape[0],X.shape[1],1)) * scores.reshape(scores.shape[0],1,scores.shape[1]) / sum_all.reshape(sum_all.shape[0],1,1), axis = 0)
  loss /= N
  loss += 0.5 * reg * np.sum( W * W )

  dW /= -N
  dW += W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

