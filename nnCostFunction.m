function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1: Feedforward and Cost Function ======================================

% add the column of 1's to the X matrix 
X = [ones(m,1), X];
% compute the activations of the hidden layer (m x hidden_layer_size)
hiddenLayerZ = X * Theta1';
hiddenLayer = sigmoid(hiddenLayerZ);
% add the bias unit to the hidden layer (m x (hidden_layer_size + 1))
hiddenLayer = [ones(m,1), hiddenLayer];
% compute the activations of the output layer (m x num_labels)
outputLayerZ = hiddenLayer * Theta2';
% each row is the hypothesis for one training example
outputLayer = sigmoid(outputLayerZ);

% utilize vectorization by creating a (m x num_labels) matrix for y by 
% expanding each entry into a num_labels-dimensional vector
yVectors = zeros(m, num_labels);
for i = 1:m
	value = y(i);
	yVectors(i, value) = 1;
endfor

% compute the cost function by comparing each entry in outputLayer (h(x^i)k)
% with each entry in yVectors ((y^i)k)
% costs is still a m x num_labels matrix
costs = -1/m .* (yVectors.*log(outputLayer) + (1-yVectors).*log(1-outputLayer));
% compute the sum of all of the elements in costs (do a double sum since 2D matrix)
notRegularizedCost = sum(sum(costs));
% compute the regularization term, but don't regularize the bias terms
% set the first column (bias terms) of Theta1 and Theta2 both to all 0's first
Theta1(:, 1) = 0;
Theta2(:, 1) = 0;
regularizationTerm = lambda./(2.*m).*(sum(sum(Theta1 .^ 2)) + sum(sum(Theta2 .^ 2)));
% add the regularizationTerm to get the final cost 
J = notRegularizedCost + regularizationTerm;

% Part 2: Backpropagation and Gradients ===================================

% already computed the activations of all units in the previous part
% compute delta for the output layer
% both outputLayer and yVectors are m x num_labels
dOutput = outputLayer - yVectors;
% compute delta for the hiddenLayer (m x hidden_layer_size) 
% don't want to compute delta for the bias unit, so don't use the first
% column of Theta2
% (m x num_labels) * (num_labels x hidden_layer_size)
dHidden = dOutput * Theta2(:, 2:end) .* sigmoidGradient(hiddenLayerZ);
% compute Delta1 (product of dHidden and a1): hidden_layer_size x input_layer_size
% (hidden_layer_size x m) * (m x input_layer_size)
Delta1 = dHidden' * X;
% compute Delta2 (product of dOutput and a2): num_labels x hidden_layer_size + 1
% (num_labels x m) * (m x (hidden_layer_size + 1))
Delta2 = dOutput' * hiddenLayer;
% compute Theta1_grad and Theta2_grad by dividing Delta1 and Delta2 by m
Theta1_grad = Delta1./m;
Theta2_grad = Delta2./m;

% add regularization for the gradients, but not for the bias terms
% don't include the first columns of Theta1 and Theta2 (bias terms)
Theta1(:, 1) = 0;
Theta2(:, 1) = 0;
Theta1_grad = Theta1_grad + lambda/m.*Theta1;
Theta2_grad = Theta2_grad + lambda/m.*Theta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
