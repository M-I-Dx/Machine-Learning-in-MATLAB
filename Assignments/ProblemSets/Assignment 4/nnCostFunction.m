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
sizeX = size(X);
X_with_bias = [ones(sizeX(:,1), 1) X];
z2 = Theta1*X_with_bias';
z2 = z2';
a2 = sigmoid(z2);
size_z2 = size(z2);
a2_with_bias = [ones(size_z2(:,1), 1) a2];
z3 = Theta2*a2_with_bias';
z3 = z3';
a3 = sigmoid(z3);
hyp = a3;
y_class = zeros(size(hyp));

for k = 1:num_labels
    pos = find(y==k);
    temp = zeros(size(X,1), 1);
    temp(pos) = 1;
    y_class(:, k) = temp;
end

for k = 1:num_labels
    Jx = sum((-y_class(:,k).*log(hyp(:,k)) - (1-y_class(:,k)).*(log(1-hyp(:,k)))));
    Jx = Jx/m;
    J = J + Jx;
end

Theta1_x = Theta1;
Theta1_x(:,1) = 0;
Theta2_x = Theta2;
Theta2_x(:,1) = 0;

reg = sum(sum(power(Theta1_x, 2))) + sum(sum(power(Theta2_x, 2)));
reg = (lambda/(2*m))*reg;
J = J + reg;



Theta2_without_bias = Theta2;
Theta2_without_bias(:,1) = [];
Delta2 = zeros(size(Theta2));
Delta1 = zeros(size(Theta1));

for k = 1:m
    ax1 = X_with_bias(k,:);
    ax2 = a2_with_bias(k,:);
    ax3 = a3(k,:)';
    d3 = ax3 - y_class(k,:)';
    d2 = (Theta2_without_bias'*d3).*(sigmoidGradient(z2(k,:))');
    Delta2 = Delta2 + d3*ax2;
    Delta1 = Delta1 + d2*ax1;
end



Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;
reg_theta1 = (Theta1)*(lambda/m);
reg_theta1(:,1) = 0;
reg_theta2 = (Theta2)*(lambda/m);
reg_theta2(:, 1) = 0;

Theta1_grad = Theta1_grad + reg_theta1;
Theta2_grad = Theta2_grad + reg_theta2;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
