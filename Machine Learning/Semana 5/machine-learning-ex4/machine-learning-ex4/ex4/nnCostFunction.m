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

%adicionando coluna de 1's em X
X = [ones(size(X,1),1) X];

%Foward Propagation
zTwo = X*Theta1';
aTwo = sigmoid(zTwo);
aTwo = [ones(size(aTwo,1),1) aTwo];
zThree = aTwo*Theta2';
g = sigmoid(zThree); % g = aThree = h(x)

% =======================================================

%computando o custo J(Theta)

for k=1:num_labels
    y_t = zeros(m,1);
    y_t(find(y==k)) = 1;
    J = J+sum(-y_t.*log(g(:,k))-(1-y_t).*log(1-g(:,k)));
end

J = J/m;

regTheta1 = sum(sum(Theta1(:,2:end).^2));
regTheta2 = sum(sum(Theta2(:,2:end).^2));
regularization = (lambda/(2*m))*(regTheta1+regTheta2);
J = (J +regularization);




% -------------------------------------------------------------
%calculando gradiente
for k=1:num_labels
    y_t = zeros(m,1);
    y_t(find(y==k)) = 1;
    deltaMin3(:,k) = g(:,k)-y_t;
end

%BackPropagation
gLinha2 = sigmoidGradient(zTwo);
gLinha2 = [ones(m,1) gLinha2];
deltaMin2 = deltaMin3*Theta2.*gLinha2;

delta3 = aTwo'*deltaMin3;
delta2 = X'*deltaMin2;

Theta1_grad = delta2(:,2:end)'/m;
Theta2_grad = delta3'/m;
% =======================================================

%regularizando Theta2
regularization = (lambda/m)*Theta2;
aux = Theta2_grad(:,1);
Theta2_grad = Theta2_grad+regularization;
Theta2_grad(:,1) = aux;

%regularizando Theta1
regularization = (lambda/m)*Theta1;
aux = Theta1_grad(:,1);
Theta1_grad = Theta1_grad + regularization;
Theta1_grad(:,1) = aux;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
