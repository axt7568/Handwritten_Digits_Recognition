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

%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J.
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. 
%
%        
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         

a_one = [ones(m, 1) X];
z_two = a_one * Theta1';
a_two = sigmoid(z_two);
a_two = [ones(size(a_two,1), 1) a_two];
z_three = a_two * Theta2';
a_three = sigmoid(z_three);
h = a_three; 

Y = zeros(m,num_labels);

for j = 1:m
  Y(j,y(j)) = 1;  
end

J = (1/m) * sum(sum(-1 * Y.* log(h)-(1-Y).*log(1-h)));
k = lambda/(2*m);
reg = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * k;
J = J + reg;

% Part 2 - Backpropagation algorithm 

for t = 1:m
  a_one = [1; X(t,:)'];
  z_two = Theta1 * a_one;
  a_two = [1; sigmoid(z_two)];
  z_three = Theta2 * a_two;
  a_three = sigmoid(z_three);
  h = a_three;
  y_req = ([1:num_labels]==y(t))';
  d3 = h - y_req;
  d2 = (Theta2' * d3).*[1; sigmoidGradient(z_two)];
  d2 = d2(2:end);
  Theta1_grad = Theta1_grad + d2 * a_one';
  Theta2_grad = Theta2_grad + d3 * a_two';
end

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];;
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
