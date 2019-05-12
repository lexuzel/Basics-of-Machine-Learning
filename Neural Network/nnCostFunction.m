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

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m,1) X];
z2 = Theta1*X';             % 25x5000
a2 = sigmoid(z2);

a2 = [ones(1,m); a2];
z3 = Theta2*a2;             % 10x5000
a3 = sigmoid(z3);

delta3 = a3;

for i = 1:m
    for j = 1:num_labels
        if y(i)==j 
            J = J - log(a3(j,i));
            delta3(j,i) = delta3(j,i) - 1;
        else 
            J = J - log(1-a3(j,i));
        end
    end
end
r1 = Theta1(:, 2:end).^2;
r2 = Theta2(:, 2:end).^2;

regTheta1 = sum(sum(r1));
regTheta2 = sum(sum(r2));

J = (J+lambda/2*(regTheta1 + regTheta2))/m;

delta2 = (Theta2(:,2:end)' * delta3 .* sigmoidGradient(z2));

th2 = Theta2;
th2(:,1) = 0;
th1 = Theta1;
th1(:,1) = 0;

Theta2_grad = delta3*a2' + lambda*th2;
Theta1_grad = delta2*X + lambda*th1;

grad = [Theta1_grad(:) ; Theta2_grad(:)];
grad = grad/m;

end
