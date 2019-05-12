function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

z = (theta'*X')';
r = -y.*log(sigmoid(z)) - (1-y).*log(1-sigmoid(z));
r2 = theta;
r2(1) = 0;
J = (sum(r) + lambda/2*sum(r2.^2))/m;

grad = (X'*(sigmoid(z) - y) + lambda*r2)/m;
grad = grad(:);

end