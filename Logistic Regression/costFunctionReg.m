function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

z = (theta'*X')';
r = -y.*log(sigmoid(z)) - (1-y).*log(1-sigmoid(z));
r2 = theta(2:end);
J = (sum(r) + lambda/2*sum(r2.^2))/m;

grad(1) = sum((sigmoid(z) - y).*X(:,1))/m;
for i=2:size(theta)
grad(i) = (sum((sigmoid(z) - y).*X(:,i)) + lambda*theta(i))/m;
end

end