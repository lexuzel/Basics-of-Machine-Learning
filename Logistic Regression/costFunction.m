function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

z = (theta'*X')';
r = -y.*log(sigmoid(z)) - (1-y).*log(1-sigmoid(z));
J = sum(r)/m;
for i=1:size(theta)
grad(i) = sum((sigmoid(z) - y).*X(:,i))/m;
end

end