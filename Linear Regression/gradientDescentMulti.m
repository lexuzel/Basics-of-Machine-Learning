function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

h = (theta'*X')';
divJ = (h - y) .* X;
sum_divJ = zeros(size(X,2), 1);
for i = 1:size(X,2)
sum_divJ(i) = sum(divJ(:,i));
end
theta = theta - alpha * 1/m * sum_divJ;
J_history(iter) = computeCostMulti(X, y, theta);

J_history(iter) = computeCostMulti(X, y, theta);

end

end