function J = costFunctionJ(X,y,theta)

m = size(X,1);
predictions = X*theta;
predictions
sqrErrors = (predictions -y).^2;
sqrErrors
J = 1/(2*m) * sum(sqrErrors);
J