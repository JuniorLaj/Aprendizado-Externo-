function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


positivo = find(y==1);
negativo = find(y==0);
plot(X(positivo,1),X(positivo,2),'b+','MarkerSize',6);
hold on;
plot(X(negativo,1),X(negativo,2),'ro','MarkerSize',6);


% =========================================================================



hold off;

end
