function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
z1 = zeros(size(X, 1), size(Theta1, 1));
X = [ones(m, 1) X];
for i = 1:size(Theta1, 1)
    z1(:,i) = X*Theta1(i,:)';
end  
z1 = sigmoid(z1);
a1 = [ones(size(z1, 1), 1) z1];

z2 = zeros(size(z1, 1), size(Theta2, 1));
for j = 1:num_labels
    z2(:,j) = a1*Theta2(j,:)';
end    
z2 = sigmoid(z2);
a2 = z2;
for k = 1:m
    temp = a2(k,:);
    [M, I] = max(temp, [], 'linear');
    p(k,:) = I;
end


% =========================================================================


end
