% CSE 847 Homework 4
% Logistic Regression Experiment

% Load spam email data
email_data = readmatrix('spam_email/data.txt');
email_labels = readmatrix('spam_email/labels.txt');

% Remove first two columns of labels
% -> unnecessarily there because of extra spacing in txt file
email_labels = email_labels(:,3);

% Add column of ones to data matrix
dimensions = size(email_data);
num_datapoints = dimensions(1);
ones_column = ones(num_datapoints, 1);
email_data = [email_data ones_column];

% Split into train and test sets
% First 2000 rows training, the remaining test
train_data = email_data(1:2000,:);
train_labels = email_labels(1:2000,:);
test_data = email_data(2001:end,:);
test_labels = email_labels(2001:end,:);


train_set_sizes = [200, 500, 800, 1000, 1500, 2000];
accuracies = [0, 0, 0, 0, 0, 0];
% Iterate over different training set sizes
% n = 200, 500, 800, 1000, 1500, 2000
for iSize = 1:6
    train_subset_data = train_data(1:train_set_sizes(iSize),:);
    train_subset_labels = train_labels(1:train_set_sizes(iSize),:);
    
    % Train model
    w = logistic_train(train_subset_data, train_subset_labels, 1e-5, 1000);
    
    % Get accuracy of predictions on test set
    accuracies(iSize) = all_predictions(test_data, test_labels, w);
end

accuracies


function accuracy = all_predictions(X, y, w)
    dimensions = size(X);
    num_datapoints = dimensions(1);
    num_correct_pred = 0;
    for i = 1:num_datapoints
       % get data point
       xt = X(i,:);
       % get prediction
       pred = xt * w;
       if pred > 0
           pred = 1;
       else
           pred = 0;
       end
       % get ground truth
       truth = y(i);
       % Evaluate accuracy by comparing prediction to ground truth
       if abs(pred - truth) < 0.0001
           num_correct_pred = num_correct_pred + 1;
       end
    end
    accuracy = num_correct_pred / num_datapoints * 100.0;
end

function result = perform_sigmoid(x)
    result = (1./(1+exp(-x)));
end

function weights = logistic_train(data, labels, epsilon, maxiter)
    % Inputs:
    %    data     = n * (d+1) matrix with n samples and d features, where
    %               column d+1 is all ones (corresonding to the intercept term)
    %    labels   = n * 1 vector of class labels (taking values 0 or 1)
    %               epsilon = optional argument specifying the convergence
    %               criterion - if the change in the absolute difference in
    %               predictions, from one iteration to the next, averaged aross
    %               input features, is less than epsilon, then halt
    %               (if unspecified, use a default value of 1e-5)
    %    maxiter  = optional argument that specifies the maximum number of
    %               iterations to execute 
    %               (if unspecified can be set to 1000)
    % Output:
    %    weights  = (d+1) * 1 vector of weights where the weights correspond to
    %                the columns of "data"

    % Using step size 0.01
    step_size = 0.01;

    data_dimensions = size(data);
    n = data_dimensions(1);
    d_plus_1 = data_dimensions(2);
    %d = d_plus_1 - 1;

    % Initialize weights to 0
    weights = zeros(d_plus_1, 1);

    % Iterate maxiter times
    for i = 1:maxiter        
        % Compute gradient of likelihood w.r.t. weights
        b = perform_sigmoid(data * weights) - labels;
        gradient = (1.0 / n) .* (transpose(data) * b);
        
        % new weights = old weights - (step size * gradient)
        new_weights = weights - (step_size * gradient);
        
        % if diff < epsilon, halt
        if sum(abs(new_weights - weights)) / d_plus_1 < epsilon
            weights = new_weights;
            break
        end
        weights = new_weights;
    end
end

