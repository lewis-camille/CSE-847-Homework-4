% CSE 847 Homework 4
% Sparse Logistic Regression Experiment

% Load Alzheimer's data
load('alzheimers/ad_data.mat');

train_dimensions = size(X_train);
train_num_datapoints = train_dimensions(1);
train_num_features = train_dimensions(2);

parameter_values = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

dim = size(parameter_values);
num_par = dim(2);

% Set up containers to hold:
%    - number of non-zero entries in w
%    - AUC
set_nonzero_weights = zeros(num_par, 1);
set_AUC = zeros(num_par, 1);

% Iterate over the different regularization parameter values
for iPar = 1:num_par
    % Train the sparse lr model
    [w, c] = logistic_l1_train(X_train, y_train, parameter_values(iPar));
    
    % Determine number of non-zero entries in w
    for iWeight = 1:train_num_features
        if w(iWeight) ~= 0
            set_nonzero_weights(iPar) = set_nonzero_weights(iPar) + 1;
        end
    end
    
    % Predict on test set and compute AUC
    scores = all_predictions(X_test, w, c);
    [X, Y, T, AUC] = perfcurve(y_test, scores, 1);
    set_AUC(iPar) = AUC;
end

% Plot num nonzero weights over varying parameter values
f1 = figure('Name', 'Number of non-zero weights for varying parameter values');
scatter(parameter_values, set_nonzero_weights, 'filled')
xlabel('Regularization parameter value')
ylabel('Number of non-zero weights')
title('Number of non-zero weights for varying parameter values')

% Plot AUC over varying parameter values
f2 = figure('Name', 'AUC for varying parameter values');
scatter(parameter_values, set_AUC, 'filled')
xlabel('Regularization parameter value')
ylabel('AUC')
title('AUC for varying parameter values')

% Function to compute predictions (raw probabilities)
function predictions = all_predictions(X, w, c)
    dimensions = size(X);
    num_datapoints = dimensions(1);
    predictions = zeros(num_datapoints, 1);
    
    for i = 1:num_datapoints
       % get data point
       xt = X(i,:);
       % get prediction
       predictions(i) = xt * w + c;
    end
end


% Wrapper to call LogisticR
function [w, c] = logistic_l1_train(data, labels, par)
% OUTPUT w is equivalent to the first d dimension of weights in logistic train
% c is the bias term, equivalent to the last dimension in weights in logistic train.
% Specify the options (use without modification).
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations

[w, c] = LogisticR(data, labels, par, opts);
end
