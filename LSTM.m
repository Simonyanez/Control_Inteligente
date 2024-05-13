clc
clear all
close all

data = load("data.mat");
y = data.y';
u = data.u';

N_reg = 1;
% Inicializar los regresores como una matriz vacía
Z = [];

% Y
for i=1:N_reg % índice del regresor
    y_i = y(N_reg+1-i:end-i); % Toma los regresores como ventana deslizante 2 al final-2 y 1 al final-1
    Z = [Z y_i]; % Concatena el vector con los regresores
end

% U
% Se realiza el mismo procedimiento para u
for i=1:N_reg
    u_i = u(N_reg+1-i:end-i);
    Z = [Z u_i];
end

% Target
Y = y(N_reg+1:end);

    %Split Train, Test, Val
    n_data = length(Z);
    n_train = ceil(n_data*0.6);
    n_test = ceil(n_data*0.85);
    
    % Train
    Z_train = Z(1:n_train,:);
    Y_train = Y(1:n_train,:);
    disp(["Train length data " size(Z_train)]);
    % Test
    Z_test = Z(n_train+1:n_test,:);
    Y_test = Y(n_train+1:n_test,:);
    
    % Val
    Z_val = Z(n_test+1:end,:);
    Y_val = Y(n_test+1:end,:);
        
    % Normalize by the training set
    mu = mean(Z_train);
    sigma = std(Z_train);
    Z_ntrain = (Z_train - mu) ./ sigma;
    Z_ntest = (Z_test - mu) ./ sigma;
    Z_nval = (Z_val - mu) ./ sigma;
    
    % Normalize the outputs as well
    mu_y = mean(Y_train);
    sigma_y = std(Y_train);
    Y_ntrain = (Y_train - mu_y) ./ sigma_y;
    Y_ntest = (Y_test - mu_y) ./ sigma_y;
    Y_nval = (Y_val - mu_y) ./ sigma_y;

    % Layer Array Neural Networ Architecture
    numFeatures = size(Z_train,2); % ?
    numHiddenUnits = 20;
    numResponses = size(Y_train,2); % ?

    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits,OutputMode="sequence")
        fullyConnectedLayer(numResponses)];
    
    % Training
    options = trainingOptions("adam", ...
        MaxEpochs=400, ...
        MiniBatchSize=25, ...
        InitialLearnRate=0.005, ...
        Plots="training-progress", ...
        Verbose=0, ...
        ValidationFrequency = 25, ...  % a 1/8 of epochs needed for full test
        ValidationData={Z_ntest,Y_ntest}, ...
        LearnRateSchedule='piecewise', ... % Use piecewise learning rate schedule
        LearnRateDropFactor= 0.2, ... % Factor to drop learning rate
        LearnRateDropPeriod= 50)  % Every time it fully goes over train set
        
    lstm_model = trainnet(Z_ntrain,Y_ntrain,layers,'mse',options);
    save lstm_model;