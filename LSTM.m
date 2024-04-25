function lstm_model = LSTM
    data = load("data_p1.mat");
    y = data.y;
    u = data.u;
    % Concatenate regressors
    % Split data into train, val and test
    N_reg = 1; % No se necesitan más regresores. No hay análisis de sensibilidad
    Z = [];
    
    for i=1:N_reg
        y_i = y(N_reg+1-i:end-1);
        Z = [Z y_i];
    end
    
    for i=1:N_reg
        u_i = u(N_reg+1-i:end-i);
        Z = [Z,u_i];
    end

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
        

    % Normalize sets
    Z_ntrain = normalize(Z_train); % Norma 1
    Y_ntrain = normalize(Y_train);

    Z_ntest = normalize(Z_test);
    Y_ntest = normalize(Y_test);

    Z_nval = normalize(Z_val);
    Y_nval = normalize(Y_val);

    % Layer Array Neural Networ Architecture
    numFeatures = size(Z_train,2); % ?
    numHiddenUnits = 50;
    numResponses = size(Y_train,2); % ?

    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits,OutputMode="sequence")
        fullyConnectedLayer(numResponses)];
    
    % Training
    options = trainingOptions("adam", ...
        MaxEpochs=800, ...
        MiniBatchSize=60, ...
        InitialLearnRate=0.01, ...
        GradientThreshold=1, ...
        Shuffle="never", ...
        Metrics="rmse", ...
        Plots="training-progress", ...
        Verbose=0, ...
        ValidationFrequency = 25, ...  % a 1/8 of epochs needed for full test
        ValidationData={Z_ntest,Y_ntest}, ...
        LearnRateSchedule='piecewise', ... % Use piecewise learning rate schedule
        LearnRateDropFactor= 0.6, ... % Factor to drop learning rate
        LearnRateDropPeriod= 200)  % Every time it fully goes over train set
        
    lstm_model = trainnet(Z_ntrain,Y_ntrain,layers,"mse",options);
    save lstm_model;
end