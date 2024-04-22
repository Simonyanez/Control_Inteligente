function model = LSTM
    data = load("data_p1.mat");
    y = data.y;
    u = data.u;
    % Concatenate regressors
    % Split data into train, val and test
    N_reg = 0; % No es necesario para LSTM, la componente temporal es parte de la estructura
    Z = [];

    %Split Train, Test, Val
    n_data = length(y);
    n_train = ceil(n_data*0.6);n_data = length(Z);
    n_test = ceil(n_data*0.85);
    
    % Train
    y_train = y(1:n_train,:);
    u_train = u(1:n_train,:);
    
    % Test
    y_test = y(n_train+1:n_test,:);
    u_test = u(n_train+1:n_test,:);
    
    % Val
    y_val = y(n_test+1:end,:);
    u_val = u(n_test+1:end,:);
        

    % Normalize sets
    y_ntrain = normalize(y_train); % Norma 1
    u_ntrain = normalize(u_train);

    y_ntest = normalize(y_test);
    u_ntest = normalize(u_test);

    y_nval = normalize(y_val);
    u_nval = normalize(u_val);

    % Layer Array Neural Networ Architecture
    numFeatures = size(y_train,2); % ?
    numHiddenUnits = 50;
    numResponses = size(u_train,2); % ?

    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits,OutputMode="sequence")
        fullyConnectedLayer(numResponses)];
    
    % Training
    options = trainingOptions("adam", ...
        MaxEpochs=80, ...
        MiniBatchSize=20, ...
        InitialLearnRate=0.01, ...
        GradientThreshold=1, ...
        Shuffle="never", ...
        Metrics="rmse", ...
        Plots="training-progress", ...
        Verbose=0, ...
        ValidationFrequency = 10, ...
        ValidationData={y_nval,u_nval});
        
    netTrained = trainnet(y_ntrain,u_ntrain,layers,"mse",options);
end