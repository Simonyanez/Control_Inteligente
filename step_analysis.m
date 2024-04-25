function [] = step_analysis(model)
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
    % Define the prediction horizons
    predictionHorizons = [1, 9, 18];
    for i = 1:length(predictionHorizons)
        % Step prediction
        offset = predictionHorizons(i);
        model = resetState(model);
        numTimeSteps = size(Z_ntest,1);
        numPredictionTimeSteps = numTimeSteps - offset;
        [Y_pred,state] = predict(model,Z_ntest(1:offset,:));
        model.State = state;
    
        Y = zeros(numPredictionTimeSteps);
        Y(1,:) = Y_pred(end,1);

        for t = 1:numPredictionTimeSteps-1
            Z_ntestt = Z_ntest(offset+t,:);
            [Y(t+1,:),state] = predict(model,Z_ntestt);
            model.State = state;
        end

        % Timesteps
        
        
        % Initialize arrays to store performance metrics for each model
        %mseResults = zeros(numel(predictionHorizons), 1);
        %rmseResults = zeros(numel(predictionHorizons), 1);
    
        % Add more metrics as needed
        %mse = mean((Y_hor_pred - Y_hor_test).^2, 'all');
        %rmse = sqrt(mse);
    
        % Loop over each prediction horizon
        figure
        t = tiledlayout(1,1);
        title(t,["Open Loop Forecasting for " offset "steps" ])
    

        nexttile
        plot(Z_ntest(:,1))
        hold on
        disp(["Offset size" offset])
        disp(["Y size" size(Y(:,1))])
        plot(offset:numTimeSteps,[Z_ntest(offset,1) Y(:,1)'],"--")
        ylabel("Channel " + 1)

        
        xlabel("Time Step")
        legend(["Input" "Forecasted"])
                
        % Display or compare the performance of each model based on the computed metrics
        %disp("Performance Metrics on Test Set:");
        %disp("Prediction Horizon    MSE        RMSE");
        %for i = 1:numel(predictionHorizons)
        %    fprintf("%13d    %8.4f    %8.4f\n", predictionHorizons(i), mseResults(i), rmseResults(i));
    end
end

