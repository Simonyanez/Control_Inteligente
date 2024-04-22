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
    
    % Initialize arrays to store performance metrics for each model
    mseResults = zeros(numel(predictionHorizons), 1);
    rmseResults = zeros(numel(predictionHorizons), 1);
    % Add more metrics as needed
    
    % Loop over each prediction horizon
    for i = 1:numel(predictionHorizons)
        % Make predictions using the trained network for the current prediction horizon
        % Assuming Z_ntest and Y_ntest are your test data
        Z_hor_test = Z_ntest(1:predictionHorizons(i),:, :); % Adjust input data for current prediction horizon
        Y_hor_test = Y_ntest(predictionHorizons(i) + 1:end,:); % Adjust target data for current prediction horizon
        Y_hor_pred = predict(model, Z_hor_test);
        
        % Evaluate performance using appropriate metrics
        % For example, compute MSE and RMSE
        mse = mean((Y_hor_pred - Y_hor_test).^2, 'all');
        rmse = sqrt(mse);
    
        disp(["This is Z_test, Y_test and Y_pred sizes " size(Z_hor_test) size(Y_hor_test) size(Y_hor_pred)]);
        % Plot the comparison
        numSamples = size(Y_test, 1);
        figure;
        plot(1:numSamples, Y_hor_pred, 'b-', 'LineWidth', 1.5); % Predicted values in blue
        hold on;
        plot(1:numSamples, Y_hor_test, 'r--', 'LineWidth', 1.5); % Actual values in red dashed
        hold off;
        % Customize the plot
        title(sprintf('Predicted vs Actual Values at Prediction Horizon %d', predictionHorizons(i)));
        xlabel('Sample Index');
        ylabel('Value');
        legend('Predicted', 'Actual');
        grid on;

    
        plot(Y_hor_test)
        % Store results
        mseResults(i) = mse;
        rmseResults(i) = rmse;
        % Store more metrics as needed
    end
        
    % Display or compare the performance of each model based on the computed metrics
    disp("Performance Metrics on Test Set:");
    disp("Prediction Horizon    MSE        RMSE");
    for i = 1:numel(predictionHorizons)
        fprintf("%13d    %8.4f    %8.4f\n", predictionHorizons(i), mseResults(i), rmseResults(i));
    end
end

