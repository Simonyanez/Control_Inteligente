function [] = step_analysis3(model)
    data = load("data.mat");
    y = data.y;
    u = data.u;
    y = y';
    u = u';
    
    % Number of regressors
    N_reg = 1; 
    
    % Initialize the regressor matrix Z
    Z = [];
    
    % Add y regressors to Z
    for i = 1:N_reg
        y_i = y(N_reg + 1 - i:end - i); 
        Z = [Z y_i]; 
    end
    
    % Add u regressors to Z
    for i = 1:N_reg
        u_i = u(N_reg + 1 - i:end - i);
        Z = [Z u_i];
    end
    
    % Define target vector Y
    Y = y(N_reg + 1:end);

    % Split data into train, validation, and test sets
    n_data = length(Y);
    n_train = ceil(n_data * 0.6);
    n_test = ceil(n_data * 0.85);
    
    % Training data
    Z_train = Z(1:n_train, :);
    Y_train = Y(1:n_train, :);
    
    % Test data
    Z_test = Z(n_train + 1:n_test, :);
    Y_test = Y(n_train + 1:n_test, :);
    
    % Validation data
    Z_val = Z(n_test + 1:end, :);
    Y_val = Y(n_test + 1:end, :);
        
    % Normalize by the training set statistics
    mean_z = mean(Z_train);
    std_z = std(Z_train);
    Z_ntrain = (Z_train - mean_z) ./ std_z;
    Z_ntest = (Z_test - mean_z) ./ std_z;
    Z_nval = (Z_val - mean_z) ./ std_z;
    
    % Normalize the outputs as well
    mean_y = mean(Y_train);
    std_y = std(Y_train);
    Y_ntrain = (Y_train - mean_y) ./ std_y;
    Y_ntest = (Y_test - mean_y) ./ std_y;
    Y_nval = (Y_val - mean_y) ./ std_y;

    % Define prediction horizons
    predictionHorizons = [1, 9, 18];
    mseResults = zeros(length(predictionHorizons), 1);
    rmseResults = zeros(length(predictionHorizons), 1);
    maeResults = zeros(length(predictionHorizons), 1);

    for i = 1:length(predictionHorizons)
        predictionHorizon = predictionHorizons(i);
        total_data = size(Z_nval, 1);

        % Reset the state of the model
        model = resetState(model);
        numPredictionTimeSteps = 600;

        % Initial prediction using validation data
        [Y_pred, state] = predict(model, Z_nval(1:total_data - numPredictionTimeSteps - predictionHorizon, :));
        model.State = state;
        
        % Initialize array for predictions
        Y = zeros(numPredictionTimeSteps, 1);
        Y(1) = Y_pred(end); 
        
        % Get input data for further predictions
        U = Z_nval(total_data - numPredictionTimeSteps - predictionHorizon + 1:total_data, 2);
        
        % Step through the prediction horizon
        for t = 1:numPredictionTimeSteps
            Z_aux = Z_nval(total_data - numPredictionTimeSteps - predictionHorizon + t, :);
            for j = 1:predictionHorizon
                [Y_pred, state] = predict(model, Z_aux);
                model.State = state;
                Z_aux = [Y_pred(end) U(t + j, :)]; % Ensure correct indexing for U

                if j == predictionHorizon
                    Y(t) = Y_pred(end, :);
                end
            end
        end
        
        % Calculate performance metrics
        mseResults(i) = mean((Y - Z_nval(total_data - numPredictionTimeSteps + 1:total_data, 1)).^2, 'all');
        rmseResults(i) = sqrt(mseResults(i));
        maeResults(i) = mean(abs(Y - Z_nval(total_data - numPredictionTimeSteps + 1:total_data, 1)), 'all');
        
        % Plot predictions
        figure;
        t = tiledlayout(1, 1);
        title(t, ["Pronóstico a lazo cerrado para " num2str(predictionHorizon) " pasos"]);
        nexttile;
        plot(Z_nval(1:total_data - numPredictionTimeSteps, 1));
        hold on;
        plot(total_data - numPredictionTimeSteps + 1:total_data, Y, "--");
        ylabel("Channel 1");
        xlabel("Muestras");
        legend(["Real", "Pronosticado"]);
        
        % Comparison plot
        Y_nval_comp = Z_nval(total_data - numPredictionTimeSteps + 1:total_data, 1);
        figure;
        plot(Y, 'Color', 'r', 'LineWidth', 0.5); 
        hold on;
        scatter(1:length(Y_nval_comp), Y_nval_comp, 5, 'bo', 'filled'); 
        xlabel('Muestra');
        ylabel('Salida');
        legend('Predecido', 'Real');
        title(['Comparison of Predicted and Real Values at ' num2str(predictionHorizon)]);
        hold off;
    end

    % Display performance metrics
    for i = 1:numel(predictionHorizons)
        fprintf("%13d    %8.4f    %8.4f    %8.4f\n", predictionHorizons(i), mseResults(i), rmseResults(i), maeResults(i));
    end
end
