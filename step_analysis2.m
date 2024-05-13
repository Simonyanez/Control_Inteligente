function [] = step_analysis2(model)
    data = load("data.mat");
    y = data.y;
    u = data.u;
    y=y';
    u=u';
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
    mseResults = zeros(3, 1);
    rmseResults = zeros(3, 1);
    maeResults = zeros(3, 1);
    for i = 1:length(predictionHorizons)
        % Step prediction
        total_data = size(Z_nval, 1); % Get the number of rows
        

        % Horizonte actual
        predictionHorizon = predictionHorizons(i);

        % Reset state of model
        model = resetState(model);
        numPredictionTimeSteps = 600;
        [Y_pred,state] = predict(model, Z_nval(1:total_data-numPredictionTimeSteps,:)); % Predict using initial data
        model.State = state;
        
        % Store predicted data
        Y = zeros(numPredictionTimeSteps, 1);
        Y(1) = Y_pred(end); % Initialize first prediction
        
        % Input data for prediction go on. Independant of prediction
        U = Z_nval(total_data-numPredictionTimeSteps+1:total_data, 2); % Input from 4800 to 5000
        
        % Needed for first iteration total_data - numpredictions + horizon
        % so it has sufficient data before

        
        for t = 1:numPredictionTimeSteps
            % If over in prediction horizon use actual data to predict
            if mod(t,predictionHorizon)==0 || t==1
                [Y(t+1), state] = predict(model, Z_nval(total_data-numPredictionTimeSteps+t,:));
                model.State = state;
            % If not in prediction horizon, use predicted data to keep
            % predicting
            else
                Z_aux = [];
                Y_aux = Y(t-1:t);
                U_aux = U(t-1:t);
                for j = 1:N_reg
                    y_i = Y_aux(N_reg+1-j:end-1);
                    Z_aux = [Z_aux y_i];
                end
    
                for j = 1:N_reg
                    u_i = U_aux(N_reg+1-j:end-j);
                    Z_aux = [Z_aux, u_i];
                end
                [Y(t+1), state] = predict(model, Z_aux);
                model.State = state;
            end
            
        end
        % Initialize arrays to store performance metrics for each model
        
        
        mseResults(i) = mean((Y(:,1) - Z_nval(total_data-numPredictionTimeSteps:total_data,1)).^2, 'all');
        rmseResults(i) = sqrt(mseResults(i));
        maeResults(i) = mean(abs(Y(:,1) - Z_nval(total_data-numPredictionTimeSteps:total_data,1)), 'all');

        
        figure
        t = tiledlayout(1,1);
        title(t,["Pronóstico a lazo abierto para " num2str(predictionHorizon) " pasos" ])
    

        nexttile
        plot(Z_nval(1:total_data - numPredictionTimeSteps,1))
        hold on

        
   
        plot(total_data-numPredictionTimeSteps:total_data, Y(:,1),"--")
        ylabel("Channel " + 1)

        
        xlabel("Muestras")
        legend(["Real" "Pronosticado"])
                
        % Display or compare the performance of each model based on the computed metrics
        Y_nval = Z_nval(total_data-numPredictionTimeSteps:total_data,1);
        figure;
        plot(Y(:,1), 'Color', 'r', 'LineWidth', 0.5); % Plot predicted values as orange curve
        hold on;
        scatter(1:length(Y_nval), Y_nval, 5,'bo', 'filled'); % Plot real values as blue points
        xlabel('Muestra');
        ylabel('Salida');
        legend('Predecido', 'Real');
        title(['Comparison of Predicted and Real Values at ' num2str(predictionHorizon)]);
        hold off;
    end

    for i = 1:numel(predictionHorizons)
    fprintf("%13d    %8.4f    %8.4f    %8.4f\n", predictionHorizons(i), mseResults(i), rmseResults(i), maeResults(i));
    end
end
