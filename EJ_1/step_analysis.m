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
        offset = size(Z_ntest,1);
        offset = offset(1)
        disp(["This is offset" offset])
        predictionHorizon = predictionHorizons(i);
        model = resetState(model);
        numPredictionTimeSteps = 200;
        [Y_pred,state] = predict(model,Z_ntest(1:offset-numPredictionTimeSteps,:)); % 1 - 4800
        model.State = state;
    
        Y = zeros(numPredictionTimeSteps,1);

        Y(1,:) = Y_pred(end,1);
        %U = generate_u_vals(offset,numPredictionTimeSteps);
        U = Z_ntest(offset-numPredictionTimeSteps+1:offset,2); % Input from 4800 to 5000
        Z_aux = [];

        disp(["This is U size " size(U)])
        disp(["This Y size" size(Y)]);
        
        Z_ntest_copy = Z_ntest(1:offset-1-numPredictionTimeSteps+predictionHorizon,:);
        for t = 1:numPredictionTimeSteps-predictionHorizon
            disp(["This Z size" size(Z_ntest_copy)]);
            disp(["Index value" offset-numPredictionTimeSteps+t])
            [Y(t+1,:),state] = predict(model,Z_ntest_copy(offset-1-numPredictionTimeSteps+t,:));
            if mod(t,predictionHorizon) == 0
                Z_aux = [];
                disp("In condition")
                Y_aux = Y(t:t+predictionHorizon,:);
                U_aux = U(t:t+predictionHorizon,:);
                for i=1:N_reg
                    y_i = Y_aux(N_reg+1-i:end-1);
                    Z_aux = [Z_aux y_i];
                end
                
                for i=1:N_reg
                    u_i = U_aux(N_reg+1-i:end-i);
                    Z_aux = [Z_aux,u_i];
                    
                end
                disp(["Zaux size" size(Z_aux)])
                Z_ntest_copy = vertcat(Z_ntest_copy,Z_aux);
            end
            model.State = state;
        end

        % Timesteps
        
        
        % Initialize arrays to store performance metrics for each model
        mseResults = zeros(numel(predictionHorizons), 1);
        rmseResults = zeros(numel(predictionHorizons), 1);
    
        % Add more metrics as needed
        disp(offset-numPredictionTimeSteps);
        disp(["Y size" size(Y)])
        disp(["Y_pred size" size(Z_ntest(offset-numPredictionTimeSteps:offset,1))])
        mse = mean((Y(:,1) - Z_ntest(offset-numPredictionTimeSteps+1:offset,1)).^2, 'all');
        rmse = sqrt(mse);
    
        % Loop over each prediction horizon
        numTimeSteps = offset + numPredictionTimeSteps;
        
        figure
        t = tiledlayout(1,1);
        title(t,["Open Loop Forecasting for " predictionHorizon "steps" ])
    

        nexttile
        plot(Z_ntest(1:offset-numPredictionTimeSteps,1))
        hold on
        disp(["Offset size" size(Z_ntest(offset,1))])
        
        disp([size(offset:numTimeSteps)])
        plot(offset-numPredictionTimeSteps+1:offset, Y(:,1),"--")
        ylabel("Channel " + 1)

        
        xlabel("Time Step")
        legend(["Input" "Forecasted"])
                
        % Display or compare the performance of each model based on the computed metrics
        disp("Performance Metrics on Test Set:");
        disp("Prediction Horizon    MSE        RMSE");
        for i = 1:numel(predictionHorizons)
           fprintf("%13d    %8.4f    %8.4f\n", predictionHorizons(i), mseResults(i), rmseResults(i));
    
        end
    end
end

