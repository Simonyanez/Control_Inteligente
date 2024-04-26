function [] = predict_train(model)
    data = load("data_p1_v2.mat");
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
    Y_ntrain_pred = predict(model, Z_ntrain);
    
    % Plot the actual training data and the predicted values
    figure;
    plot(Y_ntrain_pred(1:600,1), 'Color', 'r', 'LineWidth', 0.5); % Plot predicted values as orange curve
    hold on;
    scatter(1:length(Y_ntrain(1:600,1)), Y_ntrain(1:600,1), 5,'bo', 'filled'); % Plot real values as blue points
    ylim([-10,15])
    xlabel('Muestra');
    ylabel('Salida');
    legend('Predecido', 'Real');
    title('Comparacion entrenamiento');
    hold off;
end