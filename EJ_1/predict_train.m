function [] = predict_train(model)
    data = load("data.mat");
    y = data.y';
    u = data.u';
    % Concatenate regressors
    % Split data into train, val and test
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

%%
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