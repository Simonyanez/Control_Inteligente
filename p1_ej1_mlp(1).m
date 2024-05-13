close all
clear
clc
%%
addpath("Toolbox_TS_y_MLP/Toolbox NN/")
%%
% Cargar datos
data = load("data_p1_v2.mat");
y = data.y;
u = data.u;

% Warm start: descartar los primeros 1000 datos
y = y(1000:end);
u = u(1000:end);
%%
N_reg = 20;
% Inicializar los regresores como una matriz vacía
Z = [];

% Y
for i=1:N_reg % índice del regresor
    y_i = y(N_reg+1-i:end-i); % Toma los regresores como ventana deslizante
    Z = [Z y_i]; % Concatena el vector con los regresores
end

% U
% Se realiza el mismo procedimiento para u
for i=1:N_reg
    u_i = u(N_reg+1-i:end-i);
    Z = [Z,u_i];
end

% Target
Y = y(N_reg+1:end);

%%
% Dividir en conjuntos de train, test y validación 
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

%% Normalización
% Cálculo de escala
% Z
Z_train_min = min(Z_train);
Z_train_max = max(Z_train);
% Y
Y_train_min = min(Y_train);
Y_train_max = max(Y_train);
% Resta max - min
z_diff = Z_train_max - Z_train_min;
y_diff = Y_train_max - Y_train_min;

% Normalización (re-escala) de conjuntos
% Train
Z_train_norm = (Z_train - Z_train_min)./z_diff;
Y_train_norm = (Y_train - Y_train_min)./y_diff;
% Test
Z_test_norm = (Z_test- Z_train_min)./z_diff;
Y_test_norm = (Y_test- Y_train_min)./y_diff;
% Val
Z_val_norm = (Z_val- Z_train_min)./z_diff;
%Y_val_norm = (Y_val - Y_train_mean)./Y_train_std;

%% Análisis de sensibilidad
% Red inicial
rng('default');
nn_inicial = fitnet(30);
nn_inicial.trainParam.showWindow = 0;
nn_inicial.trainParam.epochs = 1000;
nn_inicial.trainFcn = 'traingdx';
nn_inicial = train(nn_inicial, Z_train_norm', Y_train_norm', 'useParallel', 'no');

y_pred_inicial = nn_inicial(Z_test_norm')';
rmse_inicial = rmse(y_pred_inicial,Y_test_norm);
%% Análisis de sensibilidad
[p,indices] = sensibilidad_nn(Z_train_norm, nn_inicial);
%% Escoger regresores dado un umbral
mu = 1.2; % Umbral que se cambia en función del gráfico
reg_opt = find(indices>mu); % Descartar las variables que no tienen mu
% Actualizar los regresores en función de estos índices
Z_train_norm = Z_train_norm(:,reg_opt);
Z_test_norm = Z_test_norm(:,reg_opt);
Z_val_norm =  Z_val_norm(:,reg_opt);
save("sens_Z.mat", "Z_train_norm", "Z_test_norm","Z_val_norm")
%% Determinar hidden size óptimo

N_hidden_max = 200; % Nro. máximo de neuronas en capa oculta
hidden_interval = 10:10:N_hidden_max; % Intervalo de prueba
error_test = zeros(size(hidden_interval)); % Error en test 
j = 1; % índice para error en test

% Entrenar redes neuronales dado un intervalo para tamaño capa oculta
for i=hidden_interval
    disp(i)
    net = fitnet(i);
    net.trainFcn = 'traingdx';
    net.divideParam.trainRatio = 1.0;
    net.trainParam.epochs = 1000; % 1000 épocas como estándar
    net.trainParam.showWindow=0;
    net = train(net,Z_train_norm',Y_train_norm','useParallel','no');
    y_pred = net(Z_test_norm')';
    error_test(j) = rmse(y_pred, Y_test_norm);
    j = j+1;
end
%% error en test
figure()
plot(hidden_interval, error_test);
title("Error en test")
ylabel("RMSE")
xlabel("Número de neuronas")

%% Red que minimiza el rmse
nn = fitnet(10);
net.divideParam.trainRatio = 1.0;
nn.trainParam.showWindow = 0;
nn.trainFcn = 'traingdx';
nn.trainParam.epochs = 1000;
nn = train(nn, Z_train_norm', Y_train_norm', 'useParallel', 'no');

%%
rmse_final = rmse(nn(Z_test_norm')',Y_test_norm);
%% Predicción a pasos
% Horizonte de predicción
prediction_horz = 18;
% Pasar por todos los datos de validación
n_val = n_data - n_test;
% Se guardan los y(t+j)
predictions_y = zeros(n_val,prediction_horz);
% Ciclo iterativo para todos los datos
input_seq = Z_val_norm;
for j=1:prediction_horz
    % Determino la predicción y(t+j)
    predictions_y(:,j) = nn(input_seq');
    % Los regresores van avanzando junto con j
    input_seq = [predictions_y(:,j),input_seq(:,2:end)];
    
end

predictions_y_unnorm = predictions_y.*y_diff + Y_train_min;
%% Gráficos
y_pred_val = nn(Z_val_norm').*y_diff + Y_train_min;
% Predicciones del modelo
figure()
plot(y_pred_val,'-r')
hold on
plot(Y_val,'.b')
title('Predicciones de salida (RMSE final = 0.0575)')
xlabel('Número de muestra')
ylabel('Valor de y [u.a]')
xlim([0 600])
legend('Predicción','Datos')
savefig('predictions_mlp_y.fig')

% 1 paso
figure()
plot(predictions_y_unnorm(:,1),'-r')
hold on
plot(Y_val,'.b')
title('Predicciones de salida a 1 paso')
xlabel('Número de muestra')
ylabel('Valor de y [u.a]')
xlim([0 600])
legend('Predicción','Datos')
savefig('predict_mlp_1.fig')

% 9 pasos
figure()
plot(predictions_y_unnorm(:,9),'-r')
hold on
plot(Y_val,'.b')
title('Predicciones de salida a 9 pasos')
xlabel('Número de muestra')
ylabel('Valor de y [u.a]')
xlim([0 600])
savefig('predict_mlp_9.fig')

% 18 pasos
figure()
plot(predictions_y_unnorm(:,18),'-r')
hold on
plot(Y_val,'.b')
title('Predicciones de salida a 18 pasos')
xlabel('Número de muestra')
ylabel('Valor de y [u.a]')
xlim([0 600])
legend('Predicción','Datos')
savefig('predict_mlp_18.fig')
%% Métricas
rmse_1 = rmse(predictions_y_unnorm(:,1),Y_val);
rmse_9 = rmse(predictions_y_unnorm(:,9),Y_val);
rmse_18 = rmse(predictions_y_unnorm(:,18),Y_val);

mae_1 = mae(predictions_y_unnorm(:,1),Y_val);
mae_9 = mae(predictions_y_unnorm(:,9),Y_val);
mae_18 = mae(predictions_y_unnorm(:,18),Y_val);
%%
metricss = [rmse_1 rmse_9 rmse_18;mae_1 mae_9 mae_18];
rmse_mlp = metricss(1,:);
mae_mlp = metricss(2,:);
predictions_1 = predictions_y(:,1);
predictions_9 = predictions_y(:,9);
predictions_18 = predictions_y(:,18);
save('neural_network_mlp_def.mat','nn');
save('results_mlp.mat','rmse_mlp','mae_mlp');
%save('predictions_mlp.mat','predictions_1',"predictions_9","predictions_18",'Y_val');
save('normalizers_mlp.mat','Y_train_max','Y_train_min','Z_train_max','Z_train_min')
save('z_sets.mat','Z_train_norm','Z_test_norm','Z_val_norm')
save('y_sets.mat',"Y_train_norm","Y_test_norm","Y_val")