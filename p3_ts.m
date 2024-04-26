close all
clear
clc
addpath(".\Toolbox_TS_y_MLP\Toolbox difuso\")
addpath(".\Toolbox_Numeros_Difusos\Toolbox\Toolbox fuzzy-numbers\")
%addpath("\Toolbox_NN\")
%%  
%Se cargan los datos entrada y salida
data = load('data_p1.mat');
y = data.y;
u = data.u;

% Warm start: descartar los primeros 1000 datos
%y = y(1000:end);
%u = u(1000:end);
%% Selección de variables relevantes

%Se entrena un modelo inicial
N_reg = 5; % inicialización número de regresores para (y,u)
N_datos = length(u)-N_reg; % Número de datos, se le quita la cantidad de regresores
N_train = floor(N_datos*0.6); % Porcentaje de train: 60, test: 25, val: 15
N_test = floor(N_datos*0.85);
Max_reglas = 40; %Número máximo de reglas (clusters difusos)


% Vector que almacena los regresores
Z = [];

for i = 1:N_reg %Se rellena el vector de regresores con los regresores de la salida(y)
   y_i= y(N_reg+1-i:end-i);
   Z = [Z, y_i];
end

for i = 1:N_reg %Se rellena el vector de regresores con los regresores de la entrada(u)
   u_i= u(N_reg+1-i:end-i);
   Z = [Z, u_i];
end

Y = y(N_reg+1:end); %Vector de salida
U = u(N_reg+1:end); %Vector de entrada

%Conjuntos de regresores
Z_ent = Z(1:N_train,:); 
Z_test = Z(N_train+1:N_test,:);
Z_val = Z(N_test + 1 :end,:);

%Conjuntos de salidas
Y_ent = Y(1:N_train,:);
Y_test = Y(N_train+1:N_test,:);
Y_val = Y(N_test + 1 :end,:);
%%
[p, indices, ~] = sensibilidad(Y_ent, Z_ent, 18,[1 2 2] );
%Elegimos los regresores con sensibilidad mayor a 1 en este caso, puede ser
%un numero distinto dependiendo de como se distribuye la sensibilidad
gtn_1 = find(indices>1);
%Elegimos solo los regresores a ocupar
Z_optim_ent = Z_ent(:, gtn_1);
Z_optim_val = Z_val(:, gtn_1);
%% Modelo TS 
[modelTS, ~] = TakagiSugeno(Y_ent, Z_optim_ent, 18, [1 2 2]);

%%
CP = 0.9;
[s_up, s_lw] = sintonizacion_fn(Y_ent,Z_optim_ent, modelTS, CP);

%%
% Prediccion valor esperado
y_ts = ysim(Z_optim_val,modelTS.a,modelTS.b,modelTS.g);

% Intervalos
[int_up, int_lw] = calc_intervals(Z_optim_val, modelTS.a, modelTS.b, modelTS.g, s_up, s_lw);
y_up = y_ts + int_up;
y_lw = y_ts - int_lw;

%%
f = figure;
grid on
hold on

plot(Y_val, '.')
hold on
plot(y_ts, 'LineWidth',2)
hold on
plot(y_up, '-k')
hold on
plot(y_lw, '-k') % Para graficar un area revisar la funcion 'fill'

xlabel('Muestras')
ylabel('Salida')
xlim([1000 1500])
legend('Valor real', 'Prediccion', 'Location','best');
title(['RMSE = ' num2str(rmse(Y_val, y_ts))])
hold off