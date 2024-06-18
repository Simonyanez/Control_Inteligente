close all
clear
clc
%%
% Definimos la referencia
k_reg = 1;
nsteps = 10;
total_time = 151;
r = zeros(total_time+k_reg+1,1);
r(1:51) = 0.8;
r(102:end) = 1.5;
%%
rng('default')
lamb_1 = 0.35;

% Construcción de los vectores y(k), u(k), beta(k)
y = zeros(total_time+k_reg+1,1);
u = zeros(total_time+k_reg+1,1);
beta = normrnd(0,0.1,size(y));

% Parametros de la optimización

% Número de variables
nVar = 10;          
VarSize = [1 nVar];

% Cotas inferior y superior
VarMin = -2;
VarMax = 2;

%Máximo iteraciones
MaxIt = 200;

% Tamaño de la población
nPop = 50;

T = zeros(total_time+1,1);
% Tiempo de inicio
costos = zeros(total_time,1);
for j=2:total_time+1
    % Inicio de conteo de tiempo
    tic
    % Función objetivo
    costo = @(u_k) cost_function_J(y,u,u_k',r(j),lamb_1,j,nsteps);
    % Optimización
    [u_opt, fval] = tlbo(costo, nVar, VarSize, VarMin, VarMax, MaxIt, nPop);
    % Tiempo de optimización
    T(j) = toc;
    % Costo actual
    costos(j-1) = fval;
    % u(t) =  u(t|t)
    u(j) = u_opt(1);
    % Actualiza la salida
    y(j+1) = (0.8 - 0.5*exp(-y(j)^2))*y(j) + (0.3-0.9*exp(-y(j)^2))*y(j-1) + ...
    u(j) + 0.2*u(j-1) + 0.1*u(j)*u(j-1) + 0.5*exp(-y(j)^2)*beta(j+1);
end
%%
y = y(2:end-1);
r = r(2:end-1);
u = u(2:end-1);
T = T(2:end);
costos = costos(2:end);

%% Gráficos

figure(1)
plot(y)
hold on
plot(r)
legend(["Salida","Referencia"],'Location','best')
xlabel("Muestra")
ylabel("Valor de salida [u.a]")
title("Señal de salida")
grid("on")
hold off

%%
figure(2)
plot(u)
grid("on")
title("Señal de entrada")
xlabel("Muestra")
ylabel("Valor de entrada [u.a]")

%%
figure(3)
plot(costos)
grid("on")
title(["Costo para algoritmo TLBO, valor de \lambda=",num2str(lamb_1)]);
xlabel("Muestra")
ylabel("Costo [u.a]")