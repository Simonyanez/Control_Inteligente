close all
clear
clc
%%
% Definimos la referencia
nsteps = 10;
total_time = 150;
r = zeros(total_time+nsteps+1,1);
r(1:51) = 0.8;
r(102:end) = 1.5;

%%
rng('default')
lamb_1 = 0.5;

% Construcción de los vectores y(k), u(k), beta(k)
y = zeros(total_time+nsteps+1,1);
u = zeros(total_time+nsteps+1,1);
beta = normrnd(0,0.1,size(y));

% Restricciones: no hay restricciones lineales, sólo cotas para u
A = [];
b = [];
Aeq = [];
beq = [];
noncon = [];
% Cotas inferior y superior
lb = -2*ones(10,1);
ub = 2*ones(10,1);

T = zeros(total_time+1,1);
% Tiempo de inicio
options = optimoptions('fmincon','Display','iter','Algorithm','interior-point');
costos = zeros(total_time+1,1);
for j=3:total_time+1 
    % Vector de solución inicial
    u0 = random('Uniform',-2,2,[10 1]);
    % Inicio de conteo de tiempo
    tic
    % Función objetivo
    costo = @(u_k) cost_function_J(y,u,u_k,r,beta,lamb_1,j,nsteps);
    % Optimización
    [u_opt, fval] = fmincon(costo,u0,A,b,Aeq,beq,lb,ub,noncon,options);
    % Tiempo de optimización
    T(j) = toc;
    % Costo actual
    costos(j-2) = fval;
    % u(t) =  u(t|t)
    u(j) = u_opt(1);
    % Actualiza la salida
    y(j) = (0.8 - 0.5*exp(-y(j-1)^2))*y(j-1) + (0.3-0.9*exp(-y(j-1)^2))*y(j-2) + ...
    u(j-1) + 0.2*u(j-2) + 0.1*u(j-1)*u(j-2) + 0.5*exp(-y(j-1)^2)*beta(j);
end

y = y(1:end-nsteps);
r = r(1:end-nsteps);
u = u(1:end-nsteps);
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
title(["Costo para algoritmo Interior Point, valor de \lambda=",num2str(lamb_1)]);
xlabel("Muestra")
ylabel("Costo [u.a]")