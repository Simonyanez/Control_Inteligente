%%
function [y,r,u,costos] = pso_tarea2(lamb_1,swarm_s)
%lamb_1 = 0.35; % Penalización de la entrada

%%
% Definimos la referencia
nsteps = 10;
total_time = 150;
r = zeros(total_time+nsteps+1,1);
r(1:51) = 0.8;
r(102:end) = 1.5;

%%
rng('default')
% Construcción de los vectores y(k), u(k), beta(k)
y = zeros(total_time+nsteps+1,1);
u = zeros(total_time+nsteps+1,1);
beta = normrnd(0,0.1,size(y));

% Cotas inferior y superior
lb = -2*ones(10,1);
ub = 2*ones(10,1);

T = zeros(total_time+1,1);
% Tiempo de inicio
options = optimoptions('particleswarm','Display','iter','SwarmSize',swarm_s);
costos = zeros(total_time+1,1);
for j=3:total_time+1 
    % Inicio de conteo de tiempo
    tic
    % Función objetivo
    costo = @(u_k) cost_function_J(y,u,u_k',r,beta,lamb_1,j,nsteps);  % Recibe u_k entrada
    % Optimización
    [u_opt, fval] = particleswarm(costo,10,lb,ub,options);
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


end