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
lambdas = [0.1 0.35 0.5 1.0];
n_l = length(lambdas);

% Construcción de los vectores y(k), u(k), beta(k)
y = zeros(total_time+k_reg+1,n_l);
u = zeros(total_time+k_reg+1,n_l);
beta = normrnd(0,0.1,[total_time+k_reg+1 1]);

% Restricciones: no hay restricciones lineales, sólo cotas para u
A = [];
b = [];
Aeq = [];
beq = [];
noncon = [];
% Cotas inferior y superior
lb = -2*ones(nsteps,1);
ub = 2*ones(nsteps,1);

T = zeros(total_time+1,n_l);
% Tiempo de inicio
options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
costos = zeros(total_time+1,n_l);
for i=1:n_l
    for j=2:total_time+1
        % Vector de solución inicial
        u0 = random('Uniform',-2,2,[10 1]);
        % Inicio de conteo de tiempo
        tic
        % Función objetivo
        costo = @(u_k) cost_function_J(y(:,i),u(:,i),u_k,r(j),lambdas(i),j,nsteps);
        % Optimización
        [u_opt, fval] = fmincon(costo,u0,A,b,Aeq,beq,lb,ub,noncon,options);
        % Tiempo de optimización
        T(j,i) = toc;
        % Costo actual
        costos(j-1,i) = fval;
        % u(t) =  u(t|t)
        u(j,i) = u_opt(1);
        % Actualiza la salida
        y(j+1,i) = (0.8 - 0.5*exp(-y(j,i)^2))*y(j,i) + (0.3-0.9*exp(-y(j,i)^2))*y(j-1,i) + ...
        u(j,i) + 0.2*u(j-1,i) + 0.1*u(j,i)*u(j-1,i) + 0.5*exp(-y(j,i)^2)*beta(j+1);
    end
end
%% Vectores finales 
y = y(2:end-1,:);
r = r(2:end-1,:);
u = u(2:end-1,:);
T = T(2:end,:);
costos = costos(2:end,:);

%% Métricas
var_u_lamb = std(u);
j_y_lamb = mean((y-r.*ones(size(y))).^2);
j_u_lamb = mean(diff(u).^2);

tot_time = sum(T,1);

% Máximo overshoot
ovs_lamb = zeros(1,size(y,2));
for i=1:numel(ovs_lamb)
    k = find(y(:,i) == max(y(:,i)));
    ovs_lamb(i) = 100*(y(k,i)-r(k))/r(k);
end
%% Gráficos

figure(1)
plot(y)
hold on
plot(r, LineWidth=2,Color='Black',LineStyle='--')
legend(["\lambda = 0.1", "\lambda = 0.35", "\lambda = 0.5", "\lambda = 1","Referencia"],'Location','best')
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
title("Costos para algoritmo SQP con distintos valores de \lambda");
xlabel("Muestra")
ylabel("Costo [u.a]")

%%
figure(4)
lambss = ["0.1" "0.35" "0.5" "1.0"];
lambss = categorical(lambss);
bar(lambss,sum(T,1))
title("Tiempos de ejecución para distintos valores de \lambda")

%% Valor lambda fijo, cambio de sol. inicial
lamb_1 = 0.35;
r = zeros(total_time+k_reg+1,1);
r(1:51) = 0.8;
r(102:end) = 1.5;

u0s = [random('Uniform',-2,2,[10 1]) zeros(10,1) 1.5*ones(10,1) -2*ones(10,1)];
n_v = size(u0s,2);

y_vecs = zeros(total_time+k_reg+1,n_v);
u_vecs = zeros(total_time+k_reg+1,n_v);

T_vs = zeros(total_time+1,n_v);
cost_v = zeros(total_time+1,n_v);

for i=1:n_v
    for j=2:total_time+1
        % Vector de solución inicial
        u0 = u0s(:,i);
        % Inicio de conteo de tiempo
        tic
        % Función objetivo
        costo = @(u_k) cost_function_J(y_vecs(:,i),u_vecs(:,i),u_k,r(j),lamb_1,j,nsteps);
        % Optimización
        [u_opt, fval] = fmincon(costo,u0,A,b,Aeq,beq,lb,ub,noncon,options);
        % Tiempo de optimización
        T_vs(j,i) = toc;
        % Costo actual
        cost_v(j-1,i) = fval;
        % u(t) =  u(t|t)
        u_vecs(j,i) = u_opt(1);
        % Actualiza la salida
        y_vecs(j+1,i) = (0.8 - 0.5*exp(-y_vecs(j,i)^2))*y_vecs(j,i) + (0.3-0.9*exp(-y_vecs(j,i)^2))*y_vecs(j-1,i) + ...
        u_vecs(j,i) + 0.2*u_vecs(j-1,i) + 0.1*u_vecs(j,i)*u_vecs(j-1,i) + 0.5*exp(-y_vecs(j,i)^2)*beta(j+1);
    end
end

%% Vectores finales 
y_vecs = y_vecs(2:end-1,:);
r = r(2:end-1,:);
u_vecs = u_vecs(2:end-1,:);
T_vs = T_vs(2:end,:);
cost_v = cost_v(2:end,:);
%% Métricas
var_u_vec = std(u_vecs);
j_y_vec = mean((y_vecs-r.*ones(size(y_vecs))).^2);
j_u_vec = mean(diff(u_vecs).^2);

tot_time_vec = sum(T_vs,1);

% Máximo overshoot
ovs_vec = zeros(1,size(y_vecs,2));
for i=1:numel(ovs_vec)
    k = find(y_vecs(:,i) == max(y_vecs(:,i)));
    ovs_vec(i) = 100*(y_vecs(k,i)-r(k))/r(k);
end
%% Gráficos

figure(5)
plot(y_vecs)
hold on
plot(r,LineWidth=2,Color='Black',LineStyle='--')
legend(["u_0 \sim Unif([-2,2])", "u_0 = 0", "u_0 = 1.5", "u_0 = 2","Referencia"],'Location','best')
xlabel("Muestra")
ylabel("Valor de salida [u.a]")
title("Señal de salida")
grid("on")
hold off

%%
figure(6)
plot(u_vecs)
grid("on")
title("Señal de entrada")
xlabel("Muestra")
ylabel("Valor de entrada [u.a]")

%%
figure(7)
plot(cost_v)
grid("on")
title("Costos para algoritmo SQP con distintos valores de u_0 (\lambda = 0.35)");
xlabel("Muestra")
ylabel("Costo [u.a]")

%%
figure(8)
vss = ["Uniforme" "0" "1.5" "Extremo (-2)"];
vss = categorical(vss);
bar(vss,sum(T_vs,1))
title("Tiempos de ejecución para distintos valores de u_0")