function [s_up, s_lw] = sintonizacion_fn(data_y, zin,model, CP)
% sintonizacion_fn calcula los spreads s_sup, s_inf para aplicar el metodo
% de numeros difusos a un modelo difuso Takagi-Sugeno
% Variables:
%   - data_y = vector de datos de salida N x 1
%   - data_y_reg = matriz de regresores para la salida N x N_regs_y
%   - data_u = vector de datos de entrada N x 1
%   - data_u_reg = matriz de regresores para la entrada N x N_regs_u
%   - model = parámetros del modelo difuso que entrega la funcion
%             TakagiSugeno
%   - CP = Porcentaje de cobertura deseado por los intervalos en decimal

%% Generar predicciones
% En este ejemplo se utiliza la prediccion para y(k), resultando en un
% intervalo para y(k). Para generar intervalos a más pasos se debe primero
% obtener la prediccion a y(k+j)
y_p = ysim(zin, model.a, model.b, model.g);

%% Se inician los parámetros para el problema de optimización
options = optimoptions('particleswarm','Display','iter');
options.SwarmSize=100;
options.FunctionTolerance = 1e-3;
options.MaxStallIterations=15;
options.UseParallel=true; % para usar varios procesos a la vez

% Cantidad de variables a optimizar
[a,b]=size(model.g);
nvars=2*a*b; % 2*( (numero de regresores + 1) * numero de reglas)

% Ponderadores eta1, eta2 de la función de costos
param.eta1=100;
param.eta2=200;

% Porcentaje de cobertura deseado
param.coverage=CP;

%% Se define la funcion de costos
fun= @(spreads) fn_obj_sint(spreads, data_y, y_p, zin, model, param);

%% Limites para el optimizador
lb=zeros(nvars,1);
ub=100*ones(nvars,1);

%% Optimizador
[sol,fval,exitflag,output] = particleswarm(fun, nvars, lb, ub, options);

% Separar spreads optimos en superiores e inferiores
[N_reglas, N_regresores]=size(model.g);
n_spreads = N_reglas*N_regresores;
s_up = sol(1:n_spreads);
s_lw = sol(n_spreads+1:end);

% Transformar vector en matriz
s_up = reshape(s_up, N_reglas, N_regresores); % Matriz de spreads (N_reglas x N_regresores)
s_lw = reshape(s_lw, N_reglas, N_regresores); % Matriz de spreads (N_reglas x N_regresores)

end