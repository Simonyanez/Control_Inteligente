function [int_up, int_lw] = calc_intervals(zin, a, b, g, s_up, s_lw)
% calc_intervals(zin, a, b, s_up, s_lw) calcula los intervalos superiores e
% inferiores utilizando solo los spreads, para luego sumarlo/restarlo de la
% prediccion del valor esperado
% Salidas:
%   - int_up: Intervalo calculado con spreads superior tal que y_up = y_p + int_up
%   - int_lw: Intervalo calculado con spreads inferiores tal que y_up = y_p - int_lw
% Variables
%   - zin: matriz de regresores siendo el primer regresor data_y(k-1)
%   - a,b: Desviacion estandar y media de los clusters del modelo difuso.
%   - s_up, s_lw: Matriz que contiene los spreads superiores e inferiores.
%                 Esta matriz contiene los parametros de cada regla en sus
%                 filas

%% Calcular grados de activación normalizados (N_datos x N_reglas)
% Utilicen los parametros a,b del modelo difuso para calcular los grados de
% activacion normalizados de cada regla
Wn = mf(zin,a,b);
%disp(size(Wn))
%% Calcular la salida de cada regla, pero ahora utilizando los spreads
% s_j = s_0 + s_1 y(k-1) + s_2 y(k-2)...
% recordar que zin contiene los regresores [y(k-1), y(k-2),...]
% ademas, s_up, s_lw consideran un parametro afin
s_up = normalize(s_up);
s_lw = normalize(s_lw);
% Spreads con regresores sin considerar activación
s_j_up = s_up*[ones(1,size(zin,1));abs(zin)'];
s_j_lw = s_lw*[ones(1,size(zin,1));abs(zin)'];
%disp(size(s_j_up))
%disp(size(b))
% Salida de cada regla considerando activación
y_j_up = g*[ones(1,size(zin,1));zin'] + s_j_up;
y_j_lw = g*[ones(1,size(zin,1));zin'] - s_j_lw;

%disp(size(y_j_up))
% Estaríamos calculando los intervalos considerando que ya tenemos el
% modelo esperado y de ahí salen los spreads (?).

%% Utilizar la salida de cada regla para calcular la salida global
% deben utilizar los s_j_up,s_j_lw calculados anteriormente y ponderarlos
% por los grados de activacion de su respectiva regla
% finalmente se suman todas las reglas ponderadas

% Combinación lineal de salidas,entrega intervalos

int_up = sum(Wn*y_j_up,2);  
int_lw = sum(Wn*y_j_lw,2);

end