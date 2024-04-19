function PINAW = calc_pinaw(y_up, y_lw, y_t)

%Implementación basada en PPT Aplicación de Intervalos Difusos.pdf diapositiva 24. PICP es Prediction Interval Coverage Probability.

%Parameters:
%    - y_up = intervalo superior
%    - y_lw = intervalo inferior
%    - y_t = valor real

%% Calcular el PINAW segun lo intervalos superiores e inferiores y el valor real
% Recordar que en esta funcion, y_up, y_lw son los intervalos y se
% consideran conocidos

[~,n_data] = size(y_t);  % Según la cantidad de datos
R = max(y_up) - max(y_lw);
dist = sum(y_up - y_lw);
PINAW = dist/(n_data*R);

end