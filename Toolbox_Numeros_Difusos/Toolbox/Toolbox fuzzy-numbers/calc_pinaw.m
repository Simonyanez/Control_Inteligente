function PINAW = calc_pinaw(y_up, y_lw, y_t)
%% Calcular el PINAW segun lo intervalos superiores e inferiores y el valor real
% Recordar que en esta funcion, y_up, y_lw son los intervalos y se
% consideran conocidos

%Implementación basada en PPT Aplicación de Intervalos Difusos.pdf diapositiva 24. PICP es Prediction Interval Coverage Probability.

%Parameters:
%    - y_up = intervalo superior
%    - y_lw = intervalo inferior
%    - y_t = valor real

n_data = numel(y_t);  % Según la cantidad de datos
R = max(y_t) - min(y_t);
dist = sum(y_up - y_lw);
PINAW = dist/(n_data*R);


end