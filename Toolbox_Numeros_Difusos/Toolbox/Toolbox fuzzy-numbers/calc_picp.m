function PICP = calc_picp(y_up, y_lw, y_t)

%% Calcular el PICP segun lo intervalos superiores e inferiores y el valor real
% Recordar que en esta funcion, y_up, y_lw son los intervalos y se
% consideran conocidos
n_data = numel(y_t);  % Según la cantidad de datos
inside_boolean = (y_t <= y_up) & (y_lw <= y_t);
%disp(size(inside_boolean))
PICP = sum(inside_boolean)/n_data;

end