function J = cost_function_J(y,u_pasado,u_k,r,lamb,j,nsteps)
% Toma [u(k-1),u(k)...u(k+j-1)]
u_t = [u_pasado(j-1);u_k];
y_t = zeros(nsteps+2,1);
y_t(1) = y(j-1);
y_t(2) = y(j);
%disp(size(y_t))
r_t = r*ones(nsteps,1);
%disp(size(r_t))
% Construcción de predicciones
for i=2:nsteps
    y_t(i+1) = (0.8 - 0.5*exp(-y_t(i)^2))*y_t(i) + (0.3-0.9*exp(-y_t(i)^2))*y_t(i-1) + u_t(i) + 0.2*u_t(i-1) ...
    + 0.1*u_t(i)*u_t(i-1);
end
%disp(y_t(end))
% Cálculo de costos
cost_y = (y_t(3:end)-r_t).^2;

cost_u = lamb*u_k.^2;

% Suma total de costos
J = sum(cost_y + cost_u);
end