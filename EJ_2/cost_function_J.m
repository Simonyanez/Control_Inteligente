function J = cost_function_J(y,u_pasado,u_k,r,beta,lamb,j,nsteps)
% Toma [u(k-1),u(k)...u(k+j-1)]
u_t = [u_pasado(j-2);u_pasado(j-1);u_k];
y_t = y(j-2:j+nsteps);
r_t = r(j+1:j+nsteps);
% Construcción de predicciones
for i=3:nsteps+1
    y_t(i) = (0.8 - 0.5*exp(-y_t(i-1)^2))*y_t(i-1) + (0.3-0.9*exp(-y_t(i-1)^2))*y_t(i-2) + u_t(i-1) + 0.2*u_t(i-2) ...
    + 0.1*u_t(i-1)*u_t(i-2) + 0.5*exp(-y_t(i-1)^2)*beta(i);
end

% Cálculo de costos
cost_y = (y_t(4:end)-r_t).^2;

cost_u = lamb*u_k.^2;

% Suma total de costos
J = sum(cost_y + cost_u);
end