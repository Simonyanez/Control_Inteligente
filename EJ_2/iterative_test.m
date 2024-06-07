lambdas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
swarm_sizes = [25,50,75,100,200];

%%
for iter = 1:numel(lambdas)
    lambda = lambdas(iter);
    [y, r, u, costos] = pso_tarea2(lambda,50);

    % Gráficos para cada lambda
    % Salida y Referencia
    figure;
    subplot(3,1,1);
    plot(y);
    hold on;
    plot(r);
    legend(["Salida","Referencia"],'Location','best');
    xlabel("Muestra");
    ylabel("Valor de salida [u.a]");
    title(["Señal de salida, \lambda = ", num2str(lambda)]);
    grid("on");
    ylim([-0.5,2.0]);
    hold off;

    % Señal de entrada
    subplot(3,1,2);
    plot(u);
    grid("on");
    title(["Señal de entrada, \lambda = ", num2str(lambda)]);
    xlabel("Muestra");
    ylabel("Valor de entrada [u.a]");
    ylim([-0.3,1.0]);

    % Costos
    subplot(3,1,3);
    plot(costos);
    grid("on");
    title(["Costo para algoritmo PSO, \lambda = ", num2str(lambda)]);
    xlabel("Muestra");
    ylabel("Costo [u.a]");
    ylim([0,6.0]);
    drawnow; % Actualiza las figuras en cada iteración
end

%%
for iter = 1:numel(swarm_sizes)
    swarm_s = swarm_sizes(iter);
    lambda = 0.1;
    [y, r, u, costos] = pso_tarea2(lambda,swarm_s);

    % Gráficos para cada lambda
    % Salida y Referencia
    figure;
    subplot(3,1,1);
    plot(y);
    hold on;
    plot(r);
    legend(["Salida","Referencia"],'Location','best');
    xlabel("Muestra");
    ylabel("Valor de salida [u.a]");
    title(["Señal de salida, \lambda = ", num2str(lambda), "N° de partículas = ", num2str(swarm_s)]);
    grid("on");
    ylim([-0.5,2.0]);
    hold off;

    % Señal de entrada
    subplot(3,1,2);
    plot(u);
    grid("on");
    title(["Señal de entrada, \lambda = ", num2str(lambda), "N° de partículas = ", num2str(swarm_s)]);
    xlabel("Muestra");
    ylabel("Valor de entrada [u.a]");
    ylim([-0.3,1.0]);

    % Costos
    subplot(3,1,3);
    plot(costos);
    grid("on");
    title(["Costo para algoritmo PSO, \lambda = ", num2str(lambda), "N° de partículas = ", num2str(swarm_s)]);
    xlabel("Muestra");
    ylabel("Costo [u.a]");
    ylim([0,6.0]);
    drawnow; % Actualiza las figuras en cada iteración
end