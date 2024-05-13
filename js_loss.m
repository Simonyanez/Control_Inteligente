function [loss_upper,loss_crisp,loss_lower] = js_loss(predictions,targets)
    % Assumes that the predictions are Nx3 matrix lower,crisp and upper
    % Lambda hyperparameter must be changed in every iteration
    lambda = 0.1;
    N = length(predictions);
    predictions = double(predictions);
    targets = double(targets);

    % Lower squared ReLu
    low_diff = targets-predictions(1,:);
    lower = sum(max(0, low_diff).^2)/N;
    lower_mse = sum((targets-predictions(1,:)).^2)/N ;

    % Upper squared ReLU
    up_diff = targets-predictions(3,:);
    upper = sum(max(0, up_diff).^2)/N;
    upper_mse = sum((targets-predictions(3,:)).^2)/N;

    % Custom loss
    loss_crisp = sum((targets-predictions(2,:)).^2)/N;
    loss_lower = lower_mse + lambda*lower;
    loss_upper = upper_mse + lambda*upper;

end