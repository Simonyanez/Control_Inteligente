function [U_set] = generate_u_vals(offset, numPredictionTimeSteps)
    U_set = zeros(numPredictionTimeSteps, 1);  % Initialize U_set as a column vector
    for i = offset:offset+numPredictionTimeSteps-1  % Adjust loop range
        angle = 2*pi*i/25;
        val = sin(angle);
        U_set(i-offset+1) = val;  % Adjust indexing
    end
    U_set = U_set + 0.1 * randn(size(U_set));  % Add Gaussian noise
end
