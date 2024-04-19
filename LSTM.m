function model = LSTM
    data = load("data_p1.mat");
    y = data.y;
    u = data.u

    
    numFeatures = 12;
    numHiddenUnits = 125;
    numResponses = 1;
    %Split Train, Test, Val

    % Layer Array Neural Networ Architecture
    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits,OutputMode="sequence")
        fullyConnectedLayer(numResponses)];
    
    % Training
end