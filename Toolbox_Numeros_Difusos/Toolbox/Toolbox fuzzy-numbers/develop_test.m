
% Test cases for calc_picp and calc_pinaw functions
% Test case 1: Perfect coverage for PICP, PINAW should be 1.0
y_up = [2, 4, 6, 8];  % Upper bounds
y_lw = [0, 2, 4, 6];  % Lower bounds
y_t = [1, 3, 5, 7];   % True values
assert(calc_picp(y_up, y_lw, y_t) == 1.0);
assert(calc_pinaw(y_up, y_lw, y_t) == 1.0);

% Test case 2: No coverage for PICP, PINAW should be NaN
y_up = [1, 2, 3, 4];  % Upper bounds
y_lw = [1, 2, 3, 4];  % Lower bounds
y_t = [5, 6, 7, 8];   % True values
assert(calc_picp(y_up, y_lw, y_t) == 0.0);
assert(isnan(calc_pinaw(y_up, y_lw, y_t)));

% Test case 3: Partial coverage for PICP, PINAW should be between 0 and 1
y_up = [3, 6, 9, 12];  % Upper bounds
y_lw = [1, 4, 7, 10];  % Lower bounds
y_t = [2, 7, 6, 11];   % True values
picp = calc_picp(y_up, y_lw, y_t);
pinaw = calc_pinaw(y_up, y_lw, y_t);
assert(picp > 0 && picp < 1);
assert(pinaw >= 0 && pinaw <= 1);

disp('All tests passed!');
