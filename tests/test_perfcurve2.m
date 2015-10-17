% Make sure perfcurve() and perfcurve2() agree for small data sets.

% This bit comes directly from the perfcurve() man page
load fisheriris 
x = meas(51:end,1:2);        % iris data, 2 classes and 2 features
y = (1:100)'>50;             % versicolor=0, virginica=1
b = glmfit(x,y,'binomial');  % logistic regression
p = glmval(b,x,'logit');     % get fitted probabilities for scores

% convert {'setosa', 'virginica'} -> {0,1}
yTrue = double(strcmp(species(51:end), 'virginica'));

[x1,y1] = perfcurve(yTrue,p,1);
[x2,y2] = perfcurve2(yTrue,p);

assert(abs(trapz(x1,y1) - trapz(x2,y2)) < 1e-3);
fprintf('[%s]: all tests passed!\n', mfilename);

% visually confirm these two calculations are close
figure;
plot(x1, y1, '-o', x2, y2, '-o');
xlabel('pfa'); ylabel('pd');

