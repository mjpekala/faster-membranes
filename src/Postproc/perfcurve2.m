function [fpr, recall, precision] = perfcurve2(Y, Scores)
% PERFCURVE2  A faster (but less feature rich) version of Matlab's perfcurve()
%
% October 2015, mjp

% Assumes class labels live in [0,1]
% Any score < 0 will be ignored in this analysis.
yAll = unique(Y);
assert(sum(yAll >= 0) == 2);
assert(length(intersect([0 1], yAll)) == 2);


thresh = fliplr(0:.01:1);

Yhat = bsxfun(@ge, Scores(:), thresh);  % scores -> {0,1}

nPos = sum(Y(:) == 1);
nNeg = sum(Y(:) == 0);

% Precision:  |relevant ^ retrieved| / |retrieved|
% Recall/TPR: |relevant ^ retrieved| / |relevant|
% FPR:        |false pos.| / |negative|
precision = sum(bsxfun(@and, Yhat==1, Y(:) == 1), 1) ./ sum(Yhat==1,1);
recall = sum(bsxfun(@and, Yhat==1, Y(:) == 1), 1) / nPos;
fpr = sum(bsxfun(@and, Yhat==1, Y(:) == 0), 1) / nNeg;

