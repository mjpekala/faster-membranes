function Yr = regression_labels(Y)
% REGRESSION_LABELS  Maps binary class labels to regression labels.

if ~any(Y(:) == 1), error('no labels of 1 in this volume!'); end
if length(size(Y)) ~= 3, error('not a 3d volume'); end


Yr = NaN * ones(size(Y));
Yr(Y==1) = 0;

for ii = 1:size(Y,3)
    fprintf('[%s]: mapping slice %d\n', mfilename, ii);
    Yi = Yr(:,:,ii);
    Yr(:,:,ii) = manhattan_fill(Yi);
end

end % regression_labels()


function Xout = manhattan_fill(X)
shift_up = @(X) X([2:end 1], :);
shift_down = @(X) X([end 1:end-1], :);
shift_left = @(X) X(:, [2:end 1]);
shift_right = @(X) X(:, [end 1:end-1]);

Xout = X;

while any(isnan(Xout(:)))
    Xn = shift_down(Xout);
    Xs = shift_up(Xout);
    Xe = shift_right(Xout);
    Xw = shift_left(Xout);

    Xne = shift_right(Xn);
    Xnw = shift_left(Xn);
    Xse = shift_right(Xs);
    Xsw = shift_left(Xs);
    
    NN = min(cat(3, Xn, Xs, Xe, Xw, Xne, Xnw, Xse, Xsw), [], 3, 'omitnan');
   
    Xout(isnan(Xout)) = NN(isnan(Xout)) + 1;
end

end  % manhattan_fill
