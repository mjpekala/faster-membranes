function view_train_labels(Y)
% VIEW_TRAIN_LABELS   View impact of different thresholds on 
%                     the training data labels
%
%     view_train_labels(Y)
%
%  where:
%     Y := an (m x n x p) tensor of p images

% Nov 2015, mjp

    
assert(ndims(Y) == 3);
[m,n,p] = size(Y);

idx0 = 1;
thresh0 = [0 .7];


figure; hist(Y(:)); title('Distribution of scores');
xlabel('score');


fig = figure('Position', [200, 200, 600, 500]);

sZ = uicontrol('Style', 'slider', ...
                'Min', 1, 'Max', p, ...
                'Value', idx0, ...
                'SliderStep', [1 / (p-1) 1], ...
                'Units', 'Normalized', ...
                'Position', [.05 .95 .35 .04], ...
                'Callback', @slider_cb);


% threshold lower and upper bounds
sLB = uicontrol('Style', 'slider', ...
                'Min', 0, 'Max', 1, ...
                'Value', thresh0(1), ...
                'Units', 'Normalized', ...
                'Position', [.05 .03 .15 .04], ...
                'Callback', @slider_cb);

tLB = uicontrol('Style', 'text', ...
                'String', sprintf('lower bound: %0.2f', thresh0(1)), ...
                'Units', 'Normalized', ...
                'Position', [.20 .03 .15 .04]);

sUB = uicontrol('Style', 'slider', ...
                'Min', 0, 'Max', 1, ...
                'Value', thresh0(2), ...
                'Units', 'Normalized', ...
                'Position', [.55 .03 .15 .04], ...
                'Callback', @slider_cb);

tUB = uicontrol('Style', 'text', ...
                'String', sprintf('upper bound: %0.2f', thresh0(2)), ...
                'Units', 'Normalized', ...
                'Position', [.70 .03 .15 .04]);



plot_slice(idx0, thresh0(1), thresh0(2));


function slider_cb(source, callbackdata)
  layer = round(get(sZ, 'Value'));
  lb = get(sLB, 'Value');
  ub = get(sUB, 'Value');
  
  set(tLB, 'String', sprintf('lower bound: %0.2f', lb));
  set(tUB, 'String', sprintf('upper bound: %0.2f', ub));
  
  plot_slice(layer, lb, ub);
end  % slider_cb()



function plot_slice(z, lb, ub)
    Yz = Y(:,:,z);
    Ythresh = -1*ones(size(Yz));
    Ythresh(Yz <= lb) = 0;
    Ythresh(Yz >= ub) = 1;
    
    figure(fig);
    imagesc(Ythresh)
    colorbar;
    set(gca, 'XTick', [], 'YTick', []);
    title(sprintf('slice %d / %d', z, p));
end


end
