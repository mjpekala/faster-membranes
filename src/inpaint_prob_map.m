% INPAINT_PROB_MAP  Inpaints missing membrane probabilities.
%
%   Use this to generate a complete probability cube from a
%   cube of partial evaluations (generated by emcnn.py).
%
%   Any value in the input cube < 0 will be inpainted.
%
% October 2015, mjp

addpath('./inpaint');
addpath('./tight_subplot');

[fn,path] = uigetfile('*.mat', 'Select CNN probability cube');
load(fullfile(path, fn));


% get just the membrane estimates
Yhat = squeeze(Yhat(2,:,:,:));
Yhat = permute(Yhat, [2 3 1]);  % python -> matlab ordering
Yhat(Yhat < 0) = NaN;

fprintf('[%s]: recovering missing values (%0.2f%% of volume)\n', ...
  mfilename, 100*sum(isnan(Yhat(:))) / numel(Yhat));

tic
Yrepaired = zeros(size(Yhat));
for ii = 1:size(Yhat,3)
  Yi = Yhat(:,:,ii);

  % do inpainting; remap output to [0,1]
  Yr = inpaintn(Yi);
  Yr = Yr - min(Yr(:));
  Yr = Yr / (max(Yr(:)) - min(Yr(:)));

  % use original estimates, where available
  Yr(isfinite(Yi)) = Yi(isfinite(Yi));

  Yrepaired(:,:,ii) = Yr;

  fprintf('[%s]: finished slice %d (of %d); total time: %0.2f sec\n', ...
    mfilename, ii, size(Yhat,3), toc);
end

fOut = fullfile(path, [fn '.repaired.mat']);
save(fOut, 'Yrepaired', '-v7.3');

fprintf('[%s]: Result written to "%s"\n', mfilename, fOut);


% visualize result
figure('Position', [200 200 800 400]);
ha = tight_subplot(1, 2, [.03, .03]);

axes(ha(1));
imagesc(Yhat(:,:,1));
title(sprintf('input: slice 1'))
set(gca, 'Xtick', [], 'Ytick', []);
  
axes(ha(2));
imagesc(Yrepaired(:,:,1));
title(sprintf('inpainted: slice 1'))
set(gca, 'Xtick', [], 'Ytick', []);
  
