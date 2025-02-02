% Experiment to determine how far we can push the 
% subsampling+inpainting approach.

addpath('./inpaint');
addpath('./tight_subplot');

% Parameters
param=struct();
param.emFile = '../../Data/ISBI2012/ISBI_Train20/Xvalid.mat';
param.truthFile = '../../Data/ISBI2012/ISBI_Train20/Yvalid.mat';
param.interactive = 1;

param.pctToTry = [.25 .5 .6:.05:.95];

%-------------------------------------------------------------------------------
%% Load data

%----------------------------------------
% load the ground truth
%----------------------------------------
load(param.emFile);
Xvalid = permute(Xvalid, [2 3 1]);  % python -> matlab ordering
load(param.truthFile);
Yvalid = permute(Yvalid, [2 3 1]);  % python -> matlab ordering

if 0 
    fprintf('[%s]: doing feature analysis - please wait a few moments...\n', mfilename)
    plot_feat_distribution(double(Xvalid(:)), double(Yvalid(:)));
end


%----------------------------------------
% load membrane probabilities generated by Caffe
%----------------------------------------
if param.interactive
  [fn,path] = uigetfile('*.mat', 'Select CNN output for ISBI validation data');
  load(fullfile(path, fn));  % creates object "Yhat"
else
  fn = '../../Models/lenet-py/ISBI_Train20/YhatDeploy.mat';
  load(fn);
end

% Map 4d Numpy tensor to 3d Matlab tensor 
classLabel = 1;
Yhat = squeeze(Yhat(classLabel+1,:,:,:));
Yhat = permute(Yhat, [2 3 1]);  % python -> matlab ordering
Yhat(Yhat<0) = NaN;


%-------------------------------------------------------------------------------
%% Do analysis

Yhat2 = inpaint_prob_map(Yhat);
[fpr, recall, precision] = perfcurve2(Yvalid, Yhat2);
    
save([fn '.inpainted.mat'], '-v7.3');
save('./Yvalid.mat', 'Yvalid', '-v7.3');

%----------------------------------------
% visualize pixel-level classification performance
%----------------------------------------
figure;
plot(fpr, recall); grid on;
xlabel('FPR'); ylabel('TPR');
title('pixel-level classification performance');


%----------------------------------------
% visualize some slices
%----------------------------------------
z = [1 5 10];

figure('Position', [200 200 1200 400])
ha = tight_subplot(1, 3, [.03, .03]);
for ii = 1:length(z)
  axes(ha(ii));
  imagesc(Yvalid(:,:,z(ii)));
  title(sprintf('Yvalid; slice %d', z(ii)));
  set(gca, 'Xtick', [], 'Ytick', []);
end
saveas(gcf, [fn '.truth.eps'], 'epsc');


figure('Position', [200 200 1200 400])
ha = tight_subplot(1, 3, [.03, .03]);
for ii = 1:length(z)
  axes(ha(ii));
  imagesc(Yhat(:,:,z(ii)));
  title(sprintf('CNN estimate; slice %d', z(ii)));
  set(gca, 'Xtick', [], 'Ytick', []);
end
saveas(gcf, [fn '.cnn.eps'], 'epsc');


figure('Position', [200 200 1200 400])
ha = tight_subplot(1, 3, [.03, .03]);
for ii = 1:length(z)
  axes(ha(ii));
  imagesc(Yhat2(:,:,z(ii)));
  title(sprintf('inpainted; slice %d', z(ii)));
  set(gca, 'Xtick', [], 'Ytick', []);
end
saveas(gcf, [fn '.inpainted.eps'], 'epsc');



%-------------------------------------------------------------------------------
%% Run the analysis


return  % disabled for now

pctOmitted = [];
err = [];
runtime = [];


for pct = param.pctToTry;
  fprintf('[info]: for deletion percent: %0.2f\n', pct)

  % do this per-slice for now 
  % (perhaps better to do in 3d, but then many "collisions")
  p = sobolset(2);
  nToKill = pct*size(Yhat,1)*size(Yhat,2);
  P = p(1:nToKill,:);
  P = floor(bsxfun(@times, P, [size(Yhat,1), size(Yhat,2)])) + 1;
  ind = sub2ind(size(Yhat(:,:,1)), P(:,1), P(:,2));
  
  % mask out pixels 
  Ycnn = zeros(size(Yhat));
  Yrepaired = zeros(size(Yhat));
 
  % inpaint
  % inpainting in 3d seems slower...TODO: test this - I may have
  % not waited long enough...
  tic
  for ii = 1:size(Yhat,3)
      Yi = Yhat(:,:,ii);
      Yi(ind) = NaN; % simulate not evaluating these pixels w/ CNN
     
      % inpaint and rescale back to [0 1]
      Yr = inpaintn(Yi);
      Yr = Yr - min(Yr(:));
      Yr = Yr / (max(Yr(:)) - min(Yr(:)));

      if 1
          Yr(isfinite(Yi)) = Yi(isfinite(Yi));
      end
      
      Ycnn(:,:,ii) = Yi; 
      Yrepaired(:,:,ii) = Yr;
      
      fprintf('done slice %d; total time: %0.2f sec\n', ii, toc);
  end

  %----------------------------------------
  % collect some metrics
  %----------------------------------------
  runtime = [runtime toc];

  tmp = (Yrepaired - Yhat).^2;  
  err = [err sum(tmp(:))];

  nKilled = sum(isnan(Ycnn(:)));
  pctOmitted = [pctOmitted nKilled / numel(Ycnn)];

  %----------------------------------------
  % visualize
  %----------------------------------------
  figure('Position', [200 200 800 800]);
  ha = tight_subplot(2, 2, [.03, .03]);

  axes(ha(1));
  imagesc(Yvalid(:,:,param.z));
  title(sprintf('slice: %d, truth', param.z));
  set(gca, 'Xtick', [], 'Ytick', []);
  
  axes(ha(2));
  imagesc(Yhat(:,:,param.z));
  title(sprintf('full CNN estimate', param.z));
  set(gca, 'Xtick', [], 'Ytick', []);
  
  axes(ha(3));
  imagesc(Ycnn(:,:,param.z));
  title(sprintf('without %0.2f%%', 100*nKilled/numel(Yhat)));
  set(gca, 'Xtick', [], 'Ytick', []);
  
  axes(ha(4));
  imagesc(Yrepaired(:,:,param.z));
  title(sprintf('inpainted; error: %0.2f', err(end)));
  set(gca, 'Xtick', [], 'Ytick', []);
  linkaxes([ha(1) ha(2) ha(3)], 'xy');
  
 
  % also classification performance
  [fpr, recall, precision] = perfcurve2(Yvalid, Yrepaired);
  figure(1);
  hold on; plot(fpr, recall); hold off;
  
  drawnow;
end


%-------------------------------------------------------------------------------
%% summary graphs

figure(1);
lStr = cellfun(@num2str, num2cell([0 pctOmitted]), 'UniformOutput', 0);
legend(lStr, 'Location', 'SouthEast');


figure;
plot(pctOmitted, err, '-o'); grid on;
xlabel('percentage corrupted');  ylabel('error');
title('reconstruction performance');

figure;
plot(pctOmitted, runtime, '-o'); grid on;
xlabel('percentage corrupted');  ylabel('reconstruction time (sec)');
title('runtime');


