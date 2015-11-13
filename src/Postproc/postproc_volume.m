function Yhat = postproc_volume(YhatRaw, varargin)
% POSTPROC_VOLUME  Postprocess a volume of CNN predictions
%
%      Yhat = postproc_volume(Yin, 'tiffout', 'Yout');
%
%   where,
%      Yin  :  a tensor of probability estimates with dimensions 
%                    (nClasses, nSlices, width, height)
%              -or-
%                    (width, height, nSlices)
%
%              In the former case, this code will reshape the volume
%              to the latter case (i.e. four -> three dimensions).
%              The four dimensional case is an artifact of how we 
%              save estimates computed using Caffe.
%
%     Yhat : The postprocessed result; a tensor with dimensions 
%            (width, height, nSlices)
%
%  Currently, postprocessing consists of the following steps:
%   1.  Inpainting any "missing" (non-evaluated) pixels
%   2.  Smoothing the results via a filtering operation
%   3.  Re-calibrating probability estimates (TBD)
%
%  Note: make sure to add './inpaint' to matlab's search path
%  before calling this function!  (e.g. by calling updatepath.m)

% mjp 2015

p = inputParser;
p.addRequired('YhatRaw');
p.addOptional('tiffout', '');
p.addOptional('invert', 1);
p.addOptional('fdim', 7);          % filter size
p.addOptional('ford', 7*7-3);      % order parameter for filter
p.parse(YhatRaw, varargin{:});


% change from 4D caffe tensor to 3D EM tensor (if needed)
if length(size(YhatRaw)) == 4
    YhatRaw = YhatRaw(2,:,:,:);   % probabilities for class 1 (membrane)
    YhatRaw = permute(YhatRaw, [3 4 2 1]);  % implicit squeeze
end

% At this point, YhatRaw should have shape: (width, height, slices)
assert(length(size(YhatRaw)) == 3);


% Fill in any points not explicitly evaluated.
% An assumption here is that any negative values should be inpainted.
YhatRaw(YhatRaw < 0) = NaN;
if any(isnan(YhatRaw(:)))
    YhatRaw = inpaint_prob_map(YhatRaw);
end


% smoothing
% Ciresan recommended a 2 pixel median filter.
% For cases where we inpaint though, it may be preferred to
% use something closer to a maximum filter.
if p.Results.fdim > 0,  
    n = p.Results.fdim;   ord = p.Results.ford;
    
    filt = true(n);
    Yhat = zeros(size(YhatRaw));
    for ii = 1:size(YhatRaw,3)
        Yhat(:,:,ii) = ordfilt2(YhatRaw(:,:,ii), ord, filt);  
    end
    keyboard % TEMP
end


% invert so that low values correspond to high probability of
% membrane (ISBI2012 convention).
if p.Results.invert
    Yhat = 1 - Yhat;
end


% save as multi-page tiff (optional)
if length(p.Results.tiffout) 
    outFileName = [p.Results.tiffout '.tif'];
    fprintf('[%s]: Writing results to "%s"\n', mfilename, outFileName);
    save_multi_tiff(Yhat, outFileName);
end
