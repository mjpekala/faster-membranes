function Yhat = postproc_volume(YhatRaw, outFileName)
% POSTPROC_VOLUME  Postprocess a volume of CNN predictions
%
%      Yhat = postproc_volume(Yin, 'outputFile')
%
%   where,
%      Yin  :  a tensor of probability estimates with dimensions 
%                    (nClasses, nSlices, width, height)
%              -or-
%                    (width, height, nSlices)
%              In the former case, this code will reshape the volume
%              to the latter case (with three dimensions).
%              The four dimensional case is an artifact of how we 
%              save estimates computed using Caffe.
%
%     outputFile : The .tiff output file name (string).
%                  Will be a multi-page tiff in ISBI 2012 format.
%
%     Yhat : The inpainted result; a tensor with dimensions (width, height, nSlices)
%
%  Note: make sure to add './inpaint' to matlab's search path
%  before calling this function!  (e.g. by calling updatepath.m)

% mjp 2015

outFileName = [outFileName '.tif'];

% change from 4D caffe tensor to 3D EM tensor
if length(size(YhatRaw)) == 4
    YhatRaw = YhatRaw(2,:,:,:);   % probabilities for class 1 (membrane)
    YhatRaw = permute(YhatRaw, [3 4 2 1]);  % implicit squeeze
end

% assume a tensor of shape (width, height, slices)
assert(length(size(YhatRaw)) == 3);


% Fill in any points not explicitly evaluated.
% An assumption here is that any negative values should be inpainted.
YhatRaw(YhatRaw < 0) = NaN;
if any(isnan(YhatRaw(:)))
    YhatRaw = inpaint_prob_map(YhatRaw);
end


% smoothing
% TODO
filt = fspecial('gaussian', [5 5], 2);
Yhat = imfilter(YhatRaw, filt, 'same');


% invert so that low values correspond to high probability of
% membrane
Yhat = 1 - Yhat;


% save as multi-page tiff
imwrite(Yhat(:,:,1), outFileName);
for ii = 2:size(Yhat,3) 
    imwrite(Yhat(:,:,ii), outFileName, 'WriteMode', 'append');
end








