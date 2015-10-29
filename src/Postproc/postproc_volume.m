function Yhat = postproc_isbi(YhatRaw, outFileName)
% POSTPROC_VOLUME  Postprocess a volume of CNN predictions
%
%  Saves result in a multi-page .tiff file suitable for submission to ISBI 2012.

outFileName = [outFileName '.tif'];

% change from 4D caffe tensor to 3D EM tensor
if length(size(YhatRaw)) == 4
    YhatRaw = YhatRaw(2,:,:,:);   % probabilities for class 1 (membrane)
    YhatRaw = permute(YhatRaw, [3 4 2 1]);  % implicit squeeze
    assert(length(size(YhatRaw)) == 3);
end


% Fill in any points not explicitly evaluated
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








