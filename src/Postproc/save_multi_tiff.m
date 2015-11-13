function save_multi_tiff(Y, outFileName)
% SAVE_MULTI_TIFF  Saves a 3d tensor as a multi-page tiff.

assert(isnumeric(Y));
assert(length(size(Y)) == 3);

imwrite(Y(:,:,1), outFileName);
for ii = 2:size(Y,3) 
    imwrite(Y(:,:,ii), outFileName, 'WriteMode', 'append');
end

