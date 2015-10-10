function X = read_multi_tiff(fileName)
% LOAD_TIFF Reads in a multi-page .tiff file

X = imread(fileName);

for ii = 2:1000
  try
    Xi = imread(fileName, ii);
    X = cat(3, X, Xi);
  catch ME
    % assume we just hit the end of file.
    break;
  end
end
