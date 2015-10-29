% Adds required subdirectories to matlab's search path.

thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'inpaint'));
addpath(fullfile(thisDir, 'tight_subplot'));