%% Set the path for the external toolbox folder and add it to MATLAB search directory

%% Determine location
global pathappend
if ismac
 pathappend = '/Users/Shri/Documents/PhD_Cambridge/Projects/';
elseif isunix
 pathappend = '/work/imagingQ/';
end

%% Add paths now..
%SPM path
spm_toolbox = [pathappend 'SpatialAttention_Drowsiness/Scripts/toolboxes/spm12'];
addpath(spm_toolbox);


%Fieldtrip path
ftp_toolbox = [pathappend 'SpatialAttention_Drowsiness/Scripts/toolboxes/fieldtrip-20151223'];
addpath(ftp_toolbox);
%ft_defaults;

%EEGlab path
eeglab_toolbox = [pathappend 'SpatialAttention_Drowsiness/Scripts/toolboxes/eeglab13_5_4b'];
addpath(genpath(eeglab_toolbox));

%fMRIB plugin path
fMRIB_toolbox = [pathappend 'SpatialAttention_Drowsiness/Scripts/toolboxes/fmrib1.21'];
addpath(genpath(fMRIB_toolbox));

%micromeasures toolbox path
micromeasures_toolbox = [pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/microMeasAlertness_HumanEEG'];
addpath(genpath(micromeasures_toolbox));

%plot tools path
plot_toolbox = [pathappend 'SpatialAttention_Drowsiness/microMeasuresAlertness_Neuroimage2018/Scripts/plot_tools'];
addpath(genpath(plot_toolbox));