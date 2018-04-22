
%% Load the paths..
loadpath;

%% Dataset 1 ... 
% Load the input files..

subject_ids = {'105','107','109','111','113','117', ...
               '118','122','123','125','127','129', ...%, missing data..,'132','151'
               '134','137','138','139','144','147', ...      
               '149','150'};
           
Trlnums_raw_dataset_1 =[];  
Trlnums_processed_dataset_1 =[]; 
for m = 1:length(subject_ids)
    
subject = subject_ids{m};

subjname = ['subj_' subject];

fprintf('\n--Processing :%s--\n',subjname);

%1. Raw file counts from Hori file..
S.hori_filepath = [pathappend 'SpatialAttention_Drowsiness/microMeasuresAlertness_Neuroimage2018/Horidata/Dataset_1/'];
S.hori_filename = 'horidata_ABC.mat';

hori_data = load([S.hori_filepath S.hori_filename]);

subjnames = hori_data.horidataset.Properties.VarNames(:);

testsubj = strcmp(subjnames,subjname);

Trlnums_raw_dataset_1(m) = length(hori_data.horidataset{1,find(testsubj)});

   
%1. Preprocessed file -- > This contains the EEGlab preprocessed file
S.eeg_filepath = [pathappend 'SpatialAttention_Drowsiness/microMeasuresAlertness_Neuroimage2018/Preprocessed/Dataset_1/'];
S.eeg_filename = [subject '_pretrial_preprocess'];

evalexp = 'pop_loadset(''filename'', [S.eeg_filename ''.set''], ''filepath'', S.eeg_filepath);';

%load the preprocessed EEGdata set..
[T,EEG] = evalc(evalexp);

Trlnums_processed_dataset_1(m) = EEG.trials;

end

%% Dataset 2 ... 
% Load the input files..

subject_ids = {'5','7','9','13','15', ... %'8',
               '17','19','20','21','23','24', ...
               '26','27','28','29','30','36', ...
               '41','42','44','45','48','49', ...
               '51','53','54','55','56','58', ...
               '60'};
           
Trlnums_raw_dataset_2 =[];  
Trlnums_processed_dataset_2 =[]; 
for m = 1:length(subject_ids)
    
subject = subject_ids{m};

subjname = ['subj_' subject];

fprintf('\n--Processing :%s--\n',subjname);

%1. Raw file counts from Hori file..
S.hori_filepath = [pathappend 'SpatialAttention_Drowsiness/microMeasuresAlertness_Neuroimage2018/Horidata/Dataset_2/'];
S.hori_filename = 'merged_behAuditoryMasking.mat';

hori_data = load([S.hori_filepath S.hori_filename]);

subjnames = hori_data.behdataset.subj_id;

testsubj = find(subjnames == str2num(subject));

Trlnums_raw_dataset_2(m) = length(testsubj);

   
%1. Preprocessed file -- > This contains the EEGlab preprocessed file
S.eeg_filepath = [pathappend 'SpatialAttention_Drowsiness/microMeasuresAlertness_Neuroimage2018/Preprocessed/Dataset_2/'];
S.eeg_filename = ['AuMa_' subject '_pretrial_preprocess'];

evalexp = 'pop_loadset(''filename'', [S.eeg_filename ''.set''], ''filepath'', S.eeg_filepath);';

%load the preprocessed EEGdata set..
[T,EEG] = evalc(evalexp);

Trlnums_processed_dataset_2(m) = EEG.trials;

end

temp=[];

