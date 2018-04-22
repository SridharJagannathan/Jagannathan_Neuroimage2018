%% Start afresh by clearing all command windows and variables
%clc; 
clear;

rng(1); % For reproducibility

%% Load the paths..
loadpath_validation;

%% Load the input files..

subject_ids = {'5','7','9','13','15', ... %'8',
               '17','19','20','21','23','24', ...
               '26','27','28','29','30','36', ...
               '41','42','44','45','48','49', ...
               '51','53','54','55','56','58', ...
               '60'};
           
                   
RT_Hori =[];RT_Algo =[];RT_Thetaalpha =[];
           
for m = 1:length(subject_ids)

subject = subject_ids{m};

%1. Preprocessed file -- > This contains the EEGlab preprocessed file

S.eeg_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/valdas_maskingdataset/preprocess'];
S.eeg_filename = ['AuMa_' subject '_pretrial_preprocess'];

% 2. Horiscale data  --> Common for all subjects
S.hori_filepath = [pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/valdas_maskingdataset/behaviour/'];
S.hori_filename = 'merged_behAuditoryMasking.mat';

subjname = ['subj_' subject];

fprintf('\n--Processing :%s--\n',subjname);

%% load the preprocessed EEGdata set..
evalexp = 'pop_loadset(''filename'', [S.eeg_filename ''.set''], ''filepath'', S.eeg_filepath);';

%load the preprocessed EEGdata set..
[T,EEG] = evalc(evalexp);

%% Compute alpha-theta ratio now..
[misc] = process_alphathetaratio(EEG);
t_meanalphatheta = mean(misc.theta ./ misc.alpha,2);
t_meanalpha = mean(misc.alpha,2);
t_meantheta = mean(misc.theta,2);

[~,trlIdx] = sort(t_meanalphatheta,'descend');
theta_alphadatascore = nan(1,length(trlIdx));

tmp=fix(numel(trlIdx)/3);
theta_alphadatascore(trlIdx(1:tmp)) = 3;
theta_alphadatascore(trlIdx(tmp+1:2*tmp)) = 2;
theta_alphadatascore(trlIdx(2*tmp+1:end)) = 1;

%% Use only some channels ..
           
electrodes_rx = {'E75','E70','E83',... %'E75','E70','E83',
                 'E36', 'E104', ... %'E35', 'E110',
                 'E90',...
                 'E45','E108','E102','E115','E100',...%'E40','E109','E101','E115','E100',
                 'E33', 'E122', 'E11'}; %'E27', 'E123', 'E11'            

chanlabels={EEG.chanlocs.labels};
selec_elec = ismember(chanlabels,electrodes_rx);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_rx] = evalc(evalexp);

EEG = EEG_rx;

%% load the hori data now..
hori_data = load([S.hori_filepath S.hori_filename]);

hori_datascore = hori_data.behdataset.Hori(hori_data.behdataset.subj_id == str2num(subject));
rt_datascore = hori_data.behdataset.RT_ms(hori_data.behdataset.subj_id == str2num(subject));
trial_num = hori_data.behdataset.trl_num(hori_data.behdataset.subj_id == str2num(subject));

allevents = {EEG.event.type}; trl_code =[];
for e = 1:length(allevents)
    trl_code(e) = EEG_rx.event(e).codes{1,2};       
end

hori_score = nan(1,length(trl_code));
rt_score = nan(1,length(trl_code));

for idx = 1:length(trl_code)
    matchidx = find(trial_num==trl_code(idx));
    if ~isempty(matchidx)
       hori_score(idx) = hori_datascore(matchidx);
       rt_score(idx) = rt_datascore(matchidx);
    end
end
hori_datascore = hori_score;
rt_datascore = rt_score;

%% Compute the features now..

%Collect channel labels..
chanlabels={EEG.chanlocs.labels};
electrodes_occ = {'E75','E70','E83'};%{'E75','E70','E83'};
electrodes_tempero = {'E108','E102','E115','E100'};%'E109','E101','E115','E100',
electrodes_frontal = {'E33', 'E122', 'E11'};%'E27', 'E123', 'E11'
electrodes_central = {'E36', 'E104'};%'E35', 'E110',
electrodes_parietal = {'E90'};

selec_elec = ismember(chanlabels,electrodes_occ);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_occ] = evalc(evalexp);

selec_elec = ismember(chanlabels,electrodes_frontal);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_front] = evalc(evalexp);

selec_elec = ismember(chanlabels,electrodes_tempero);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_tempero] = evalc(evalexp);

selec_elec = ismember(chanlabels,electrodes_central);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_centro] = evalc(evalexp);

selec_elec = ismember(chanlabels,electrodes_parietal);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_parieto] = evalc(evalexp);

%% Computing Alert features..

fprintf('\n--Computing Variance features--\n');

[trials_alert, misc_alert]= classify_computeVariancefts(EEG_occ);


    eleclabels.frontal = {'E33', 'E122', 'E11'};%'E27', 'E123', 'E11'
    eleclabels.central = {'E36', 'E104'};%'E35', 'E110',
   % eleclabels.parietal = {'E62'};
    eleclabels.temporal =  {'E45','E108'};%'E40','E109',
    eleclabels.occipetal = {'E75','E70','E83'};%{'E75','E70','E83'};

fprintf('\n--Computing Coherence features--\n');
[coh]= classify_computeCoherencefts(EEG,eleclabels);
coh_features = table2array(coh);
    
    
%% Computing Vertex features..
fprintf('\n--Computing Vertex: monophasic features--\n');
monophasic_fts =[]; Data = double((EEG_parieto.data));
for z = 1:EEG_parieto.trials
    for s = 1:size(Data,1)
        
        [Vertex, Vertex_ft] = classify_computeVertexMonophasicfts(Data(s,:,z), EEG_parieto.srate);
        if (Vertex_ft.count>0 && Vertex_ft.negpks_1 < -15 && Vertex_ft.negpks_2 < -15 &&...
                                Vertex_ft.duration >0.1 && Vertex_ft.pospks> 30)
            monophasic_fts(s,z) = 1;
        else
            monophasic_fts(s,z) = 0;

        end
    end
end

fprintf('\n--Computing Vertex: biphasic features--\n');
biphasic_fts =[]; Data = double((EEG_parieto.data));
for z = 1:EEG_parieto.trials
    for s = 1:size(Data,1)
        
        [Vertex, Vertex_ft] = classify_computeVertexBiphasicfts(Data(s,:,z), EEG_parieto.srate);
       if (Vertex_ft.count>0 && Vertex_ft.negpks < -40 && Vertex_ft.pospks> 40) %~isempty(Kcomp.start_stop)
            biphasic_fts(s,z) = 1;
        else
            biphasic_fts(s,z) = 0;

        end
    end
end

monophasic_fts = sum(monophasic_fts,1);
monophasic_def = find(monophasic_fts); 

biphasic_fts = sum(biphasic_fts,1);
biphasic_def = find(biphasic_fts); 


%% Computing spindle features..
fprintf('\n--Computing Spindle: features--\n');

spin_ft =[]; Data = double(squeeze(EEG_tempero.data));
Freq_range = 12:16;Time_params = [0.4 1.5];

for z = 1:EEG_tempero.trials
    
   for s = 1:size(Data,1)
        
    [Spindle, detected_spindles] = classify_computeSpindlefts(Data(s,:,z), EEG_tempero.srate,Freq_range,Time_params);
        if  ~isempty(Spindle.start_stop)%~isempty(spindles_start_end)
            pospeakmean=[];negpeakmean=[];
            for idx = 1:length(detected_spindles)
                pospeakmean(idx) =  detected_spindles{1,idx}.meanpospks;
                
                negpeakmean(idx) =  detected_spindles{1,idx}.meannegpks;
            end
            
             
             if(sum(pospeakmean>9) || sum(negpeakmean>9))
                
               spin_ft(s,z) = 1;
            
             else
                 
               spin_ft(s,z) = nan;
                 
             end 
            
            
        else
            spin_ft(s,z) = 0;

        end
    end
        
end

spinsum_ft = sum(spin_ft,1);
spinnansum_ft = nansum(spin_ft,1);
spinnan_ft = sum(isnan(spin_ft), 1);
spinrecon_cand = intersect(find(spinnansum_ft>=spinnan_ft),find(spinnan_ft>=1));

spinsum_ft = sum(spin_ft,1);
spindle_def = find(spinsum_ft>0);
spindle_def = union(spindle_def,spinrecon_cand);

%% Computing k-complex features..

fprintf('\n--Computing K-complex: features--\n');

kcomp_ft =[]; Data = double(squeeze(EEG.data));

for z = 1:EEG.trials
    
    [Kcomp, Kcomp_ft] = classify_computeKcomplexfts(Data(s,:,z), EEG.srate);
    
        if (Kcomp_ft.count>0 && Kcomp_ft.negpks < -45 && Kcomp_ft.pospks-Kcomp_ft.negpks > 100 &&...
                Kcomp_ft.pospks > 0.5*abs(Kcomp_ft.negpks)) %~isempty(Kcomp.start_stop)
            kcomp_ft(s,z) = 1;
        else
            kcomp_ft(s,z) = 0;

        end
         
end

kcomp_def = find(sum(kcomp_ft,1));

kcomp_features = zeros(1, EEG.trials);
kcomp_features(kcomp_def)=1;


%% graphical elements
grapho_def = union(biphasic_def,monophasic_def);
grapho_def = union(grapho_def,kcomp_def);
nonspind_def = grapho_def;

spind_comb = zeros(EEG.trials,1);
spind_comb(spindle_def) = 1;

nonspind_comb = zeros(EEG.trials,1);
nonspind_comb(nonspind_def) = 1;

%% summarize the features..
testData = [misc_alert.varian.freqband2' misc_alert.varian.freqband5' misc_alert.varian.freqband6'...
                    misc_alert.varian.freqband10'...
                    coh_features];
                

S.model_filepath = [ pathappend 'SpatialAttention_Drowsiness/microMeasuresAlertness_Neuroimage2018/Scripts/models/'];

rmsubject_ids = {''};

for k = 1:1 %length(rmsubject_ids)                    

S.model_filename = ['model_collec64_' char(rmsubject_ids(k)) ];
clear model_collecAlert minimums_collAlert ranges_collAlert
clear model_collecGrapho minimums_collGrapho ranges_collGrapho
load([S.model_filepath S.model_filename]);
fprintf('\n--Using model:%s--\n',string(rmsubject_ids(k)));

 for idx = 1: 1
  bestModelAlert = model_collecAlert(idx);
   bestModelGrapho = model_collecGrapho(idx);
        
 %To scale the training and testing data in the same way..
 minimumsAlert = minimums_collAlert(idx,:);
 rangesAlert = ranges_collAlert(idx,:);
 
 minimumsGrapho = minimums_collGrapho(idx,:);
 rangesGrapho = ranges_collGrapho(idx,:);
 
 fprintf('\n--using %s--\n',string(S.model_filename));  
    testsubj = subject;
    fprintf('\n--Testing on subject:%s--\n',string(testsubj));
    %testtrls = find(subj_id == str2num(char(testsubj)));
    
   
    %scale the testing data now..
    OrigtestData_class = testData;
    testData_class = (testData - repmat(minimumsAlert, size(testData, 1), 1)) ./ repmat(rangesAlert, size(testData, 1), 1);
    bad_trls = find(isnan(hori_datascore));
    
       gold_test = hori_datascore;
       
       gold_test(bad_trls) =[];
       
       rt_test = rt_datascore;
       rt_test(bad_trls) =[];
       
       thetaalpha_test = theta_alphadatascore;
       thetaalpha_test(bad_trls) =[];
       
      
       gold_labels = cell(1,length(gold_test));
       
       for tempidx = 1: length(gold_test)
          if gold_test(tempidx)>=1 && gold_test(tempidx)<=2
             gold_labels{tempidx} = 'Alert'; 
          elseif gold_test(tempidx)>=3 && gold_test(tempidx)<=5
              gold_labels{tempidx} = 'Ripples';    
          elseif gold_test(tempidx)>=6 && gold_test(tempidx)<=10
              gold_labels{tempidx} = 'Grapho';
          end
           
       end
       
       
       gold_usable = nan(length(gold_labels),1);
       gold_usable(find(strcmp(gold_labels,'Alert'))) = 1;
       gold_usable(find(strcmp(gold_labels,'Ripples'))) = 2;
       
       gold_usable(find(strcmp(gold_labels,'Grapho'))) = 3;
       %gold_usable(find(strcmp(gold_test,'Spindle'))) = 3;

      
        %predict the labels now..
    if (~isempty(find(gold_usable==1)>1) || ~isempty(find(gold_usable==2)>1) || ...
              ~isempty(find(gold_usable==3))>1 || ~isempty(find(gold_usable==4)>1))
        
        
       testLabel = gold_usable;
        
       
       if ~isempty(bad_trls)
           testData_class(bad_trls,:) =[]; 
           OrigtestData_class(bad_trls,:) =[];
           spind_comb(bad_trls) =[];
           nonspind_comb(bad_trls) =[];
       end
        
        
        [predict_label, accuracy, prob_values] = svmpredict(testLabel, testData_class, bestModelAlert, '-b 1');
       
       spind_poss = intersect(find(predict_label == 2),find(spind_comb==1));
       nonspind_poss = intersect(find(predict_label == 2),find(nonspind_comb==1));
       
       subtestdatagrapho = OrigtestData_class(spind_poss,:);
       subtestLabelgrapho = testLabel(spind_poss);
       
        subtestdatagrapho = (subtestdatagrapho - repmat(minimumsGrapho, size(subtestdatagrapho, 1), 1)) ./ repmat(rangesGrapho, size(subtestdatagrapho, 1), 1);

  
        [subpredictedLabelgrapho, accuracy, decisValueWinner] = svmpredict(subtestLabelgrapho, subtestdatagrapho, bestModelGrapho, '-b 1');
       
        predactualgrapho = find(subpredictedLabelgrapho ==3);
        predict_label(spind_poss(predactualgrapho)) = 3;
        predict_label(nonspind_poss) = 3;

      
       [confusionMatrixAll,orderAll] = confusionmat(testLabel,predict_label);
        % Calculate the overall accuracy from the overall predicted class label
        accuracyAll = trace(confusionMatrixAll)/sum(confusionMatrixAll(:));
        
        %ModelACC_1 = [ModelACC_1; 100*accuracyAll];
        
        fprintf('\n--Validating :%s--\n',string(testsubj));

       fprintf('-- Accuracy rate %0.2f%% --- \n', 100*accuracyAll);
       
        if length(confusionMatrixAll) == 1
            if unique(testLabel) == 1
                names= {'Alert'}; 
            else
                names= {'Drowsy'}; 
            end
        elseif length(confusionMatrixAll) == 2
               if orderAll(1) == 1
                   names= {'Alert','Ripples'};
               else
                   names= {'Ripples','Grapho'};
               end
        elseif length(confusionMatrixAll) == 3
            names= {'Alert','Ripples','Grapho'};
        elseif length(confusionMatrixAll) == 4
            names= {'Alert','Ripples','Vertex','Spindle'};
         end
       disptable = array2table( confusionMatrixAll, 'VariableNames', names, 'RowNames', names );
       disp(disptable);
       
       rts_alert_hori = rt_test(find(strcmp(gold_labels,'Alert')));
       rts_drowsymid_hori = rt_test(find(strcmp(gold_labels,'Ripples')));
       rts_drowsygrapho_hori= rt_test(find(strcmp(gold_labels,'Grapho')));
       
       rts_drowsy_hori = union(rts_drowsymid_hori,rts_drowsygrapho_hori);
       
       
       rts_alert_algo = rt_test(find(predict_label == 1));
       rts_drowsymid_algo = rt_test(find(predict_label == 2));
       rts_drowsygrapho_algo = rt_test(find(predict_label == 3));
       
       rts_drowsy_algo = union(rts_drowsymid_algo,rts_drowsygrapho_algo);
       
       rts_alert_thetaalpha = rt_test(find(thetaalpha_test == 1));
       rts_drowsymid_thetaalpha = rt_test(find(thetaalpha_test == 2));
       rts_drowsygrapho_thetaalpha = rt_test(find(thetaalpha_test == 3));
       
       rts_drowsy_thetaalpha = union(rts_drowsymid_thetaalpha,rts_drowsygrapho_thetaalpha);
       
              
       nminTrls = 10; %10;
       
         
       RT_Hori = [RT_Hori; [str2num(testsubj).*ones(length(rts_alert_hori),1) rts_alert_hori' 1.*ones(length(rts_alert_hori),1)]];
       RT_Hori = [RT_Hori; [str2num(testsubj).*ones(length(rts_drowsymid_hori),1) rts_drowsymid_hori' 2.*ones(length(rts_drowsymid_hori),1)]];
       RT_Hori = [RT_Hori; [str2num(testsubj).*ones(length(rts_drowsygrapho_hori),1) rts_drowsygrapho_hori' 3.*ones(length(rts_drowsygrapho_hori),1)]];
      
       
       RT_Algo = [RT_Algo; [str2num(testsubj).*ones(length(rts_alert_algo),1) rts_alert_algo' 1.*ones(length(rts_alert_algo),1)]];
       RT_Algo = [RT_Algo; [str2num(testsubj).*ones(length(rts_drowsymid_algo),1) rts_drowsymid_algo' 2.*ones(length(rts_drowsymid_algo),1)]];
       RT_Algo = [RT_Algo; [str2num(testsubj).*ones(length(rts_drowsygrapho_algo),1) rts_drowsygrapho_algo' 3.*ones(length(rts_drowsygrapho_algo),1)]];
       
       RT_Thetaalpha = [RT_Thetaalpha; [str2num(testsubj).*ones(length(rts_alert_thetaalpha),1) rts_alert_thetaalpha' 1.*ones(length(rts_alert_thetaalpha),1)]];
       RT_Thetaalpha = [RT_Thetaalpha; [str2num(testsubj).*ones(length(rts_drowsymid_thetaalpha),1) rts_drowsymid_thetaalpha' 2.*ones(length(rts_drowsymid_thetaalpha),1)]];
       RT_Thetaalpha = [RT_Thetaalpha; [str2num(testsubj).*ones(length(rts_drowsygrapho_thetaalpha),1) rts_drowsygrapho_thetaalpha' 3.*ones(length(rts_drowsygrapho_thetaalpha),1)]];
       

    else

       [predict_label, accuracy, prob_values] = svmpredict(zeros(length(testData_class),1), testData_class, bestModel, '-b 1');
       fprintf('\n--Skipping Accuracy on subject:%s--\n',string(testsubj));
    

    end
       
   
 end

end


end

%%
 RT_Horisum = sum(RT_Hori,2);
 RT_Algosum = sum(RT_Algo,2);
 RT_Thetaalphasum = sum(RT_Thetaalpha,2);

 RT_Horiall = RT_Hori(find(~isnan(RT_Horisum)),:);

 RT_Algoall = RT_Algo(find(~isnan(RT_Algosum)),:);

 RT_Thetaalphaall = RT_Thetaalpha(find(~isnan(RT_Thetaalphasum)),:);

 
S.rt_filepath = [pathappend 'SpatialAttention_Drowsiness/microMeasuresAlertness_Neuroimage2018/Scripts/validation/'];
S.rt_filename = 'RT_Hori_RT_1.csv';

filename = [S.rt_filepath S.rt_filename]; 
fileID = fopen(filename,'wt');

A = {'subj_id','RT', 'State'};

[rows, columns] = size(A);
for index = 1:rows    
    fprintf(fileID, '%s,', A{index,1:end-1});
    fprintf(fileID, '%s\n', A{index,end});
end 

fclose(fileID);

csv_writedata = [RT_Horiall]; % Alert, Drowsy(mild), Drowsy(severe)

fileID = fopen(filename,'a+');
[rows, columns] = size(csv_writedata);
%columns = columns +1;
for index = 1:rows 
    fprintf(fileID, '%f,', csv_writedata(index,1:end));
    fprintf(fileID, '\n');
 end 

fclose(fileID);


S.rt_filename = 'RT_Algo_RT_1.csv';

filename = [S.rt_filepath S.rt_filename]; 
fileID = fopen(filename,'wt');

A = {'subj_id','RT', 'State'};

[rows, columns] = size(A);
for index = 1:rows    
    fprintf(fileID, '%s,', A{index,1:end-1});
    fprintf(fileID, '%s\n', A{index,end});
end 

fclose(fileID);

csv_writedata = [RT_Algoall]; % Alert, Drowsy(mild), Drowsy(severe)

fileID = fopen(filename,'a+');
[rows, columns] = size(csv_writedata);
%columns = columns +1;
for index = 1:rows 
    fprintf(fileID, '%f,', csv_writedata(index,1:end));
    fprintf(fileID, '\n');
 end 

fclose(fileID);

S.rt_filename = 'RT_AlphaTheta_RT_1.csv';

filename = [S.rt_filepath S.rt_filename]; 
fileID = fopen(filename,'wt');

A = {'subj_id','RT', 'State'};

[rows, columns] = size(A);
for index = 1:rows    
    fprintf(fileID, '%s,', A{index,1:end-1});
    fprintf(fileID, '%s\n', A{index,end});
end 

fclose(fileID);

csv_writedata = [RT_Thetaalphaall]; % Alert, Drowsy(mild), Drowsy(severe)

fileID = fopen(filename,'a+');
[rows, columns] = size(csv_writedata);
%columns = columns +1;
for index = 1:rows 
    fprintf(fileID, '%f,', csv_writedata(index,1:end));
    fprintf(fileID, '\n');
 end 

fclose(fileID);


temp =[];
