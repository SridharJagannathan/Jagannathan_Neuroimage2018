% process_alphathetaratio() - Performs alphatheta ratio
% Usage:
%       >>  [misc] = process_alphathetaratio(EEG); 
%
% Inputs:            
%       EEGlab structure
%
% Output:
%       trials struct contains alpha,theta power etc..
%
% Written by Shri (Sridhar Jagannathan)
function [misc] = process_alphathetaratio(eegStruct)

misc = [];
EEG = eegStruct;
nelec = EEG.nbchan;
    
ntrials = EEG.trials;
goodtrial = 1:ntrials;

alphafreq = [10 12]; %Frequency range in Hz 9-11
thetafreq = [4 6]; %Frequency range in Hz 4-5
%deltafreq = [0.5 4]; %Frequency range in Hz 12-14
%betafreq = [9 11]; %Frequency range in Hz 12-14


% 1. Initialize variables now..
[alpha,theta] = deal(nan(ntrials,nelec));


% 2. Compute band power and variance explained..

for k = 1:nelec
    
                
 % The function to have a spectrum for one electrode
 [ersp,itc,powbase,times,freqs,erspboot,itcboot,tfdata] = ...
             newtimef(EEG.data(k,:,:), EEG.pnts,[EEG.xmin EEG.xmax]*1000, EEG.srate, 0, ...
              'padratio', 2, 'freqs', [0.5 40], ...
              'plotersp', 'off','plotitc','off','verbose','off');  
          
  Pow  = tfdata.*conj(tfdata); % power
          
  [~, AlpfBeg] = min(abs(freqs-alphafreq(1)));
  [~, AlpfEnd] = min(abs(freqs-alphafreq(2)));
  
  [~, ThetfBeg] = min(abs(freqs-thetafreq(1)));
  [~, ThetfEnd] = min(abs(freqs-thetafreq(2)));
  
%   [~, DeltafBeg] = min(abs(freqs-deltafreq(1)));
%   [~, DeltafEnd] = min(abs(freqs-deltafreq(2)));
%   
%   [~, BetafBeg] = min(abs(freqs-betafreq(1)));
%   [~, BetafEnd] = min(abs(freqs-betafreq(2)));
                              
 % compute power in a frequency band..
   power_alphaFB = squeeze(sum(Pow(AlpfBeg:AlpfEnd,:,:),1));
   
   power_thetaFB = squeeze(sum(Pow(ThetfBeg:ThetfEnd,:,:),1));
   
%    power_deltaFB = squeeze(sum(Pow(DeltafBeg:DeltafEnd,:,:),1));
%    
%    power_betaFB = squeeze(sum(Pow(BetafBeg:BetafEnd,:,:),1));
%    
%    power_allFB = squeeze(sum(Pow(1:41,:,:),1));
   

 % mean across time points
   alpha(goodtrial,k) = mean(power_alphaFB,1);
   theta(goodtrial,k) = mean(power_thetaFB,1);
   
 
   
          
end



tmp_xalphatheta = corr(mean(theta,2),mean(alpha,2),'type','spearman','rows','complete');

fprintf('\n\n -- Details for subject %s -- ', EEG.subject);
if tmp_xalphatheta > 0
            
     fprintf('\n-- Possibly a noisy Subject --\n');
     fprintf('-- Alpha and Theta correlate positively --\n');
    
end 


misc.alpha = alpha;
misc.theta = theta;


end