function [spk_id,  sig_thresh, noise_sigma] = triggerSpikes(Vsig,thresh_sign,target_Nspks,sig_thresh)
% [spk_id,  sig_thresh, noise_sigma] = triggerSpikes(Vsig,thresh_sign,target_Nspks,<sig_thresh>)
% detects spikes from a continuous voltage signal and returns times of spike peaks
% INPUTS: 
%   Vsig: continuous voltage signal to detect spikes with
%   thresh_sign: detect of peaks (+1) or valleys (-1)
%   target_Nspks: number of spikes to detect
%   <sig_thresh>: amplitude threshold for spike-detection (if using).
% OUTPUTS:
%   spk_id: Index values of detected spike peaks
%   sig_thresh: %amplitude threshold on spike peaks
%   noise_sigma: Estimated noise level

if nargin < 4
    sig_thresh = [];
end

if thresh_sign == -1 %if triggering off negative peaks, just invert the signal
    Vsig = -Vsig;
end
if size(Vsig,2) < size(Vsig,1) %make sure we're dealing with a row vector
    Vsig = Vsig';
end

%find index values of local maxima
sgn = diff(sign(diff(Vsig,1,2)),1,2); 
id = find(sgn(1,:) < 0)+1;

noise_sigma = median(abs(Vsig))/0.6745; %Quiroga 2004 Neural Comp

if isempty(sig_thresh) %if no amplitude threshold is provided
    n_extrema = length(id); 
    if strcmp(target_Nspks,'median') %if setting based on multiple of noise-level
        sig_thresh = 4*noise_sigma; %Quiroga 2004 Neural Comp
    else %if setting threshold to achieve desired number of spikes
        prc = target_Nspks .* 100./n_extrema; %target prctile of spikes 
        sig_thresh = prctile(Vsig(id),100-prc); %get amplitude threshold
    end
end
id = id(Vsig(id) > sig_thresh); %keep only spikes above amplitude threshold

%if the ISI is very  short, and the trigger channel does not go back to
%near zero (amp_th/3) between two trigger points, then throw away the event
%with the smaller triggger amplitude as a putative double-trigger event
min_ISI_check = 20; %ISI below which we check for double-triggers (in samples)
rel_amp_thresh = 1/3; %waveform needs to dip below this fraction of the amp-thresh to avoid being counted as a double-trig
sid = find(diff(id) < min_ISI_check); %putative double-trigger events
if length(sid) < length(Vsig)*0.01 %check if we have too many putative double-triggers 
    okid = [];
    for j = 1:length(sid)
        if min(Vsig(id(sid(j)):id(sid(j)+1))) < sig_thresh*rel_amp_thresh %if waveform drops below this value between trig events its OK
            okid = [okid sid(j)];
        end
    end
    sid = setdiff(sid,okid); %don't throw out okid
    v = cat(1,Vsig(id(sid)),Vsig(id(sid+1)));%get amplitude of the first and second component of putative double-trigs
    [~,smaller_peak_loc] = min(v); %find the one with the smaller amp
    xid = id(sid+smaller_peak_loc-1); %set to get rid of
    %     fprintf('Removing %d double Triggers (from %d/%d maxima)\n',length(xid),length(id),n_extrema);
    spk_id = setdiff(id,xid); %remove double-trigs
else %potentially some problem
    fprintf(sprintf('Too many double Triggers (%d/%d)\n',length(sid),length(Vsig)));
    spk_id = id; 
end

