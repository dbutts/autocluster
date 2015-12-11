function [clusterDetails,spike_features,sum_fig] = detect_and_cluster_init(sfile,params,use_chs)
% [clusterDetails,spike_features,sum_fig] = detect_and_cluster_init(sfile,<params>,<use_chs>)
% takes a data file, detects spikes, and then performs initial clustering
% INPUTS:
%   sfile: data file. Either filename, or struct containing voltage, timestamp, and Fs 
%   <params>: struct of params
%   <use_chs>: channels to use for clustering (default is all channels provided)
% OUTPUTS:
%   clusterDetails: struct containing information about the resulting clustering
%   spike_features: (Nxp) array of features extracted from spikes, N is num spikes, p is num dims
%   sum_fig: handle for summary figure


%% DEFAULT PARAMETERS
if nargin < 3 || isempty(use_chs)
    use_chs = nan;
end

if nargin < 2 || isempty(params)
    params = struct(); %set to empty struct if none provided
end
if ~isfield(params,'filt_cutoff') 
    params.filt_cutoff = [100 nan]; %high-pass for filtering raw V for spike detection (if sfile is provided as file name)
end
if ~isfield(params,'add_Vmean') 
    params.add_Vmean = false; %whether or not to add in across-electrode average FullV (if sfile is provided as file name)
end
if ~isfield(params,'thresh_sign') 
    params.thresh_sign = -1; %whether to detect on peaks (+1) or valleys (-1)
end
if ~isfield(params,'target_rate')
    params.target_rate = 50; %target spike detection rate (Hz). Set to 'median' to use a fixed amp-threshold
end
if ~isfield(params,'spk_pts') 
    params.spk_pts = [-12:27]; %set of time points relative to trigger to use for classification
end
if ~isfield(params,'outlier_thresh')
    params.outlier_thresh = 7; %threshold on Mahal distance (sqrt) to count a point as an outlier
end
if ~isfield(params,'verbose')
    params.verbose = 2; %display text during fitting
end
if ~isfield(params,'use_best_only')
    params.use_best_only = false; %use only the best spike waveform (if using multi-channel) for calculating template features.
end
if ~isfield(params,'cluster_bias')
    params.cluster_bias = 0.85; %bias to control type 2 errors for classifying SUs
end
if ~isfield(params,'summary_plot') 
    params.summary_plot = 1;  %1 == make plot but keep it invisible, 2 == make visible plot
end
if ~isfield(params,'reg_lambda')
    params.reg_lambda = 0; %regularization on Cov Mats for EM (Matlab based, doesnt seem to help)
end
if ~isfield(params,'n_pcs')
    params.n_pcs = 4; %number of PCs
end
if ~isfield(params,'n_tdims')
    params.n_timds = 4; %number of voltage dimensions
end
if ~isfield(params,'try_features')
    %[PCs voltage energy templates]
    params.try_features = [1 2 4]; %which features to try clustering with.
end
if ~isfield(params,'min_Pcomp')
    params.min_Pcomp = 0.005; %minimum cluster probability (basically minimum firing rate)
end
if ~isfield(params,'max_n_retriggers')
    params.max_n_retriggers = 3; %max number of times to try retriggering
end
if ~isfield(params,'noise_thresh') 
    params.noise_thresh = 3.5; %number of sigma (robust estimate of bkgnd noise) to use as max trigger value
end
if ~isfield(params,'max_back_comps')
    params.max_back_comps = 3; %maximum number of Gaussians to try modeling background noise with
end

params.max_retrig_rate = 250; %in Hz

%% LOAD VOLTAGE SIGNAL
%loads in (high-pass filtered) voltage signal
if ischar(sfile) %if file name given
    [V,Vtime,Fs] = Load_FullV(sfile, params.add_Vmean, params.filt_cutoff,use_chs);
else %if already provided as struct
    V = sfile.V(:,use_chs);
    Vtime = sfile.Vtime;
    Fs = sfile.Fs;
end

%% DETECT SPIKES
if strcmp(params.target_rate,'median')
    target_Nspks = 'median'; %if using fixed amp-threshold
else
    target_Nspks = params.target_rate*length(V)/Fs; %target number of spikes
end

%determine channel from which to trigger spikes
if length(use_chs) == 1
    trig_ch = 1;
elseif length(use_chs) == 2 %if using only 2 chs, assume we're at the edge of the probe
    if use_chs(1) == 1 %if we're at the top edge, trigger off the top ch.
        trig_ch = 1;
    else
        trig_ch = 2; %otherwise, we're at the bottom edge, trigger off the lower ch
    end
elseif length(use_chs) == 3
    trig_ch = 2; %if using 3, trigger off the middle one
else %havent written this condition yet
    error('need to write method to select trigger channel with this number of chs');
end

[spk_id, trig_thresh,noise_sigma] = triggerSpikes(V(:,trig_ch),params.thresh_sign,target_Nspks); %trigger spikes from trig_ch

%check if identified trigger threshold is too high above the estimated noise level. 
% If so, lower threshold and retrigger
if trig_thresh/noise_sigma >= params.noise_thresh
    fprintf('Lowering trigger threshold to %.2f sigma\n',params.noise_thresh);
    new_trig = params.noise_thresh*noise_sigma;
    [spk_id, trig_thresh] = triggerSpikes(V(:,trig_ch),params.thresh_sign,target_Nspks,new_trig);
end
spk_id(spk_id <= abs(params.spk_pts(1)) | spk_id >= length(V)-params.spk_pts(end)) = []; %get rid of spikes at the edges

%extract spike snippets
Spikes = getSpikeSnippets(V,Vtime,spk_id,params.spk_pts,trig_ch);

% detect putative artifacts and get rid of them
artifact_ids = find_spike_artifacts(Spikes);
Spikes.V(artifact_ids,:,:) = [];
Spikes.times(artifact_ids) = [];
Spikes.trig_vals(artifact_ids) = [];
spk_id(artifact_ids) = []; 
if params.verbose > 0
    fprintf('Removed %d potential artifacts\n',length(artifact_ids));
end

[N_spks, N_samps, N_chs] = size(Spikes.V);

%% RUN GMM AUTOCLUSTERING USING MULTIPLE FEATURES AND MULTIPLE INITIALIZATIONS. 
params.trig_ch = trig_ch;
[clusterDetails, spike_xy, spike_features] = autocluster_init(Spikes,params);
if ischar(sfile)
    clusterDetails.rawV_file = sfile;
end
clusterDetails.trig_thresh = trig_thresh;
clusterDetails.trig_ch = trig_ch;
clusterDetails.use_chs = use_chs;
clusterDetails.recDur = length(V)/Fs;
clusterDetails.params = params;
clusterDetails.times = Spikes.times;
clusterDetails.spk_inds = spk_id(:);
clusterDetails.Fs = Fs;
clusterDetails = compute_cluster_stats(clusterDetails,Spikes,spike_features);

%% Check if trigger threshold is too high
if clusterDetails.iso_dists(1) > 2 %if there is a reasonable SU
    needToRetrigger = true;
    n_retriggers = 0;
    while needToRetrigger && n_retriggers < params.max_n_retriggers %try up to max_n_retriggers times
        su_inds = find(clusterDetails.spike_clusts == 2); %SU spikes
        
        %compute the density of SU trigger values
        [nn,xx] = hist(Spikes.trig_vals(su_inds),500); %distribution of SU trigger values
        nn = smooth(nn,50)/sum(nn); %smoothed spike amp density
        if params.thresh_sign > 0
            nn = flipud(nn);
        end
        
        prev_rate = sum(clusterDetails.n_spks)/clusterDetails.recDur; %total spike rate
        new_rate = prev_rate*2; %try doubling trigger rate
        
        %check whether the density of SU triggers is low enough at the cutoff (or if were triggering
        %above the max-rate)
        if nn(end) < max(nn)/5 || new_rate > params.max_retrig_rate
            needToRetrigger = false;
            break;
        end
        
        fprintf('Possible SU cutoff detected, trying retriggering\n');
        new_params = params;
        new_params.target_rate = new_rate;
        fixed = -1; %retrigger and fit new model params
        [clusterDetails,spike_features,spike_xy,Spikes] = apply_clustering(sfile,clusterDetails,new_params,fixed);
        
        %try adding background comps if were not already over our threshold
        cur_n_back_comps = clusterDetails.Ncomps - 1;
        [clusterDetails.gmm_fit,clusterDetails.dprime,clusterDetails.comp_idx,clusterDetails.cluster_labels] = add_background_comps(...
            Spikes,spike_features,clusterDetails.gmm_fit,clusterDetails.dprime,clusterDetails.cluster_labels,clusterDetails.comp_idx,params,cur_n_back_comps);
        n_retriggers = n_retriggers + 1;
    end
end


%% CHECK FOR ADDITIONAL SUs BY FITTING MODELS TO THE BACKGROUND SPIKES RECURSIVELY
max_n_sus = 5; %maximum number of SUs 
if clusterDetails.iso_dists(1) > 2 %if the cluster separation is decent, try finding more SUs
    cur_n_SUs = 1;
    new_cluster = clusterDetails;
    while cur_n_SUs <= max_n_sus
        fprintf('Trying to fit %d SUs\n',cur_n_SUs+1);
        
        %fit model to background spikes (with random init)
        cur_back_spikes = find(new_cluster.spike_clusts == 1); %current background spikes
        [back_GMM, back_dist,back_comp_idx,back_clust_labels,back_cluster_stats] = ...
            GMM_fit(Spikes.V(cur_back_spikes,:),spike_features(cur_back_spikes,:),2,params);
        
        if isobject(back_GMM)
            
            if params.max_back_comps > 1 %try adding additional components to model background spikes
                back_Spikes.trig_vals = Spikes.trig_vals(cur_back_spikes,:);
                back_Spikes.V = Spikes.V(cur_back_spikes,:,:);
                [back_GMM,back_dist,back_comp_idx,back_clust_labels] = add_background_comps(...
                    back_Spikes,spike_features(cur_back_spikes,:),back_GMM,back_dist,back_clust_labels,back_comp_idx,params);
            end
                        
            init_comp_idx = clusterDetails.comp_idx;
            uids = init_comp_idx <= 0;
            buids = back_comp_idx <= 0;
            cuids = setdiff(1:N_spks,[uids; cur_back_spikes(buids)]); %non outlier spikes
            %add in new component
            init_comp_idx = init_comp_idx + length(back_clust_labels);
            init_comp_idx(cur_back_spikes) = back_comp_idx;
            [~,~,init_comp_idx(cuids)] = unique(init_comp_idx(cuids)); %bug fixed 11-15-13
            
            %add in new cluster labels
            init_cluster_labels = clusterDetails.cluster_labels;
            init_cluster_labels(init_cluster_labels == 1) = [];
            init_cluster_labels = [back_clust_labels init_cluster_labels + 1];
            
            %refit GMM with all data
            n_comps = length(init_cluster_labels);
            [back_GMM, back_dist,back_comp_idx,back_clust_labels,back_cluster_stats] = ...
                GMM_fit(Spikes.V,spike_features,n_comps,params,init_comp_idx,init_cluster_labels);
            
            %check if the new SU is good
            [new_Lratio,new_iso_dist] = compute_cluster_Lratio(spike_features,back_GMM,back_comp_idx,back_clust_labels);
            n_good_SUs = sum(new_iso_dist > 2 & new_Lratio < 1e3); %number of good SUs
            if n_good_SUs > cur_n_SUs
                clusterDetails.gmm_fit = back_GMM;
                clusterDetails.comp_idx = back_comp_idx;
                clusterDetails.cluster_labels = back_clust_labels;
                clusterDetails = compute_cluster_stats(clusterDetails,Spikes,spike_features);  
                new_cluster = clusterDetails;
            else
                break;
            end
        end
        cur_n_SUs = cur_n_SUs + 1;
    end
end

%% CREATE SUMMARY FIGURE
if params.summary_plot > 0
    
    sum_fig = create_summary_cluster_fig(clusterDetails,Spikes,spike_xy,clusterDetails.params);
    
end
