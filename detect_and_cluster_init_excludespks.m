function [clusterDetails,all_spike_features,sum_fig,Spikes] = detect_and_cluster_init_excludespks(sfile,params,use_chs,exclude_spk_inds)
% [clusterDetails,all_spike_features,sum_fig,Spikes] = detect_and_cluster_init_excludespks(sfile,params,<use_chs>,<exclude_spk_inds>)
% like detect_and_cluster_init, but with specified set of spikes excluded
% INPUTS:
%   sfile: data file. Either filename, or struct containing voltage, timestamp, and Fs 
%   <params>: struct of params
%   <use_chs>: channels to use for clustering (default is all channels provided)
%   <exclude_spk_inds>: index values of spikes to exclude from clustering
% OUTPUTS:
%   clusterDetails: struct containing information about the resulting clustering
%   all_spike_features: (Nxp) array of features extracted from ALL spikes, N is num spikes, p is num dims
%   sum_fig: handle for summary figure
%   Spikes: Spike data struct (NOT including excluded spikes)

%% DEFAULT PARAMETERS
if nargin < 3 || isempty(use_chs)
    use_chs = nan;
end
if nargin < 4 
    exclude_spk_inds = [];
end

if nargin < 2 || isempty(params)
    params = struct();
end
if ~isfield(params,'filt_cutoff') %high-pass for filtering raw V for spike detection
    params.filt_cutoff = [100 nan]; 
end
if ~isfield(params,'add_Vmean') %whether or not to add in across-electrode average FullV
    params.add_Vmean = 0;
end
if ~isfield(params,'thresh_sign') %whether to detect on peaks or valleys
    params.thresh_sign = -1;
end
if ~isfield(params,'target_rate') %target spike detection rate. 
    params.target_rate = 50;
end
if ~isfield(params,'spk_pts')
    params.spk_pts = [-12:27]; %set of time points relative to trigger to use for classification
end
if ~isfield(params,'outlier_thresh')
    params.outlier_thresh = 7; %threshold on Mahal distance (sqrt) to count a point as an outlier
end
if ~isfield(params,'verbose')
    params.verbose = 2; %display text
end
if ~isfield(params,'use_best_only')
    params.use_best_only = 0; %use only the best spike waveform for calculating template features.
end
if ~isfield(params,'cluster_bias')
    params.cluster_bias = 0.85; %bias to control type 2 errors for classifying SUs
end
if ~isfield(params,'summary_plot')
    params.summary_plot = 1; 
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
    min_Pcomp = 0.005; %minimum cluster probability (basically minimum firing rate)
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

%% LOAD VOLTAGE SIGNAL
%loads in (high-pass filtered) voltage signal
if ischar(sfile)
    [V,Vtime,Fs] = Load_FullV(sfile, params.add_Vmean, params.filt_cutoff,use_chs);
else
    V = sfile.V(:,use_chs);
    Vtime = sfile.Vtime;
    Fs = sfile.Fs;
end
%% DETECT SPIKES
if strcmp(params.target_rate,'median')
    target_Nspks = 'median';
else
    target_Nspks = params.target_rate*length(V)/Fs;
end

%determine channel from which to trigger spikes
if length(use_chs) == 1
    trig_ch = 1;
elseif length(use_chs) == 2
    if use_chs(1) == 1
        trig_ch = 1 ;
    else
        trig_ch = 2;
    end
else
    trig_ch = 2;
end

[spk_id, trig_thresh,noise_sigma] = triggerSpikes(V(:,trig_ch),params.thresh_sign,target_Nspks);%trigger spikes from trig_ch

%check if identified trigger threshold is too high above the estimated
%noise level. If so, lower threshold and retrigger
if trig_thresh/noise_sigma >= params.noise_thresh
    fprintf('Lowering trigger threshold to %.2f sigma\n',params.noise_thresh);
    new_trig = params.noise_thresh*noise_sigma;
    [spk_id, trig_thresh] = triggerSpikes(V(:,trig_ch),params.thresh_sign,target_Nspks,new_trig);
end
spk_id(spk_id <= abs(params.spk_pts(1)) | spk_id >= length(V)-params.spk_pts(end)) = []; %get rid of spikes at the edges

%extract spike snippets
Spikes = getSpikeSnippets(V,Vtime,spk_id,params.spk_pts,trig_ch);

% artifact detection
artifact_ids = find_spike_artifacts(Spikes);
Spikes.V(artifact_ids,:,:) = [];
Spikes.times(artifact_ids) = [];
Spikes.trig_vals(artifact_ids) = [];
spk_id(artifact_ids) = [];
if params.verbose > 0
    fprintf('Removed %d potential artifacts\n',length(artifact_ids));
end

%save data for ALL spikes
all_Spikes = Spikes;
all_spk_id = spk_id;
all_times = Spikes.times;

%exclude desired spikes
exclude_spks = find(ismember(spk_id,exclude_spk_inds)); %find set of excluded spikes
Spikes.V(exclude_spks,:,:) = [];
Spikes.times(exclude_spks) = [];
Spikes.trig_vals(exclude_spks) = [];
spk_id(exclude_spks) = [];
[N_spks, N_samps, N_chs] = size(Spikes.V);

%% GMM AUTOCLUSTERING USING MULTIPLE FEATURES AND MULTIPLE INITIALIZATIONS. 
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
clusterDetails.Fs = Fs;
clusterDetails.spk_inds = spk_id(:);

clusterDetails = compute_cluster_stats(clusterDetails,Spikes,spike_features);

%% CHECK FOR ADDITIONAL SUs BY FITTING MODELS TO THE BACKGROUND SPIKES RECURSIVELY
max_n_sus = 5;
if clusterDetails.iso_dists(1) > 2
    cur_n_SUs = 1;
    new_cluster = clusterDetails;
    while cur_n_SUs <= max_n_sus
        fprintf('Trying to fit %d SUs\n',cur_n_SUs+1);
        
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
            cuids = setdiff(1:N_spks,[uids; cur_back_spikes(buids)]);
            init_comp_idx = init_comp_idx + length(back_clust_labels);
            init_comp_idx(cur_back_spikes) = back_comp_idx;
            [~,~,init_comp_idx(cuids)] = unique(init_comp_idx(cuids)); %bug fixed 11-15-13
            
            init_cluster_labels = clusterDetails.cluster_labels;
            init_cluster_labels(init_cluster_labels == 1) = [];
            init_cluster_labels = [back_clust_labels init_cluster_labels + 1];
            
            n_comps = length(init_cluster_labels);
            [back_GMM, back_dist,back_comp_idx,back_clust_labels,back_cluster_stats] = ...
                GMM_fit(Spikes.V,spike_features,n_comps,params,init_comp_idx,init_cluster_labels);
            
            [new_Lratio,new_iso_dist] = compute_cluster_Lratio(spike_features,back_GMM,back_comp_idx,back_clust_labels);
            n_good_SUs = sum(new_iso_dist > 2 & new_Lratio < 1e3);
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

%% NOW COMPUTE ISOLATION MEASURES WITH RESPECT TO ALL SPIKES
%get all spike features
if strcmp(clusterDetails.fin_space,'PC')
    if N_chs > 1
        AllV = reshape(all_Spikes.V,length(all_Spikes.times),N_samps*N_chs);
    else
        AllV = all_Spikes.V;
    end
    all_spike_features = AllV*clusterDetails.pc_vecs;
elseif strcmp(clusterDetails.fin_space,'voltage')
    if N_chs > 1
        AllV = reshape(all_Spikes.V,length(all_Spikes.times),N_samps*N_chs);
    else
        AllV = all_Spikes.V;
    end
    all_spike_features = AllV(:,clusterDetails.tdims);
elseif strcmp(clusterDetails.fin_space,'energy')
    spike_energy = squeeze(sqrt(sum(all_Spikes.V.^2,2)));
    spike_dt_energy = squeeze(sqrt(sum(diff(all_Spikes.V,1,2).^2,2)));
    all_spike_features = [spike_energy spike_dt_energy];
elseif strcmp(clusterDetails.fin_space,'template')
    templates = clusterDetails.templates;
    n_templates = size(templates,2);
    all_spike_features = get_template_scores(all_Spikes.V,clusterDetails.templates,clusterDetails.template_params);
else
    error('Unrecognized feature space');
end
[clusterDetails.Lratios,clusterDetails.iso_distance] = compute_cluster_Lratio(all_spike_features,clusterDetails.gmm_fit,clusterDetails.comp_idx,clusterDetails.cluster_labels)

all_spike_xy = all_spike_features*clusterDetails.xy_projmat;

%make excluded spikes -1
included_spikes = setdiff(1:length(all_spk_id),exclude_spks);
all_comp_idx = -1*ones(length(all_spk_id),1);
all_comp_idx(included_spikes) = clusterDetails.comp_idx;
all_spike_clusts = -1*ones(length(all_spk_id),1);
all_spike_clusts(included_spikes) = clusterDetails.spike_clusts;

clusterDetails.times = all_times;
clusterDetails.spk_inds = all_spk_id;
clusterDetails.spike_xy = all_spike_xy;
clusterDetails.comp_idx = all_comp_idx;
clusterDetails.spike_clusts = all_spike_clusts;


%% CREATE SUMMARY FIGURE
if params.summary_plot > 0
    
    sum_fig = create_summary_cluster_fig(clusterDetails,all_Spikes,all_spike_xy,clusterDetails.params);
    
end
