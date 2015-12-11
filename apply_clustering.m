function [new_cluster,spike_features,spike_xy,Spikes] = apply_clustering(loadedData,init_cluster,params,fixed,Spikes,spike_features)
% [new_cluster,spike_features,spike_xy,Spikes] = apply_clustering(loadedData,init_cluster,<params>,<fixed>,<Spikes>,<spike_features>)
% Apply GMM clustering model to new spike data
% INPUTS: 
%   loadedData: data file. Either filename, or struct containing voltage, timestamp, and Fs 
%   init_cluster: clusterDetails struct
%   params: clustering params struct
%   fixed: indicator to specify what the function should do (this should be changed to be more transparent)
%       Set fixed == -1 to retriger spikes but dont fit new GMM (not fully implemented yet?)
%       Set fixed == 0 to fit new GMM with input gmm used for initialization. 
%       Set fixed == 1 to just grab spike features.
%       Set fixed == 2 if you want to keep the model the same but recompute cluster stats, etc.
%       Default is 0
%   Spikes: struct of spike data, if precalculated
%   spike_features: array of spike features if precalculated
% OUTPUTS:
%   new_cluster: new clusterDetails struct
%   spike_features: array of spike features
%   spike_xy: XYprojection of spike data
%   Spikes: struct of spike data

%% defaults
if nargin < 3 || isempty(params)
    params = init_cluster.params;
end
if nargin < 4 || isempty(fixed)
    fixed = 0;
end
if nargin < 5
    Spikes = [];
end
if nargin < 6
    spike_features = [];
end
new_cluster = init_cluster;

%% LOAD VOLTAGE SIGNAL AND DETECT SPIKES
if isempty(Spikes) %if spike data isn't provided
    %loads in (high-pass filtered) voltage signal
    if ischar(loadedData)
        [V,Vtime,Fs] = Load_FullV(loadedData, params.add_Vmean, params.filt_cutoff,new_cluster.use_chs);
    else
        V = loadedData.V(:,new_cluster.use_chs);
        Vtime = loadedData.Vtime;
        Fs = loadedData.Fs;
    end
    
    if fixed == -1 %if retriggering
        if ischar(params.target_rate)
            target_Nspks = params.target_rate;
        else
            target_Nspks = params.target_rate*size(V,1)/Fs;
        end
        [spk_id, trig_thresh] = triggerSpikes(V(:,new_cluster.trig_ch),params.thresh_sign,target_Nspks);
        new_cluster.trig_thresh = trig_thresh;
    else
        trig_thresh = new_cluster.trig_thresh;
        [spk_id, trig_thresh] = triggerSpikes(V(:,new_cluster.trig_ch),params.thresh_sign,[],trig_thresh);
    end
    spk_id(spk_id <= abs(params.spk_pts(1)) | spk_id >= length(V)-params.spk_pts(end)) = []; %get rid of spikes at the edges
    
    %extract spike snippets
    Spikes = getSpikeSnippets(V,Vtime,spk_id,params.spk_pts,new_cluster.trig_ch);
    
    % artifact detection
    artifact_ids = find_spike_artifacts(Spikes);
    Spikes.V(artifact_ids,:,:) = [];
    Spikes.times(artifact_ids) = [];
    Spikes.trig_vals(artifact_ids) = [];
    spk_id(artifact_ids) = []; 
    Spikes.spk_inds = spk_id(:);
    if params.verbose > 0
        fprintf('Removed %d potential artifacts\n',length(artifact_ids));
    end
    
    new_cluster.times = Spikes.times;
    new_cluster.spk_inds = spk_id(:);
    new_cluster.recDur = length(V)/Fs;
    
else
    new_cluster.times = Spikes.times;
    new_cluster.spk_inds = Spikes.spk_inds;
end

[N_spks, N_samps, N_chs] = size(Spikes.V);

%% get desired spike features
if isempty(spike_features)
    if strcmp(new_cluster.fin_space,'PC')
        if N_chs > 1
            AllV = reshape(Spikes.V,N_spks,N_samps*N_chs);
        else
            AllV = Spikes.V;
        end
        spike_features = AllV*new_cluster.pc_vecs;
    elseif strcmp(new_cluster.fin_space,'voltage')
        if N_chs > 1
            AllV = reshape(Spikes.V,N_spks,N_samps*N_chs);
        else
            AllV = Spikes.V;
        end
        spike_features = AllV(:,new_cluster.tdims);
    elseif strcmp(new_cluster.fin_space,'energy')
        spike_energy = squeeze(sqrt(sum(Spikes.V.^2,2)));
        spike_dt_energy = squeeze(sqrt(sum(diff(Spikes.V,1,2).^2,2)));
        spike_features = [spike_energy spike_dt_energy];
    elseif strcmp(new_cluster.fin_space,'template')
        templates = new_cluster.templates;
        n_templates = size(templates,2);
        spike_features = get_template_scores(Spikes.V,new_cluster.templates,new_cluster.template_params);
    else
        error('Unrecognized feature space');
    end
end
spike_xy = spike_features*init_cluster.xy_projmat;
new_cluster.spike_xy = spike_xy;

%% apply cluster model
uids = new_cluster.comp_idx > 0; %non outlier spikes

%get cluster assignments
spike_clusts = int16(nan(size(new_cluster.comp_idx)));
spike_clusts(uids) = (new_cluster.cluster_labels(new_cluster.comp_idx(uids)));
new_cluster.spike_clusts = spike_clusts(:);

if ismember(fixed,[-1,0]) %if we want to cluster new data
    
    init_comp_idx = cluster(new_cluster.gmm_fit,spike_features); %apply cluster model to new spike features
    %detect outliers
    mah = sqrt(mahal(new_cluster.gmm_fit,spike_features));
    best_mah = min(mah,[],2);
    outliers = find(best_mah > new_cluster.params.outlier_thresh);
    init_comp_idx(outliers) = -1;
    
    init_cluster_labels = new_cluster.cluster_labels;

    %if any components have no assigned spikes, remove them
    N_initial_clusts = length(unique(init_cluster_labels));
    n = hist(init_comp_idx,1:length(init_cluster_labels));
    init_cluster_labels(n==0) = [];
    if length(unique(init_comp_idx(init_comp_idx > 0))) < 2 %if there is only one cluster apparent in the data try random inits
        gmm_obj = nan;
    else %use initial clustering to initialize new GMM model
        [gmm_obj, gmm_distance, clust_ids, cluster_labels, cluster_stats, outliers] = ...
            GMM_fit(Spikes.V, spike_features, [], new_cluster.params,init_comp_idx,init_cluster_labels);
    end
    
    %if fixed initialization fails try random initialization
    if ~isobject(gmm_obj)
        fprintf('Fixed-initialization failed, trying random initialziation\n');
        [gmm_obj, gmm_distance, clust_ids, cluster_labels, cluster_stats, outliers] = ...
            GMM_fit(Spikes.V, spike_features, length(unique(new_cluster.cluster_labels)), new_cluster.params);
        if ~isobject(gmm_obj)
            new_cluster.spike_clusts = ones(size(new_cluster.times));
            new_cluster.comp_idx = nan(size(new_cluster.times));
            new_cluster.failed = 1;
            return;
        end
    end
    
    %if we need to add back in any clusters which dont have spikes in this
    %data (save as null clusters for bookkeeping)
    if N_initial_clusts > length(unique(cluster_labels))
        n_null_clusts = N_initial_clusts - max(cluster_labels);
        cluster_labels = cat(2,cluster_labels,(1:n_null_clusts)+max(cluster_labels));
    else
        n_null_clusts = 0;
    end
    
    new_cluster.gmm_fit = gmm_obj;
    new_cluster.comp_idx = clust_ids;
    new_cluster.cluster_labels = cluster_labels;
    if ~isempty(gmm_obj)
        new_cluster.LL = gmm_obj.NlogL;
        new_cluster.dprime = gmm_distance;
    else
        new_cluster.LL = nan;
        new_cluster.dprime = nan;
    end
end

%% compute stats on new clustering
if fixed ~= 1    
    if fixed == 0 %if its a new model fit, relabel clusters
        [cluster_labels, cluster_stats] = relabel_clusters(Spikes.V,new_cluster.comp_idx,new_cluster.cluster_labels);
        new_cluster.cluster_labels = cluster_labels;
    elseif ismember(fixed,[-1,2])
        orig_n_clusts = size(init_cluster.mean_spike,2);
        new_n_clusts = length(unique(init_cluster.cluster_labels));
        n_null_clusts = orig_n_clusts - new_n_clusts;
    else
        error('Unsupported option for input fixed');
    end
    new_cluster = compute_cluster_stats(new_cluster,Spikes,spike_features,n_null_clusts);
end
