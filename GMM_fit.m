function [gmm_obj, gmm_distance, comp_idx, cluster_labels, cluster_stats, outliers] = ...
    GMM_fit(SpikeV, X, n_comps, params,init_cluster_data,init_cluster_labels)

% [gmm_obj, gmm_distance, comp_idx, cluster_labels, cluster_stats, outliers] = GMM_fit(SpikeV, X, n_comps, <params>,<init_cluster_data>,<init_cluster_labels>)
% Fit GMM to spike data. Wraps around Matlab's gmdistribution class, with some added functionality.
% INPUTS:
%   SpikeV: [Nspikes x Nsamps x Nchs] spike waveform data
%   X: [Nspikes x Ndims] array of spike features
%   n_comps: number of Gaussian components
%   <params>: struct of parameters
%   <init_cluster_data>: struct of cluster data or vector of component indices from initial clustering to use for initialization
%   <init_cluster_labels>: %cluster labels of each component in initial model (used when specifying initial model fit with init_cluster_data)
% OUTPUTS:
%   gmm_obj: Matlab GMM object
%   gmm_distance: a measure of the cluster separability
%   comp_idx: component assignment for each spike
%   cluster_labels: cluster label for each gaussian component
%   cluster_stats: statistics for each cluster
%   outliers: indices of spikes deemed outliers

warning('off','stats:gmdistribution:FailedToConverge');

%% SET DEFAULTS
if nargin < 4 || isempty(params)
    params = struct();
end
if nargin < 5
    init_cluster_data = [];
end
if nargin < 6 
    init_cluster_labels = 1:n_comps; 
end
if ~isempty(init_cluster_data)
    use_init = true;
else
    use_init = false;
end
if ~isfield(params,'gmm_inits')
    params.gmm_inits = 50;
end
if ~isfield(params,'outlier_thresh')
    params.outlier_thresh = 7;
end
if ~isfield(params,'reg_lambda')
    params.reg_lambda = 1e-5;
end
if ~isfield(params,'max_iters')
    params.max_iters = 10;
end
if ~isfield(params,'TolFun')
    params.TolFun = 1e-4;
end
if ~isfield(params,'min_Pcomp')
    params.min_Pcomp = .0001;
end

%%
fail = zeros(10,1); %init

if size(X,3) > 1 %flatten spikewaveforms across channels
    X = reshape(X,size(X,1),size(X,2)*size(X,3));
end

%% if an initial clustering is specified
if use_init 
    if isstruct(init_cluster_data) %if providing struct info
        n_comps = size(init_cluster_data.mu,1);
        uids = 1:size(X,1);
        Gs{1} = fit_GMM(X(uids,:),n_comps,'Start',init_cluster_data);   
    else %if providing initial cluster assignments
        uids = init_cluster_data >= 1;
        n_comps = length(unique(init_cluster_data(uids)));
        if length(unique(init_cluster_data(uids))) == 1
            init_cluster_data = ceil(rand(size(init_cluster_data))*n_comps);
        end
        if length(unique(init_cluster_data(uids))) == 1
            Gs{1} = nan;
        else
            Gs{1} = fit_GMM(X(uids,:),n_comps,'Start',init_cluster_data(uids));
        end
    end
    ds(1) = gmm_dprime(Gs{1}); %compute variant of d-prime to assess cluster quality
    clust_labels{1} = init_cluster_labels;
    if isobject(Gs{1})
        comp_idx = cluster(Gs{1},X);
    else
        comp_idx = nan;
        fail(1) = 1;
    end
    [L(:,1),iso_distance(:,1)] = compute_cluster_Lratio(X,Gs{1},comp_idx,clust_labels{1});
else
    %% IF NO INITIAL CLUSTER INFO IS PROVIDED, TRY FITTING WITH RANDOM INITS
    temp_Gs = cell(params.gmm_inits,1);
    temp_ds = nan(params.gmm_inits,1);
    temp_min_pcomp = nan(params.gmm_inits,1);
    for ii = 1:params.gmm_inits %try this many models with different random init and pick the best according to dprime
        if params.reg_lambda > 0 %if using regularization on the cov mats
            temp_Gs{ii} = fit_GMM(X,n_comps,'Regularize',params.reg_lambda,'Options',statset('MaxIter',params.max_iters,'TolFun',params.TolFun));
        else
            temp_Gs{ii} = fit_GMM(X,n_comps,'Options',statset('MaxIter',params.max_iters,'TolFun',params.TolFun));
        end
        temp_ds(ii) = gmm_dprime(temp_Gs{ii}); 
        if ~isnan(temp_ds(ii))
            temp_min_pcomp(ii) = min(temp_Gs{ii}.PComponents);
        end
    end
    temp_ds(temp_min_pcomp < params.min_Pcomp) = nan; %dont count any clusterings where there is too little probability assigned to one of the clusters.
    [ds(1),best] = max(temp_ds); %find best clustering
    if ~isnan(ds(1))
        Gs{1} = temp_Gs{best};
        clust_labels{1} = 1:n_comps;
        comp_idx = cluster(Gs{1},X);
        [L(:,1),iso_distance(:,1)] = compute_cluster_Lratio(X,Gs{1},comp_idx,clust_labels{1});
        fail(1) = 0;
    else
        fail(1) = 1;
        Gs{1} = nan;
        ds(1) = nan;
        L(:,1) = nan;
        iso_distance(:,1) = nan;
    end
    
    %% TRY FITTING WITH PREDEFINED INIT
    %sets the initial component means to be separated along the first PC. If 3
    %clusters, seperates the 3rd along the 2nd PC. Initializes the component
    %covariances to all be some fraction of the full covariance of X
    
    if size(X,2) > 1
        C = cov(X);
        [E,V] = eig(C);
        pc = E(:,end); %first PC
        pcb = E(:,end-1); %second pc
        pc = pc./max(abs(pc)); %normalize to have max 1
        pcb = pcb./max(abs(pcb));
        for j = 1:size(X,2)
            S.mu(1,j) = mean(X(:,j)) + pc(j);
            S.mu(2,j) = mean(X(:,j)) - pc(j);
            if n_comps == 3
                S.mu(3,j) = mean(X(:,j)) + pcb(j);
            end
        end
        for j = 1:n_comps
            S.Sigma(:,:,j) = C./sqrt(2);
        end
        
        Gs{2} = fit_GMM(X,n_comps,'Start',S,'Regularize',params.reg_lambda);
        ds(2) = gmm_dprime(Gs{2});
        clust_labels{2} = 1:n_comps;
        if isobject(Gs{2})
            comp_idx = cluster(Gs{2},X);
        else
            comp_idx = nan;
            fail(2) = 1;
        end
        [L(:,2),iso_distance(:,2)] = compute_cluster_Lratio(X,Gs{2},comp_idx,clust_labels{2});
    end
    
    %% TRY INITIALIZING WITH K-MEANS
    kmeans_idx = kmeans(X,n_comps);
    Gs{3} = fit_GMM(X,n_comps,'Regularize',params.reg_lambda,'Start',kmeans_idx);
    ds(3) = gmm_dprime(Gs{3});
    clust_labels{3} = 1:n_comps;
    if isobject(Gs{3})
        comp_idx = cluster(Gs{3},X);
    else
        comp_idx = nan;
        fail(3) = 1;
    end
    [L(:,3),iso_distance(:,3)] = compute_cluster_Lratio(X,Gs{3},comp_idx,clust_labels{3});
    
end
%% PICK BEST FIT
if all(fail == 1)
    fprintf('No GMM fits succeeded! Aborting.\n');
    gmm_obj = [];
    gmm_distance = nan;
    comp_idx = ones(size(X,1),1);
    cluster_labels = 1;
    cluster_stats = get_cluster_stats(SpikeV,comp_idx);
    outliers = [];
    return;
end
[gmm_distance,best] = max(ds); %pick clustering with largest cluster sep
gmm_obj = Gs{best};
cluster_labels = clust_labels{best};

%% CHECK FOR OUTLIERS
if ~isnan(params.outlier_thresh) && isobject(gmm_obj)
    use_nclusts = size(gmm_obj.mu,1);
    [idx,nlogl,P,logpdf,M] = cluster(gmm_obj,X);
    min_mah_dists = sqrt(min(M,[],2)); %smallest mahalanobis distance for each spike to any cluster
    outliers = find(min_mah_dists > params.outlier_thresh); 
    use_idx = setdiff(1:size(X,1),outliers); %non-outlier spikes
    S.mu = gmm_obj.mu; S.Sigma = gmm_obj.Sigma;
    new_gmm_obj = fit_GMM(X(use_idx,:),use_nclusts,'Start',S,'Regularize',params.reg_lambda); %refit models excluding outliers
    if isobject(new_gmm_obj)
        gmm_obj = new_gmm_obj;
    end
    gmm_distance = gmm_dprime(gmm_obj,cluster_labels);
    comp_idx = cluster(gmm_obj,X);
    comp_idx(outliers) = -1;
else
    if isobject(gmm_obj)
        comp_idx = cluster(gmm_obj,X);
    else
        comp_idx = nan;
    end
    outliers = [];
end

if isobject(gmm_obj)
    %make sure smallest Pcomp is bigger than minPcomp
    if min(gmm_obj.PComponents) < params.min_Pcomp
        gmm_distance = nan;
    end
    
    [cluster_labels,cluster_stats] = relabel_clusters(SpikeV,comp_idx,cluster_labels);%re-order clusters based on amp of avg waveform
else
    
    gmm_distance = nan;
    cluster_labels = nan;
    cluster_stats = nan;
end
