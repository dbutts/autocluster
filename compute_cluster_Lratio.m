function [Lratio,iso_distance] = compute_cluster_Lratio(X,gmm_fit,comp_idx,cluster_labels)
% [Lratio,iso_distance] = compute_cluster_Lratio(X,gmm_fit,comp_idx,cluster_labels)
% computes the Lratio and iso_distance metrics for a given clustering
% INPUTS: 
%   X: set of spike features
%   gmm_fit: gmmdistribution object
%   comp_idx: assigned component indices for each spike
%   cluster_labels: vector of cluster labels for each Gaussian component
% OUTPUTS:
%   Lratio: 
%   iso_distance:
%%
%handle case when gmm_fit is not a model object
if ~isobject(gmm_fit)
    Lratio = nan;
    iso_distance = nan;
    return;
end

N_sus = nanmax(cluster_labels)-1; %number of SUs
[N_spks,df] = size(X);
if N_sus == 0 %if there are no SUs
    Lratio = nan;
    iso_distance = nan;
    return;
end

Lratio = nan(N_sus,1);
iso_distance = nan(N_sus,1);
for ii = 1:N_sus
    cluster_comps = find(cluster_labels == ii+1); %set of components for this cluster
    clust_spikes = find(ismember(comp_idx,cluster_comps)); %set of spikes for this cluster
    non_clust_spikes = setdiff(1:N_spks,clust_spikes); %non-cluster spikes
    
    if ~isempty(clust_spikes)
        %if single-component cluster, use the model stats for that comp
        if length(cluster_comps) == 1
            clust_mean = gmm_fit.mu(cluster_comps,:);
            clust_Sigma = squeeze(gmm_fit.Sigma(:,:,cluster_comps));
        else %use empirical mean if more than one gaussians used for cluster
            clust_mean = mean(X(clust_spikes,:));
            clust_Sigma = cov(X(clust_spikes,:));
        end
        mean_seps = bsxfun(@minus,X,clust_mean);
        D = sum((mean_seps/clust_Sigma) .* mean_seps,2); %mahalanobis D
        
        Lratio(ii) = sum(1 - chi2cdf(D(non_clust_spikes),df)); %see Schmitzer-Torbert Neuroscience 2005
                
        if isempty(clust_spikes) || isempty(non_clust_spikes)
            iso_distance(ii) = nan; %cant compute iso-distance if there's not at least some spikes in both the cluster and background
        else
            D = sqrt(sort(D(non_clust_spikes))); %again, see Schmitzer-Torbert 2005
            if length(clust_spikes) <= length(non_clust_spikes)
                iso_distance(ii) = D(length(clust_spikes));
            else
                iso_distance(ii) = D(end);
            end
        end
    else
        Lratio(ii) = nan;
        iso_distance(ii) = nan;
    end
end