function [new_gmm_fit, gmm_distance, new_comp_idx, new_clust_labels] = ...
    try_backgnd_splitting(SpikeV,spike_features,spike_xy,comp_idx,clust_labels,params)
% [new_gmm_fit, gmm_distance, new_comp_idx, new_clust_labels] = try_backgnd_splitting(gmm_fit,SpikeV,spike_features,spike_xy,xyproj_mat,comp_idx,clust_labels,params)
% split the background cluster and refit GMM. Assumes there is one SU cluster (cluster_label==2) at
% this point

su_comp = find(clust_labels == 2); 
su_set = find(comp_idx == su_comp); %SU spikes

%split SU set along the first dimension of spike_xy
med_pt = median(spike_xy(su_set,1));
% med_pt = prctile(spike_xy(su_set,1),90);

new_sucomp = su_set(spike_xy(su_set,1) > med_pt);
new_backcomp = su_set(spike_xy(su_set,1) < med_pt); %assign points with smaller values of spike_x to background

%recompute component assignments and cluster_labels
new_comp_idx = comp_idx;
new_comp_idx(new_sucomp) = length(clust_labels) + 1; %make new component
new_comp_idx(new_backcomp) = su_comp; %make the new background comp = 2
new_clust_labels = clust_labels;
new_clust_labels(su_comp) = 1; %this is the second background comp
new_clust_labels(end+1) = 2; %this is the new SU comp

%refit GMM with this initialization
[new_gmm_fit, gmm_distance, new_comp_idx, new_clust_labels] = ...
    GMM_fit(SpikeV, spike_features, [], params, new_comp_idx,new_clust_labels);

if isobject(new_gmm_fit)
    %make sure smallest Pcomp is bigger than minPcomp
    if min(new_gmm_fit.PComponents) < params.min_Pcomp
        gmm_distance = nan;
    end
    
    [new_clust_labels] = relabel_clusters(SpikeV,new_comp_idx,1:length(new_clust_labels));
    new_su_comp = new_clust_labels(end);
    new_clust_labels = ones(size(new_clust_labels));
    new_clust_labels(new_su_comp) = 2;
    gmm_distance = gmm_dprime(new_gmm_fit,new_clust_labels);
else
    gmm_distance = nan;
    new_clust_labels = nan;
end



   
    
    