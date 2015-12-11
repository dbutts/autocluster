function [GMM_obj,distance,comp_idx,cluster_labels] = add_background_comps(...
    Spikes,spike_features,GMM_obj,distance,cluster_labels,comp_idx,params,init_back_comps)
% [GMM_obj,distance,comp_idx,cluster_labels] = add_background_comps(Spikes,spike_features,GMM_obj,distance,cluster_labels,comp_idx,params,<init_back_comps>)
%try adding additional gaussian components to model the background dist
% INPUTS:
%   Spikes: struct of spike data
%   spike_featurs: array of spike features
%   GMM_obj: initial GMM fit object
%   distance: measure of initial cluster quality
%   cluster_labels: labels of initial gaussian comps
%   comp_idx: component assignments of each spike
%   <init_back_comps>: number of background components that were starting with (assumes 1)
% OUTPUTS:
%   GMM_obj: new GMM fit object
%   distance: new cluster quality
%   comp_idx: new component assignments
%   cluster_labels: new component cluster labels
%%
if nargin < 8 || isempty(init_back_comps)
    cur_n_back_comps = 1;
else
    cur_n_back_comps = init_back_comps;
end
while cur_n_back_comps < params.max_back_comps
    cur_n_back_comps = cur_n_back_comps + 1;
    fprintf('Trying background split %d of %d\n',cur_n_back_comps,params.max_back_comps);
    
    [spike_xy,xyproj_mat] = Project_GMMfeatures(spike_features, GMM_obj,cluster_labels);
    [spike_xy,xyproj_mat] = enforce_spikexy_convention(Spikes,spike_xy,xyproj_mat);
    
    %split background cluster and fit new model
    [new_GMM_obj, new_distance,new_comp_idx, new_clust_labels] = ...
        try_backgnd_splitting(Spikes.V,spike_features,spike_xy,comp_idx,cluster_labels,params);
    
    fprintf('Orig: %.3f New: %.3f \n',distance,new_distance);
    if new_distance > distance
        GMM_obj = new_GMM_obj;
        distance = new_distance;
        comp_idx = new_comp_idx;
        cluster_labels = new_clust_labels;
    end
end
