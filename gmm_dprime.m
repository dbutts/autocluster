function [d,Dclust]  = gmm_dprime(G,clust_labels)
% [d]  = gmm_dprime(G, <clust_labels>)
% calcualte drpime between Gaussians in gmdistribution fit
% INPUTS: 
%   G: gmm_fit object
%   clust_labels: cluster labels for each of the gaussian components
% OUTPUTS:
%   d: average dprime 
%   Dclust: matrix of Dprime values

%% input handling
if nargin < 2
    clust_labels = [];
end
if ~isobject(G)
    d = nan;
    Dclust = nan;
    return;
end
nc =size(G.mu,1); %number of components

%%
distance = mahal(G,G.mu); %mahal distances between each pair of Gaussian means
Dmat = zeros(nc,nc); %matrix of dprimes between Gauss components
for j = 1:nc
    for k = 1:j-1
        Dmat(j,k) = sqrt(2./((1./distance(j,k))+(1./distance(k,j)))); %symmetrized D-prime
        Dmat(k,j) = Dmat(j,k);
    end
end
if nc == 2 %if just 2 components there's only one value
    d = gmm_distance(G);
    Dclust = Dmat;
elseif isempty(clust_labels) %if no cluster labels are provided, just assume each component is a different cluster
    d  = mean(Dmat(Dmat>0));
    Dclust = Dmat;
else 
    weights = G.PComponents;
    
    unique_clabels = unique(clust_labels); %set of unique clusters
    for jj = 1:length(unique_clabels)
       cur_set = find(clust_labels == unique_clabels(jj)); %components within this cluster
       weights(cur_set) = weights(cur_set)/sum(weights(cur_set)); %normalize weights of components for this cluster
    end
    Dclust = zeros(length(unique_clabels));
    for jj = 1:(length(unique_clabels)-1) %for each cluster
        gauss_set1 = find(clust_labels == unique_clabels(jj)); %get set of cluster1 components
        for kk = (jj+1):length(unique_clabels) %for each other cluster
            gauss_set2 = find(clust_labels == unique_clabels(kk)); %get set of cluster2 components
%             Dclust(jj,kk) = weights(gauss_set1)*Dmat(gauss_set1,gauss_set2)*weights(gauss_set2)';
            Dclust(jj,kk) = mean(reshape(Dmat(gauss_set1,gauss_set2),1,[])); %compute average of symmetrized D-prime between component gaussians
        end
    end
    d  = mean(Dclust(Dclust>0)); %average of dprimes for all pairs of clusters
end


