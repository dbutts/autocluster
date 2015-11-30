function distance  = gmm_distance(G)
% distance  = gmm_distance(G)
% compute the symmetrized dprime between two gaussians in gmm_fit object G
try
    D = mahal(G,G.mu);
    distance = sqrt(2./((1./D(1,2))+(1./D(2,1))));
catch %in case fit failed
    distance = 0;
end
