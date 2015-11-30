function artifact_ids = find_spike_artifacts(Spikes)
% artifact_ids = find_spike_artifacts(Spikes,params)
% locates putative artifacts in detected Spikes struct
% INPUTS: 
%   Spikes: struct of spike data
% OUTPUTS:
%   artifact_ids: index values of spike events that are putative artifacts
%NOTE: this code is directly copied from AllVPcs. Haven't looked at it in detail yet.

meanV = squeeze(mean(Spikes.V,3))'; %get average spike waveforms (over chs)

allbid = []; %init vector of bad spikes
gid = 1:size(Spikes.V,1); %initialize vector of good spike ids
avar = sum(meanV(:,gid).^2); %compute sum squares over time samples for each spike
bid = find(avar > prctile(avar,99) * 2); %potential artifacts are ones that are outliers in terms of variance over time
while ~isempty(bid)
    allbid = [allbid bid];
    gid = setdiff(1:size(Spikes.V,1),allbid); %recompute set of good spikes without the current artifacts
    avar = sum(meanV(:,gid).^2); %recompute avar for good spikes
    bid = find(avar > prctile(avar,99) * 2); %find outliers
    bid = gid(bid); %new set of artifacts
end

avar = sum(abs(diff(meanV(:,gid)))); %now compute avar in terms of the summed derivative magnitude
% bvar = smooth(avar,10); %smooth across nearby spikes
bid = find(avar > prctile(avar,99) * 1.5); %find outliers in terms of avar
while ~isempty(bid) %repeat the same process
    allbid = [allbid bid];
    gid = setdiff(1:size(Spikes.V,1),allbid);
    avar = sum(abs(diff(meanV(:,gid))));
    bid = find(avar > prctile(avar,99) * 1.5);
    bid = gid(bid);
end
bid = allbid;

artifact_ids = bid;


