function Spikes = getSpikeSnippets(V,Vtime,spk_id,spk_pts,trig_ch)
% Spikes = getSpikeSnippets(V,spk_id,spk_pts,<trig_ch>)
% extract spike snippets from V
% INPUTS: 
%   V: Txc array of voltage data (T is number timestamps, c is number of chs)
%   Vtime: vector of timestamps
%   spk_id: index value of detected spikes (peaklocs)
%   spk_pts: vector of relative index values (0 is the peak) to extract 
%   <trig_ch>: channel used to trigger spikes
% OUTPUTS:
%   Spikes: struct containing voltage waveform (.V), time of spike peak (.times), and spike peak
%       amplitude (.trig_vals)

if nargin < 5 || isempty(trig_ch)
    trig_ch = 1;
end
allid = bsxfun(@plus,spk_id,spk_pts'); %all the index values we're going to extract
Spikes.V = reshape(V(allid',:),[length(spk_id) length(spk_pts) size(V,2)]);
Spikes.times = Vtime(spk_id);
Spikes.trig_vals = V(spk_id,trig_ch);
