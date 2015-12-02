function [spike_xy,xyproj_mat] = enforce_spikexy_convention(Spikes,spike_xy,xyproj_mat)
% [spike_xy,xyproj_mat] = enforce_spikexy_convention(Spikes,spike_xy,xyproj_mat)
%implement a sign convention that positive values of the first dimension
%correspond to higher spike amplitudes

bb = corr(spike_xy(:,1),abs(Spikes.trig_vals(:)));
if bb < 0
    spike_xy(:,1) = -spike_xy(:,1);
    xyproj_mat(:,1) = -xyproj_mat(:,1);
end
