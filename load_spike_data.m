function Spikes = load_spike_data(dat_name)
% Spikes = load_spike_data(dat_name)
% helper function that loads precomputed spike data file and decompresses it
load(dat_name);
Spikes.V = (double(Spikes.V) + 2^15)*diff(Spikes.Vrange)/2^16 + Spikes.Vrange(1);

