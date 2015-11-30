function [V,Vtime,Fs] = Load_FullV(name, add_Vmean, filt_cutoff,use_chs)
% [V,Vtime,Fs] = Load_FullV(name, <add_Vmean>, <filt_cutoff>, <use_chs>)
% load voltage data from file and apply desired preprocessing
% INPUTS: 
%   name: file name containing voltage data
%   add_Vmean (default true): Whether or not to add in overall mean voltage (across probes). This is subtracted in
%       many of the raw-data files
%   filt_cutoff: cutoff frequencies of bandpass filtering (default is no filtering). Use nan to
%       indicate no filtering
%   use_chs: Vector of channels to load (default is ch 1).
% OUTPUTS:
%   V: Matrix of voltage values (Time x ch.)
%   Vtime: vector of corresponding timestamps
%   Fs: scalar giving the sample frequency

if nargin < 2 || isempty(add_Vmean)
    add_Vmean = true;
end
if nargin < 3 || isempty(filt_cutoff)
    filt_cutoff = [nan nan];
end
if nargin < 4 || isempty(use_chs) || any(isnan(use_chs))
    use_chs = 1;
end

load(name);

V = double(FullV.V(use_chs,:))'; %grab desired channels and convert to double
%if we're adding the overall (across-probe) avg back into the signal
if add_Vmean
    first_dot = find(name == '.',1);
    Vmean_fname = [name(1:(first_dot-1)) 'FullVmean.mat'];
    if ~exist(Vmean_fname,'file')
        error('Cant find FullVmean file!');
    end
    load(Vmean_fname);
    V = V + FullV.sumscale*sumv;
end

%convert to voltage if the conversion is available
if isfield(FullV,'intscale')
    lfp_int2V = FullV.intscale(1)/FullV.intscale(2);
    V = V*lfp_int2V;
end

Fs = 1/FullV.samper; %get sample-frequency

%apply any filtering using butterworth 2/4-pole
if any(~isnan(filt_cutoff))
    niqf = Fs/2;
    if all(~isnan(filt_cutoff)) %bandpass
        [bb,aa] = butter(2,filt_cutoff/niqf);
    elseif ~isnan(filt_cutoff(1)) %high-pass
        [bb,aa] = butter(2,filt_cutoff(1)/niqf,'high');
    elseif ~isnan(filt_cutoff(2)) %low-pass
        [bb,aa] = butter(2,filt_cutoff(2)/niqf,'low');
    end
    V = filtfilt(bb,aa,V); %symmetric filter
end

%get vector of timestamps if requested
if nargout > 1
    first = 1; 
    Vtime = nan(size(V,1),1);
    for j = 1:length(FullV.blklen) %loop over data chunks
        last = first+FullV.blklen(j)-1; %last index of chunk
        Vtime(first:last) = FullV.blkstart(j)+[1:FullV.blklen(j)].*FullV.samper;
        first = last+1;
    end
    bad_pts = find(isnan(Vtime));
else
    bad_pts = [];
end

%eliminate any bad samples
if ~isempty(bad_pts)
    %     fprintf('Eliminating %d of %d bad V samples\n',length(bad_pts),length(Vtime));
    V(bad_pts,:) = [];
    Vtime(bad_pts) = [];
end