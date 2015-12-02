function template_scores = get_template_scores(SpikeV,templates,template_params)
% [template_scores,templates_used] = get_template_scores(SpikeV,templates,template_params)
% calculate template scores for spike waveforms given a set of templates
% INPUTS:
%   SpikeV: spike voltage waveforms
%   templates: set of template waveforms
%   template_params: struct of params describing template scoring params
% OUTPUTS:
%   template_scores: [N x D] array of template scores

%%
[N_spks,D,n_chs] = size(SpikeV);
n_templates = size(templates,2);
channels = template_params.channels;
t_derivs = template_params.t_derivs;
use_ms = template_params.mean_sub;

template_scores = nan(N_spks,n_templates);
for tt = 1:n_templates
   cur_spikes = squeeze(SpikeV(:,:,channels(tt)));
   if t_derivs(tt) == 1
       cur_spikes = [zeros(N_spks,1) diff(cur_spikes,1,2)];
   end
   if use_ms == 1 %if doing mean-subtraction 
      cur_spikes = bsxfun(@minus,cur_spikes,mean(cur_spikes,2));
      templates(:,tt) = templates(:,tt) - mean(templates(:,tt));
   end
   template_scores(:,tt) = cur_spikes*templates(:,tt);
end
template_scores = bsxfun(@rdivide,template_scores,std(template_scores)); %normalize by SD along each score dim