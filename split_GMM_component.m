function [] = split_GMM_component(block_num,probe_num,precomp_spike_data)
% [] = split_GMM_component(block_num,probe_num,<precomp_spike_data>)
% split GMM component using GUI input
% INPUTS:
%   block_num
%   probe_num
%   <precomp_spike_data>: optional name of precomputed spike data file

if nargin < 3
    precomp_spike_data = [];
end

%%
global data_dir base_save_dir init_save_dir Expt_name Vloaded n_probes loadedData raw_block_nums

fprintf('Loading block %d Clusters\n',block_num);
cur_clust_data = [base_save_dir sprintf('/Block%d_Clusters.mat',block_num)];
load(cur_clust_data,'Clusters');

cluster = Clusters{probe_num};
new_cluster = cluster;

if ~isempty(precomp_spike_data) %if precomputed data file provided, load from there
    loadedData = [];
    Vloaded = nan;
    Spikes = load_spike_data(precomp_spike_data);
    [cluster,spike_features,spike_xy,Spikes] = apply_clustering(loadedData,cluster,[],1,Spikes); %use precomp spikes
else %load raw data and apply clutsering
    if Expt_name(1) == 'G'
        loadedData = [data_dir sprintf('/Expt%d.p%dFullV.mat',raw_block_nums(block_num),probe_num)];
    else
        sfile_name = [data_dir sprintf('/Expt%dFullV.mat',raw_block_nums(block_num))];
        if Vloaded ~= raw_block_nums(block_num)
            fprintf('Loading data file %s\n',sfile_name);
            [loadedData.V,loadedData.Vtime,loadedData.Fs] = Load_FullV(sfile_name, false, [100 nan],1:n_probes);
            Vloaded = raw_block_nums(block_num);
        end
    end
    
    [cluster,spike_features,spike_xy,Spikes] = apply_clustering(loadedData,cluster,[],1); %trigs spikes and applies clustering
end
N_spks = size(spike_xy,1);
N_sus = length(unique(cluster.cluster_labels)) - 1;
spk_inds = find(cluster.spike_clusts > 0); %non outlier spikes

%% display clustering
clear h leg_labels
f1 = figure();
subplot(2,1,1)
hold on
plot(spike_xy(spk_inds,1),spike_xy(spk_inds,2),'k.');
cmap = cluster_cmap(length(cluster.cluster_labels));
for ii = 1:length(cluster.cluster_labels)
    h(ii) = plot_gaussian_2d(cluster.gmm_xyMeans(ii,:)',squeeze(cluster.gmm_xySigma(:,:,ii)),[2],cmap(ii,:),2);
    leg_labels{ii} = sprintf('Component %d',ii);
end
legend(h,leg_labels);
axis tight
xl = xlim(); yl = ylim();

subplot(2,1,2);hold on
[handles, details] = DensityPlot_jmm(spike_xy(:,1),spike_xy(:,2),'sqrtsc','ynormal','sd',[1 1]);
set(gca,'ytick',[]);
if cluster.template_it == -1
    title('Used PCs','fontsize',12);
elseif cluster.template_it == -2
    title('Used voltage proj','fontsize',12);
elseif cluster.template_it == -3
    title('Used energy proj','fontsize',12);
else
    title('Used template projection','fontsize',12);
end
xlim(xl); ylim(yl);
fp = get(gcf,'Position'); fp(4) = fp(4) + 600; fp(3) = fp(3) + 100;
set(gcf,'Position',fp);

%% prompt user to make a split using two points and fit new GMM
target_comp = input('Which component do you want to split?');
if ~isempty(target_comp)
    target_inds = find(ismember(cluster.comp_idx,target_comp));
    subplot(2,1,1)
    fprintf('Create line with two points\n');
    [x,y] = ginput(2);
    
    %now split spike data based on this separating vec
    sep_vec = [x(2)-x(1); y(2)-y(1)];
    orth_vec = [0 1;-1 0]*sep_vec;
    proj_data = spike_xy*orth_vec;
    proj_thresh = [x(1) y(1)]*orth_vec;
    above_thresh = target_inds(proj_data(target_inds) > proj_thresh);
    
    N_comps = cluster.Ncomps + 1;
    init_comp_idx = cluster.comp_idx;
    init_comp_idx(above_thresh) = max(init_comp_idx) + 1;
    init_cluster_labels = cluster.cluster_labels;
    init_cluster_labels(end+1) = max(init_cluster_labels) + 1;
    
    %fit new GMM with this initialization
    [new_cluster.gmm_fit, new_cluster.dprime, new_cluster.comp_idx, new_cluster.cluster_labels, new_cluster.cluster_stats, outliers] = ...
        GMM_fit(Spikes.V, spike_features, N_comps, cluster.params,init_comp_idx,init_cluster_labels);
    if ~isobject(new_cluster.gmm_fit)
        fprintf('GMM fitting failed, aborting...\n');
        if ishandle(f1)
            close(f1);
        end
        return;
    end
    %compute new XY gaussian stats
    new_cluster.gmm_xyMeans = new_cluster.gmm_fit.mu*cluster.xy_projmat;
    for ii = 1:size(new_cluster.gmm_fit.Sigma,3)
        new_cluster.gmm_xySigma(:,:,ii) = cluster.xy_projmat' * squeeze(new_cluster.gmm_fit.Sigma(:,:,ii)) * cluster.xy_projmat;
    end
end

%get new spike labels
spike_labels = zeros(size(new_cluster.comp_idx));
uids = find(new_cluster.comp_idx > 0);
spike_labels(uids) = new_cluster.cluster_labels(new_cluster.comp_idx(uids));
spk_inds = find(spike_labels >= 1);

N_sus = nanmax(new_cluster.cluster_labels) - 1;
cmap = jet(N_sus);
%replot spikes and gaussian ellipses
subplot(2,1,1);hold off;
plot(spike_xy(spk_inds,1),spike_xy(spk_inds,2),'k.');
hold on
cmap = cluster_cmap(length(new_cluster.cluster_labels));
clear h leg_labels
for ii = 1:length(new_cluster.cluster_labels)
    h(ii) = plot_gaussian_2d(new_cluster.gmm_xyMeans(ii,:)',squeeze(new_cluster.gmm_xySigma(:,:,ii)),[2],cmap(ii,:),2);
    leg_labels{ii} = sprintf('Comp %d',ii);
end
legend(h,leg_labels);
axis tight

%% get user input for new cluster labels and plot new cluster assignments
new_labels = input('What are the cluster labels for the components (input as vector)?\n');
while length(new_labels) ~= size(new_cluster.gmm_xyMeans,1)
    fprintf('Must assign a label to each component!\n');
    new_labels = input('What are the cluster labels for the components (input as vector)?\n');
end
new_cluster.cluster_labels = new_labels;
spike_labels(uids) = new_cluster.cluster_labels(new_cluster.comp_idx(uids));
mu_inds = find(spike_labels == 1);
N_sus = max(new_cluster.cluster_labels) - 1;
cmap = cluster_cmap(N_sus); %color map for SUs
clear h leg_labels;
new_cluster.spike_clusts = spike_labels;

%now make new spike scatter plot color coded by clusters
hold off
h(1) = plot(spike_xy(mu_inds,1),spike_xy(mu_inds,2),'k.');
hold on
leg_labels{1} = sprintf('Cluster %d',1);
hold on
leg_cnt = 2;
for ii = 1:N_sus
    if sum(spike_labels == ii + 1) > 0
    h(ii+1) = plot(spike_xy(spike_labels == ii + 1,1),spike_xy(spike_labels == ii + 1,2),'.','color',cmap(ii,:));
    leg_labels{leg_cnt} = sprintf('Cluster %d',ii+1);
    leg_cnt = leg_cnt + 1;
    end
end
cmap = jet(length(new_cluster.cluster_labels));
for ii = 1:length(new_cluster.cluster_labels)
    plot_gaussian_2d(new_cluster.gmm_xyMeans(ii,:)',squeeze(new_cluster.gmm_xySigma(:,:,ii)),[2],'r',2);
end
legend(h,leg_labels);
axis tight

%% check whether user wants to keep the new clustering
keep = input('Use new cluster (y/n)?','s');
if strcmpi(keep,'y')
    fixed = 2;
    [new_cluster,spike_features,spike_xy,Spikes] = apply_clustering(loadedData,new_cluster,[],fixed,Spikes); %recomputes cluster stats
    Clusters{probe_num} = new_cluster;
    fprintf('Saving cluster details\n');
    save(cur_clust_data,'Clusters');
    
    rclust_dat_name = [base_save_dir '/Ref_Clusters.mat'];
    load(rclust_dat_name,'RefClusters');
    if block_num == RefClusters{probe_num}.base_block %if this is already the ref cluster
        sum_fig = create_summary_cluster_fig(new_cluster,Spikes,spike_xy,new_cluster.params);
        pname = [init_save_dir sprintf('/Probe%d_Block%d_initclust',probe_num,block_num)];
        fillPage(gcf,'papersize',[14 8]);
        print(pname,'-dpng');
        
        fprintf('Saving to RefClusters\n');
        RefClusters{probe_num} = new_cluster;
        save(rclust_dat_name,'RefClusters');
    else %if its not the refcluster, check if we want to make it the new refcluster
        resp = input('Save to RefClusters (y/n)?','s');
        if strcmpi(resp,'y')
            fprintf('Saving to RefClusters\n');
            RefClusters{probe_num} = new_cluster;
            RefClusters{probe_num}.base_block = block_num;
            save(rclust_dat_name,'RefClusters');
            
            sum_fig = create_summary_cluster_fig(new_cluster,Spikes,spike_xy,new_cluster.params);
            fillPage(gcf,'papersize',[14 8]);
            pname = [init_save_dir sprintf('/Probe%d_Block%d_initclust',probe_num,block_num)];
            print(pname,'-dpng');
         close(sum_fig);
       end
    end
end

if ishandle(f1)
    close(f1);
end

