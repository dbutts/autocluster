clear all
close all
addpath('~/autocluster/'); 

global data_dir base_save_dir init_save_dir Expt_name Vloaded n_probes loadedData raw_block_nums

data_loc = '/media/NTlab_data3/Data/bruce/'; %directory containing raw data
Expt_name = 'M012'; %short name of experiment
monk_name = 'jbe'; %(lem and jbe are the two monkeys so far)
rec_type = 'LP'; %type of recording (LP for laminar probe, or UA for Utah array)
rec_number = 1;

%directory for saving clustering output
base_save_dir = ['~/Analysis/bruce/' Expt_name '/clustering'];
if rec_number > 1 %if you're splitting the recording into multiple separate chunks for clustering
   base_save_dir = [base_save_dir sprintf('/rec%d',rec_number)]; 
end
init_save_dir = [base_save_dir '/init']; %subdir for initial cluster fits
if ~exist(base_save_dir,'dir');
    mkdir(base_save_dir);
end
if ~exist(init_save_dir,'dir');
    mkdir(init_save_dir);
end

%location of FullV files
data_dir = [data_loc Expt_name];

expt_file_loc = ['/media/NTlab_data3/Data/bruce/' Expt_name]; %location of Expts.mat files (sometimes not in same raw-data dir)

Vloaded = nan; %keeps track of which raw data file is currently loaded (Nan means none loaded)

%% SET CLUSTERING PARAMETERS
clear clust_params
clust_params.gmm_inits = 100; %number of GMM random inits
clust_params.min_Pcomp = 0.005; %minimum component probability
clust_params.use_ms = 0; %mean-subtraction on templates
clust_params.try_features = [1 2 4]; %features to use [PCs Voltage Energy Templates]
clust_params.max_back_comps = 2; %maximum N comps for modeling background
clust_params.cluster_bias = 0.5; %minimum posterior prob to assign to non-background cluster
clust_params.target_rate = 50; %target rate for spike detection

%% PICK BLOCKS TO RUN INITIAL CLUSTER FITTING ON
%set number of probes based on recording type
if strcmp(rec_type,'UA')
    n_probes = 96;
elseif strcmp(rec_type,'LP')
    n_probes = 24;
end

%LOOK AT DURATION OF EACH EXPERIMENT BLOCK AS A WAY TO PICK A SET OF BASE BLOCKS FOR INITIAL CLUSTERING
load(sprintf('%s/%s%sExpts.mat',expt_file_loc,monk_name,Expt_name));
n_blocks = length(Expts);
[block_durs,ed] = deal(nan(n_blocks,1));
for ii = 1:n_blocks
    if ~isempty(Expts{ii})
        trial_durs = [Expts{ii}.Trials(:).dur];
        expt_durs(ii) = (Expts{ii}.Header.End - Expts{ii}.Header.Start)/1e4; %duration of recording block
        sum_trial_durs(ii) = nansum(trial_durs)/1e4; %summed duration of trials
        ed(ii) = Expts{ii}.Stimvals.ed;
    end
end

%make figure showing recording duration and electrode depth (if applicable) across recording blocks
figure;
if strcmp(rec_type,'LP'); subplot(2,1,1); end;
plot(1:n_blocks,expt_durs,'o-');
hold on
plot(1:n_blocks,sum_trial_durs,'ro-');
xlabel('Block number');
ylabel('Duration (s)');
legend('Recording dur','Total trial dur');
xlim([0 n_blocks+1])
if strcmp(rec_type,'LP');
    subplot(2,1,2);
    plot(1:n_blocks,ed,'o-');
    xlabel('Block number');
    ylabel('Electrode depth (mm)');
xlim([0 n_blocks+1])
end

%get set of blocks to run initial cluster fitting on
poss_base_blocks = input('Enter vector of block numbers to fit initial models on\n');
assert(all(ismember(poss_base_blocks,1:n_blocks)),'inputs must be possible recording blocks');
fprintf('Fitting initial models on blocks %s\n',sprintf('%d ',poss_base_blocks));
target_probes = 1:n_probes;
if isfield(Expts{1}.Header,'exptno')
raw_block_nums = cellfun(@(X) X.Header.exptno,Expts,'uniformoutput',1); %block numbering for EM/LFP data sometimes isnt aligned with Expts struct
else
    raw_block_nums = 1:n_blocks;
end
%% PERFORM INITIAL CLUSTERING
for bb = 1:length(poss_base_blocks) %loop over initial set of blocks
    cur_base_block = poss_base_blocks(bb);
    cur_raw_block = raw_block_nums(cur_base_block);
    
    full_dat_name = [base_save_dir sprintf('/Block%d_Clusters.mat',cur_base_block)];
    second_dat_name = [base_save_dir sprintf('/Block%d_initClusters.mat',cur_base_block)];
    if exist(full_dat_name,'file');
        fprintf('Loading clusters for block %d\n',cur_base_block);
        load(full_dat_name,'Clusters');
    else
        fprintf('Initiating new cluster block\n');
        Clusters = cell(n_probes,1);
    end
    
    all_probe_fig = figure('visible','off');
    n_cols = ceil(sqrt(n_probes)); n_rows = ceil(n_probes/n_cols);
    for probe_num = target_probes 
       fprintf('\nClustering probe %d of %d\n',probe_num,n_probes);
        
        if Expt_name(1) == 'G' %for UTAH array data Load in Voltage signasl for each probe
            loadedData = [data_dir sprintf('/Expt%d.p%dFullV.mat',cur_raw_block,probe_num)];
            use_chs = [];
        else %for Laminar probe data load in all voltage signals for a given block
            sfile_name = [data_dir sprintf('/Expt%dFullV.mat',cur_raw_block)];
            use_chs = [probe_num-1 probe_num probe_num + 1];
            use_chs(use_chs < 1 | use_chs > n_probes) = [];
%             use_chs = probe_num;
            if Vloaded ~= cur_raw_block
                fprintf('Loading data file %s\n',sfile_name);
                [loadedData.V,loadedData.Vtime,loadedData.Fs] = Load_FullV(sfile_name, false, [100 nan],1:n_probes);
                Vloaded = cur_raw_block;
            end
        end
        
        %DETECT SPIKES AND PERFORM INITIAL CLUSTERING (2 CLUSTERS)
        [cluster_details,spike_features,sum_fig] = detect_and_cluster_init(loadedData,clust_params,use_chs);
        cluster_details.base_block = cur_base_block;
        Clusters{probe_num} = cluster_details;
        fprintf('Saving cluster details\n');
        save(full_dat_name,'Clusters');
        save(second_dat_name,'Clusters');
        
        %save unit cluster plot
        fillPage(sum_fig,'papersize',[14 8]);
        pname = [init_save_dir sprintf('/Probe%d_Block%d_initclust',probe_num,cur_base_block)];
        print(sum_fig,pname,'-dpng');
        close(sum_fig);

        %add to all probe plot
        spike_xy = spike_features*cluster_details.xy_projmat;
        N_spks = size(spike_xy,1);
        su_inds = find(cluster_details.spike_clusts == 2);
        mu_inds = find(cluster_details.spike_clusts == 1);        
        set(0,'CurrentFigure',all_probe_fig);
        subplot(n_cols,n_rows,probe_num);hold on
        plot(spike_xy(mu_inds,1),spike_xy(mu_inds,2),'k.','markersize',1);
        plot(spike_xy(su_inds,1),spike_xy(su_inds,2),'r.','markersize',1);
        set(gca,'xtick',[],'ytick',[]);axis tight
        for ii = 1:length(cluster_details.cluster_labels)
            h1 = plot_gaussian_2d(cluster_details.gmm_xyMeans(ii,:)',squeeze(cluster_details.gmm_xySigma(:,:,ii)),[2],'b',1);
        end
        title(sprintf('P%d',probe_num),'color','r');
        
    end
    
    fillPage(all_probe_fig,'papersize',[14 14]);
    pname = [base_save_dir sprintf('/Allprobe_Block%d_scatter',cur_base_block)];
    print(all_probe_fig,pname,'-dpng');
    close(all_probe_fig);
    
%     figure
%     subplot(2,1,1)
%     plot(cellfun(@(x) x.dprime,Clusters),'o-')
%     ylabel('Dprime','fontsize',12);
%     subplot(2,1,2)
%     plot(cellfun(@(x) x.Lratios,Clusters),'o-')
%     ylabel('Lratio','fontsize',12);
%     fillPage(gcf,'papersize',[5 8]);
%     pname = [base_save_dir sprintf('/Allprobe_Block%d_quality',cur_base_block)];
%     print(pname,'-dpng');
%     close(gcf);
    
    block_all_dprimes(bb,:) = cellfun(@(x) x.dprime,Clusters);
    
end

%% INITIALIZE REF CLUSTERS
fprintf('Saving REFCLUSTERS details\n');
[best_dprimes,best_blocks] = nanmax(block_all_dprimes,[],1);
%NOTE, you can manually change the reference block used for each probe here

target_base_block = poss_base_blocks(1);
dat_name = [base_save_dir sprintf('/Block%d_Clusters.mat',target_base_block)];
load(dat_name,'Clusters');
RefClusters = Clusters;
rclust_dat_name = [base_save_dir '/Ref_Clusters.mat'];
save(rclust_dat_name,'RefClusters');

for bb = 1:length(poss_base_blocks)
    target_base_block = poss_base_blocks(bb);
    dat_name = [base_save_dir sprintf('/Block%d_Clusters.mat',target_base_block)];
    load(dat_name,'Clusters');
    
    cur_probe_set = find(best_blocks == bb);
    RefClusters(cur_probe_set) = Clusters(cur_probe_set);
    save(rclust_dat_name,'RefClusters');
end
