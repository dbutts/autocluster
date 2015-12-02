function [] = autocluster_excludeSU(block_num,probe_num,exclude_SUs,clust_params)
% [] = autocluster_excludeSU(block_num,probe_num,exclude_SUs,<clust_params>)
% run autoclustering with spikes from specified cluster excluded
% INPUTS: 
%   block_num
%   probe_num
%   exclude_SU: number of SU (cluster > 1) to exclude
%   <clust_params>: additional parameter if desired

%%
global data_dir base_save_dir init_save_dir Expt_name Vloaded n_probes loadedData raw_block_nums

fprintf('Loading block %d Clusters\n',block_num);
cur_clust_data = [base_save_dir sprintf('/Block%d_Clusters.mat',block_num)];
load(cur_clust_data,'Clusters');

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
params = Clusters{probe_num}.params;

%load in any user-specified parameters
if nargin >= 4
    cur_fields = fieldnames(clust_params);
    for ii = 1:length(cur_fields)
        params = setfield(params,cur_fields{ii},getfield(clust_params,cur_fields{ii}));
    end
end
%% run new clustering without excluded spikes
%index values of spikes that belong to the excluded SU cluster
exclude_SU_inds = Clusters{probe_num}.spk_inds(ismember(Clusters{probe_num}.spike_clusts, exclude_SUs));

[clusterDetails,spike_features,sum_fig,Spikes] = detect_and_cluster_init_excludespks(loadedData,params,Clusters{probe_num}.use_chs,exclude_SU_inds);

%% check whether user wants to use the features identified with exclusion-based clustering
figure(sum_fig);
keep = input('Use these feature (y/n)?','s');
close(sum_fig);
if strcmpi(keep,'y')    
    orig_params = Clusters{probe_num}.params;
    params.outlier_thresh = orig_params.outlier_thresh; %use original outlier detection threshold
    fixed = 1;
    [~,spike_features,spike_xy,Spikes] = apply_clustering(loadedData,clusterDetails,params,fixed);
    
    n_comps = length(clusterDetails.cluster_labels) + length(exclude_SUs); %total number of components (assumes the excluded SU only had one comp)
    
    [GMM_obj, distance,all_comp_idx,all_clust_labels,cluster_stats] = ...
        GMM_fit(Spikes.V,spike_features,n_comps,params);
    newCluster = clusterDetails;
    newCluster.params = params;
    newCluster.gmm_fit = GMM_obj;
    newCluster.comp_idx = all_comp_idx;
    newCluster.cluster_labels = all_clust_labels;
    newCluster = compute_cluster_stats(newCluster,Spikes,spike_features);
        
    %make visible summary plot
    params.summary_plot = 2;
    sum_fig = create_summary_cluster_fig(newCluster,Spikes,spike_xy,params);
    
    %check whether user wants to keep the new clustering
    resp = input('Keep new clustering (y/n, d for cluster dump)?','s');
    if strcmpi(resp,'Y')
        fprintf('Saving cluster details\n');
            rclust_dat_name = [base_save_dir '/Ref_Clusters.mat'];
            load(rclust_dat_name);
        newCluster.base_block = RefClusters{probe_num}.base_block;
        Clusters{probe_num} = newCluster;
        save(cur_clust_data,'Clusters');
        
        if block_num == RefClusters{probe_num}.base_block
            fprintf('Saving to RefClusters\n');
            RefClusters{probe_num} = newCluster;
            save(rclust_dat_name,'RefClusters');
            
            fillPage(gcf,'papersize',[14 8]);
            pname = [init_save_dir sprintf('/Probe%d_Block%d_initclust',probe_num,block_num)];
            print(pname,'-dpng');
        else
            resp = input('Save to RefClusters (y/n)?','s');
            if strcmpi(resp,'y')
                fprintf('Saving to RefClusters\n');
                rclust_dat_name = [base_save_dir '/Ref_Clusters.mat'];
                load(rclust_dat_name);
                RefClusters{probe_num} = newCluster;
                RefClusters{probe_num}.base_block = block_num;
                save(rclust_dat_name,'RefClusters');
                
                fillPage(gcf,'papersize',[14 8]);
                pname = [init_save_dir sprintf('/Probe%d_Block%d_initclust',probe_num,block_num)];
                print(pname,'-dpng');
            end
        end
        close(sum_fig);
        
    elseif strcmpi(resp,'D')
        fprintf('Dumping cluster output for inspection\n');
        Cdump = newCluster;
        if ishandle(sum_fig)
            close(sum_fig);
        end
    else
        fprintf('Keeping original clustering\n');
        if ishandle(sum_fig)
            close(sum_fig);
        end
    end
end
