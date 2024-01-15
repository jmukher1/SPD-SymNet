function [cov_all, Sum_CovMaps] = spd_deepmain_v2()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This method spd_deepmain_v2 checks all the intermediate 
% matrices are Symmetric Positive Definite (SPD) and returns the
% SPD matrices
% 1. Covariance matrix
% 2. Sum of Covariance Maps
% For saving the covariance (SPD) matrices, please run deepmain_v2.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%author: Rui Wang
%department: School of artificial intelligence and computer science, Jiangnan university 

%% load data

load FPHA_train_label
load FPHA_val_label
load FPHA_train_seq
load FPHA_val_seq

%% Preparation

Train_labels = train_labels;
Test_labels = val_labels;

cov_sum = zeros(63,63); 
cov_all = zeros(63,63);

num_layers_1 = 3; % $m_1^{v2}$
p_dim_1 = 20; % $d_{m1}^{v2}$

num_layers_2 = 4; % $m_2^{v2}$
p_dim_2 = 5; % $d_{m2}^{v2}$

rectified_data_cell = cell(1,size(Train_labels,2));
maps_sum = zeros(p_dim_1,p_dim_1);
Sum_CovMaps = zeros(p_dim_1,p_dim_1);

transfer_each = cell(1,num_layers_2);

second_layer_singletr = cell(1,num_layers_1); 
second_layer_singlete = cell(1,num_layers_1); 

final_train_branch = zeros(p_dim_2^2 * num_layers_2, num_layers_1);
final_test_branch = zeros(p_dim_2^2 * num_layers_2, num_layers_1);

% activation thresholds of the two rectifying layers of SymNet-v2
eps_1 = 4e-3; % $\epsilon_1^{v2}$
eps_2 = 1e-3; % $\epsilon_2^{v2}$

eta_1 = 1e-6;  % $\eta_1^{v2}$
eta_2 = 1e-6; % $\eta_2^{v2}$

%% training stage
Train_data = train_seq; % training samples
cov_train = computeCov(Train_data); % the computed training SPD matrices

try chol(cov_train);
    disp('Matrix cov_train is symmetric positive definite.')
catch ME
    disp('Matrix cov_train is not symmetric positive definite')
end

% the first SPD matrix mapping layer

for k = 1:size(Train_labels,2)
    cov_sum = cov_sum + cov_train{k};
end

mean_train = cov_sum / k;

for i = 1:size(Train_labels,2)
    cov_all = cov_all + (cov_train{i}-mean_train)'*(cov_train{i}-mean_train);
end

cov_all = cov_all/(size(Train_labels,2)-1);
try chol(cov_all);
    disp('Matrix cov_all is symmetric positive definite.')
catch ME
    disp('Matrix cov_all is not symmetric positive definite')
end
[e_vectors,e_values] = eig(cov_all);
[~,order] = sort(diag(-e_values));
e_vectors = e_vectors(:,order);

for i = 1:num_layers_1
    T_1{i} = e_vectors(:,(i-1)*p_dim_1+1:i*p_dim_1); % the connection weights of the first mapping layer
end

for i = 1:size(Train_labels,2)
    single_sample_tr = cov_train{i};
    for j = 1:num_layers_1
        second_layer_singletr{j} = T_1{j}'*single_sample_tr*T_1{j}; % first-stage (2D)^2PCA projection
    end
    
    % the first rectifying layer
    for k = 1:num_layers_1
        mid = second_layer_singletr{k};
        idx1 = mid <= 0;
        idx2 = mid > -eta_1;
        idx = idx1 & idx2;
        mid(idx) = -eta_1;
        [U,V,D] = svd(mid);
        [a,b]=size(V);
        tol_1 = trace(V)*eps_1;
        for l = 1:a
            if V(l,l) <= tol_1
                V(l,l) = tol_1;
            end
        end
        second_layer_singletr{k} = U*V*D';
    end
    
    rectified_data_cell{i} = second_layer_singletr;
    
    for s = 1:size(second_layer_singletr,2)
        rectified_maps_matrix(:,:,s) = second_layer_singletr{s};
    end
    
    m = size(second_layer_singletr,2);
    all_rectified_maps(:,:,(i-1)*m+1:i*m) = rectified_maps_matrix;
end

% the second SPD matrix mapping layer

for i = 1:size(all_rectified_maps,3)
    maps_sum = maps_sum + all_rectified_maps(:,:,i);
end

mean_maps = maps_sum / (num_layers_1*size(Train_labels,2));

for j = 1:size(all_rectified_maps,3)
    Sum_CovMaps = Sum_CovMaps + (all_rectified_maps(:,:,j)-mean_maps)'*(all_rectified_maps(:,:,j)-mean_maps);
end

Sum_CovMaps = Sum_CovMaps / (size(all_rectified_maps,3)-1);
try chol(Sum_CovMaps);
    disp('Matrix Sum_CovMaps is symmetric positive definite.')
catch ME
    disp('Matrix Sum_CovMaps is not symmetric positive definite')
end

[e_vectors,e_values] = eig(Sum_CovMaps);
eigenValuesSum_CovMaps = diag(e_values);
[dummy,order] = sort(diag(-e_values)); 
C_Weight = e_vectors(:,order); 

%% test stage

Test_data = val_seq; % test samples
cov_test = computeCov(Test_data); % the computed test SPD matrices 
try chol(cov_test);
    disp('Matrix cov_test is symmetric positive definite.')
catch ME
    disp('Matrix cov_test is not symmetric positive definite')
end

 
