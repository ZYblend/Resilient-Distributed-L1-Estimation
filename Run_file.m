%% Run file for Multi-camera Perception
clear all
clc

%% Vehicle system Dynamics and controller
n_states = 3;  %[x,y,theta]
WheelBase=0.256;

z0 = [-1; -2; 0];  % initial state
constraint_steering =  pi/6;
constraint_velocity = [-0.5, 0.5];

load exampleMaps.mat;
startLoc = [5 5];
goalLoc = [12 3];

C = eye(2,n_states);


%% Camera system parameters
num_agents = 4;
n_meas = 3;

% camera model parameters (calibrated)
plane = 2.282825;   % height camera
dir = pwd;
% camParams = load(dir+"\nCamsParams.mat", "camParams");
% camParams = camParams.camParams;
extrinsics = load(dir+"\nCamsExtrinsics.mat", "camExtrinsics");
extrinsics = extrinsics.camExtrinsics; % R|t
% poses = load(dir+"\nCamsPoses.mat", "camPoses");
% poses = poses.camPoses;

% attack
num_attack = 2;
sparse_loc = [1,3];
e = zeros(n_meas*num_agents,1);
e(sparse_loc) = 2*rand(num_attack,1);

%% Centralized estimation
R_cent = zeros(num_agents*n_meas,n_states);
t_cent = zeros(num_agents*n_meas,1);
R_dist = cell(1,num_agents);
for cam=1:num_agents
    R_cent((cam-1)*n_meas+1:cam*n_meas,:) = extrinsics{cam}.R;
    R_dist{cam} = extrinsics{cam}.R;
    t_cent((cam-1)*n_meas+1:cam*n_meas,:) = extrinsics{cam}.Translation';
end

%% Distributed Perception parameters
% communication topology
Adj_mat = [0 1 0 0;
           1 0 0 1;
           0 0 0 1;
           0 1 1 0];
camGraph = graph(Adj_mat);
figure;
plot(camGraph);
title('Camera Graph')

% L = laplacian(camGraph); % laplacian matrix
L = [1  -1  0  0;
    -1   2  0  -1;
     0   0  1  -1;
     0  -1 -1  2];
in_degree = diag(L);
adj = kron(diag(diag(L))-L, eye(n_states));
L_bar = kron(L, eye(n_states));

disp('Has Spanning Tree?')
disp(num2str(eig(L)));

% solver
max_iter = 500;