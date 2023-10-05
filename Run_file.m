%% Run file for Multi-camera Perception
clear all
clc

% location
dir = pwd;

%% Vehicle system Dynamics and controller
n_states = 3;  %[x,y,theta]
WheelBase=0.256; 

% motion parameters
constraint_steering =  pi/6;
constraint_velocity = [-0.5, 0.5];
vd = 0.1;

% load map
load(dir+"/data/exampleMaps.mat");
startLoc = [5 5];
goalLoc = [12 3];

C = eye(2,n_states);
C_block = blkdiag(C,C,C,C);
A = @(v,theta) [0 0 -v*sin(theta);
                0 0  v*cos(theta);
                0 0 0];
disp('fully observable?')
disp( num2str(rank(obsv(A(0.1,pi/6),C))) );

%% Camera system parameters
num_agents = 4;
n_meas = 3;

% camera model parameters (calibrated)
plane = 2.282825;   % height camera
% camParams = load(dir+"\nCamsParams.mat", "camParams");
% camParams = camParams.camParams;
extrinsics = load(dir+"\data\nCamsExtrinsics.mat", "camExtrinsics");
extrinsics = extrinsics.camExtrinsics; % R|t
% poses = load(dir+"\nCamsPoses.mat", "camPoses");
% poses = poses.camPoses;

% attack
num_attack = 2;
sparse_loc = [1,2];
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

disp('Connected Graph?')
disp(num2str(eig(L)));

% distributed L1 solver
max_iter = 500;

% dynamical observer
A_obsv = kron(eye(num_agents),A(0.1,pi/6));
poles = [-1.5 -2.5 -2];
L_obsv = place(A(0.1,pi/6).',C.',poles).';
L_obsv_blk = kron(eye(num_agents),L_obsv);

z0 = [1;0;0;
      1;2;0;
      1;4;0;
      1;6;0];  % initialization of estiamte for each camera