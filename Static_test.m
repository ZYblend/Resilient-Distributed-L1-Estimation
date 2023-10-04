%% A static Test

clear all
clc

% location
dir = pwd;

%% parameters 
n_states = 3;
num_agents = 4;
n_meas = 3;

%% camera model parameters (calibrated)
plane = 2.282825;   % height camera
extrinsics = load(dir+"\data\nCamsExtrinsics.mat", "camExtrinsics");
extrinsics = extrinsics.camExtrinsics; % R|t

R_cent = zeros(num_agents*n_meas,n_states);
R_dist = cell(1,num_agents);
for cam=1:num_agents
    R_cent((cam-1)*n_meas+1:cam*n_meas,:) = extrinsics{cam}.R;
    R_dist{cam} = extrinsics{cam}.R;
end


%% attack
num_attack = 2;
sparse_loc = [1, 2];
e = zeros(n_meas*num_agents,1);
e(sparse_loc) = 2*rand(num_attack,1);

%% data
x = [-1; -2; plane];  % state
ys = R_cent*x +e;  % measurements (attacked)

% communication topology
Adj_mat = [0 1 0 0;
           1 0 0 1;
           0 0 0 1;
           0 1 1 0];
camGraph = graph(Adj_mat);
figure;
plot(camGraph);
title('Camera Graph')

L = [1  -1  0  0;
    -1   2  0  -1;
     0   0  1  -1;
     0  -1 -1  2];
in_degree = diag(L);
adj = kron(diag(diag(L))-L, eye(n_states));
L_bar = kron(L, eye(n_states));

%% Local L1 optimization
x1 = L1_minimization(R_dist{1}, ys(1:n_meas), n_meas, n_states);
x2 = L1_minimization(R_dist{2}, ys(n_meas+1:2*n_meas), n_meas, n_states);
x3 = L1_minimization(R_dist{3}, ys(2*n_meas+1:3*n_meas), n_meas, n_states);
x4 = L1_minimization(R_dist{4}, ys(3*n_meas+1:4*n_meas), n_meas, n_states);
x0 = [x1; x2; x3;x4];

% solver
max_iter = 500;
y  = cell(1,num_agents);
for i_agent = 1:num_agents
    y{i_agent} = ys(n_meas*(i_agent-1)+1:n_meas*i_agent);
end

X_opt = distributed_L1_minimization(R_dist, y, in_degree, L_bar, adj, n_states, num_agents, max_iter,x0);

disp('Real x:');
disp(num2str(x));
% disp('Centralized solved x:');
% disp(num2str(x_cent));
disp('Locally solved x:');
disp(num2str([x1, x2, x3, x4]));
disp('Distributed solved x:')
disp(num2str([X_opt(1:n_states), X_opt(n_states+1:2*n_states), X_opt(2*n_states+1:3*n_states), X_opt(3*n_states+1:4*n_states)]));