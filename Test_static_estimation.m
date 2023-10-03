%% Test file for distributed L1 optimization
clear all
clc

%% Parameters
num_agents = 3;
n_states   = 4;
[~,C1] = sysGen(8,n_states);
[~,C2] = sysGen(8,n_states);
[~,C3] = sysGen(10,n_states);
C = [C1; C2; C3];
n_meas = size(C,1);
n_meas1 = size(C1,1);
n_meas2 = size(C2,1);
n_meas3 = size(C3,1);

num_attack = 8;
sparse_loc = randperm(n_meas,num_attack);

%% Data
x = -5 + 10*rand(n_states,1);
e = zeros(n_meas,1);
e(sparse_loc) = 2*rand(num_attack,1);
y = C*x + e;

%% Centralized L1 optimization
% Minimize sum_i ||y_i - C_i * x||_1
% solve by linear programming: Minimzie   1^T * t 
% %                            Subject to -t <= y-C*x <=t
x_cent = L1_minimization(C, y, n_meas, n_states);

%% Local L1 optimization
x1 = L1_minimization(C1, y(1:n_meas1), n_meas1, n_states);
x2 = L1_minimization(C2, y(n_meas1+1:n_meas1+n_meas2), n_meas2, n_states);
x3 = L1_minimization(C3, y(n_meas1+n_meas2+1:end), n_meas3, n_states);

%% Distributed L1 optimization
% Minimize 
L = [1 -1 0;
     -1 2 -1;
     0 -1 1];

H = {C1,C2,C3};
y_dist = {y(1:n_meas1), y(n_meas1+1:n_meas1+n_meas2), y(n_meas1+n_meas2+1:end)};
max_iter = 500;   % increase it if your result does not achieve consensus
x0 = [x1; x2; x3];
[x_dist, x_store] = distributed_L1_minimization(H, y_dist, L, n_states,num_agents, max_iter);


disp('Has Spanning Tree?')
disp(num2str(eig(L)));
disp('Real x:');
disp(num2str(x));
disp('Centralized solved x:');
disp(num2str(x_cent));
disp('Locally solved x:');
disp(num2str([x1, x2, x3]));
disp('Distributed solved x:')
disp(num2str([x_dist(1:n_states), x_dist(n_states+1:2*n_states), x_dist(2*n_states+1:end)]))

subplot(2,2,1)
plot(x_store(:,1));
hold on, plot(x_store(:,5));
hold on, plot(x_store(:,9));
title('x_1')

subplot(2,2,2)
plot(x_store(:,2));
hold on, plot(x_store(:,6));
hold on, plot(x_store(:,10));
title('x_2')

subplot(2,2,3)
plot(x_store(:,3));
hold on, plot(x_store(:,7));
hold on, plot(x_store(:,11));
title('x_3')

subplot(2,2,4)
plot(x_store(:,4));
hold on, plot(x_store(:,8));
hold on, plot(x_store(:,12));
title('x_4')