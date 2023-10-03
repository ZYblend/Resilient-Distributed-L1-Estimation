function x_opt = L1_minimization(H, y, n_meas, n_states)
%% 
% Minimize ||y - H * x||_1
%
% solve by linear programming: Minimzie   1^T * t 
% %                          Subject to  -t <= y-H*x <=t
%
% Vectorization:
%               Minimize [1^T 0]*[t; x]
%             Subject to -t + Hx <=y
%                        -t - Hx <=-y
%
% Inptus:
%        H [n_meas-by-n_states]: observation matrix
%        y [n_meas-by-1]       : Measurement vector
% Output:
%        x_opt [n_states-by-1] : Estimated state
% 
% Author: Yu Zheng, yzheng6@fsu.edu 
% Florida State University, Tallahassee
% https://github.com/ZYblend/Resilient-Distributed-L1-Estimation.git
%
% © Copyright 2023 Yu Zheng.
%

f = [ones(1,n_meas), zeros(1,n_states)];
A = [-eye(n_meas)  H;
     -eye(n_meas) -H];
b = [y; -y];

% coder.extrinsic('linprog');
z = linprog(f,A,b);
x_opt = zeros(n_states, 1);
x_opt(:) = z(n_meas+1:size(z, 1));
end
