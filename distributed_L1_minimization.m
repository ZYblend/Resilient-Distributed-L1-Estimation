function [x_opt, x_store] = distributed_L1_minimization(H, y, L, n_states,num_agents, max_iter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% N agents, measurement modes:
%                              y_i = H_i*x + e_i
% where e_i is the sparse error on the measurements
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimator Design: 
%                   Minimize sum_i ||y_i - H_i * x||_1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Euqivalent distributed program:
%                   Minimize sum_i ||y_i - H_i * x_i||_1
%                 Subject to x_1 = x_2 = ... = x_N
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assume communication graph: L (laplacian matrix), 
%                             undirected graph, strongly connected
% Euqivalent program:
%                   Minimize sum_i ||y_i - H_i * x_i||_1
%                 Subject to L[x_1; x_2; ...; x_N] = 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solver: (Fully Distributed Lagrangian Dual Ascent)
%       for Iteration k+1, agent i:
%         x[i]^(k+1) = argmin_x [ f(y,x) + x^T * L[i]*mu + rho * sum_{j in i_neighbor}norm(x-x[j]^(k)) ]
%         mu[i]^(k+1) = mu[i]^(k) + rho * sum_{j in i_neighbor} (x_i^(k)-x_j^(k))
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vectorization
%        x^(k+1) = argmin_x [ sum_i ||y_i - H_i * x_i||_1 + x^T * L*mu + rho * L*norm(x-x[j]^(k)) ]
%        mu[i]^(k+1) = mu[i]^(k) + rho * sum_{j in i_neighbor} (x_i^(k)-x_j^(k))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inptus:
%        H [1-by-num_agents] cell: Cell of observation matrices H_i
%        y [1-by-num_agents] cell: Cell of Measurement vector y_i
%        n_states [scalar]       : dimension of state
%        num_agents [scalar]     : number of agents
%        max_iter [scalar]       : maximal number of iterations
%        L[num_agents-by-num_agents] : Laplacian matrix
%
% Output:
%        x_opt [num_agents*n_states-by-1] : Estimated states for all agents
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Yu Zheng, yzheng6@fsu.edu 
% Florida State University, Tallahassee
% https://github.com/ZYblend/Resilient-Distributed-L1-Estimation.git
%
% © Copyright 2023 Yu Zheng.
%    

%% Vecterization
N_states = num_agents*n_states;
x = zeros(N_states,1);
x_next = x;
x_store = zeros(max_iter,num_agents*n_states);

mu = rand(num_agents*n_states,1);
rho = 0.05;

in_degree = diag(L);
adj = kron(diag(diag(L))-L, eye(n_states));
L_bar = kron(L, eye(n_states));

for iter = 1:max_iter
    for i_agent = 1:num_agents
%         Xi = (L_bar((i_agent-1)*n_states+1:i_agent*n_states ,:).').*x;   % set the non-communicated agent's row to be zeros
%         fun = @(z) norm(y{i_agent}-H{i_agent}*z,1) + z.'*L_bar((i_agent-1)*n_states+1:i_agent*n_states ,:)*mu + rho*calc_regularization_term(i_agent,Xi,z);
        fun = @(z) norm(y{i_agent}-H{i_agent}*z,1) + z.'*L_bar((i_agent-1)*n_states+1:i_agent*n_states ,:)*mu + ...
                   (rho/2)*norm(in_degree(i_agent)*z-adj((i_agent-1)*n_states+1:i_agent*n_states ,:)*x,2)^2;
        x_next((i_agent-1)*n_states+1:i_agent*n_states) = fmincon(fun,x((i_agent-1)*n_states+1:i_agent*n_states),[],[]);
    end
    mu = mu + rho*L_bar*x;

    x_store(iter,:) = x;
    x = x_next;
%     if ( norm(x(1:n_states) - x(n_states+1:2*n_states)) < 0.001 ) ...
%                 && ( norm(x(1:n_states) - x(2*n_states+1:3*n_states)) < 0.001 )
%         disp('eary converge at iter');
%         disp(num2str(iter));
%         break;
%     end
end

x_opt = x;


% %% a support function to calculate the 2-norm regularization term
% function regu = calc_regularization_term(i,X,x)
%     regu = 0;
%     for idx = 1:size(X,1)
%         if(idx~=i)
%             if(any(X(idx,:)))
%                 regu = regu + norm(x-X(idx,:),2)^2;
%             end
%         end
%     end
% end


end