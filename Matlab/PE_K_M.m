function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = PE_K_M(Y, K, options, id_list)
% PE:online Perceptron algorithm
%--------------------------------------------------------------------------
% Input:
%        Y:    the vector of lables
%        K:    precomputed kernel for all the example, i.e., K_{ij}=K(x_i,x_j)
%  id_list:    a randomized ID list
%  options:    a struct containing rho, sigma, C, n_label and n_tick;
% Output:
%   err_count:  total number of training errors
%    run_time:  time consumed by this algorithm once
%    mistakes:  a vector of mistake rate 
% mistake_idx:  a vector of number, in which every number corresponds to a
%               mistake rate in the vector above
%         SVs:  a vector records the number of support vectors 
%     size_SV:  the size of final support set
%--------------------------------------------------------------------------

%% initialize parameters
t_tick = options.t_tick;
alpha = [];
SV = [];
ID = id_list;
err_count = 0;
mistakes = [];
mistakes_idx = [];
SVs = [];
TMs=[];
%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    if (isempty(alpha)), % init stage
        f_t = 0; 
    else
        k_t = K(id,SV(:))';
        f_t = alpha*k_t;            % decision function
    end

    hat_y_t = sign(f_t);        % prediction
    if (hat_y_t==0)
        hat_y_t=1;
    end
    % count accumulative mistakes
    if (hat_y_t~=Y(id)),
        err_count = err_count + 1;
    end
    
    if (hat_y_t~=Y(id)), %update
        alpha = [alpha Y(id)];
        SV = [SV id];
    end
    run_time=toc;
    if (mod(t,t_tick)==0)
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs length(SV)];
        TMs=[TMs run_time];
    end
end
classifier.SV = SV;
classifier.alpha = alpha;
fprintf(1,'The number of mistakes = %d\n', err_count);
run_time = toc;
