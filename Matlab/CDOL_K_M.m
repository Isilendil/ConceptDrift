function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = CDOL_K_M(Y, K, options, id_list)
% CDOL: Concept Drift Online Learning Algorithm
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
C = options.C; % 1 by default
P = options.Period;
t_tick = options.t_tick;
ID = id_list;

err_count = 0;
mistakes = [];
mistakes_idx = [];
SVs = [];
TMs=[];

alpha_v = [];
SV_v = [];

alpha_w = [];
SV_w = [];


a_1t = 0;
a_2t = 1;
eta  = 1/2;

alpha_u1 = [];
SV_u1 = [];

alpha_u2 = [];
SV_u2 = [];

err_u1 = 0;
err_u2 = 0;


%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    y_t = Y(id);
    
 
        if (isempty(alpha_u1))
            f_u1=0;
        else
            k_u1 = K(id, SV_u1(:))';
            f_u1 = alpha_u1*k_u1;
        end
        
        hat_y_u1 = sign(f_u1);
        if hat_y_u1 ==0,
            hat_y_u1=1;
        end
        if hat_y_u1~=y_t,
            err_u1 = err_u1 +1;
        end
        
        l_u1 = max(0, 1-y_t*f_u1);
        s_t = K(id,id);
        
        if (l_u1>0) && (s_t~=0),
            tau_u1 = min(C, l_u1/s_t);
            alpha_u1 = [alpha_u1 tau_u1*y_t];
            SV_u1 = [SV_u1, id];
        end
        
        if (isempty(alpha_u2))
            f_u2=0;
        else
            k_u2 = K(id, SV_u2(:))';
            f_u2 = alpha_u2*k_u2;
        end
        
        hat_y_u2 = sign(f_u2);
        if hat_y_u2 ==0,
            hat_y_u2=1;
        end
        if hat_y_u2~=y_t,
            err_u2 = err_u2 +1;
        end
        
        l_u2 = max(0, 1-y_t*f_u2);
        s_t = K(id,id);
        
        if (l_u2>0) && (s_t~=0),
            tau_u2 = min(C, l_u2/s_t);
            alpha_u2 = [alpha_u2 tau_u2*y_t];
            SV_u2 = [SV_u2, id];
        end
        
    
    if (isempty(alpha_v)),
        fv =0;
    else
        k_v=K(id,SV_v(:))';
        fv = alpha_v*k_v;
    end
    
    if (isempty(alpha_w)),
        fw = 0;
    else
        k_w = K(id, SV_w(:))';
        fw = alpha_w*k_w;
    end
    
    hat_fv = max(0, min(1, (fv+1)/2));
    hat_fw = max(0, min(1, (fw+1)/2));
    f_t = a_1t*hat_fv + a_2t*hat_fw -1/2;
    hat_y_t = sign(f_t);
    if hat_y_t ==0,
        hat_y_t =1;
    end
    
    if hat_y_t~=y_t,
        err_count = err_count +1;
    end
    
    lw = max(0, 1-y_t*fw);
    if (lw>0)&&(s_t~=0),
        gamma_t =min(C, lw/s_t);
        alpha_w = [alpha_w, gamma_t*y_t];
        SV_w = [SV_w, id];
    end
    
    Pi_y = (y_t+1)/2;
    ell_1 = (hat_fv-Pi_y)^2;
    ell_2 = (hat_fw-Pi_y)^2;
    
    a_1t = a_1t*exp(-eta*ell_1);
    a_2t = a_2t*exp(-eta*ell_2);
    sum_a = a_1t + a_2t;
    a_1t = a_1t/sum_a;
    a_2t = a_2t/sum_a;
    
    if mod(t,P)==0,
        if err_u1>err_u2,
            if a_2t>a_1t,
                alpha_v=alpha_w;
                SV_v = SV_w;
            end
            alpha_w = [];
            SV_w = [];
            a_1t = 1/2;
            a_2t = 1/2;
        end
        err_u1 = 0;
        err_u2 = 0;
        alpha_u1=alpha_u2;
        SV_u1 = SV_u2;
        alpha_u2 = [];
        SV_u2 = [];
    end
    
    run_time=toc;
    if (mod(t,t_tick)==0)
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs length(SV_w)+length(SV_v)];
        TMs=[TMs run_time];
    end
end
classifier.SV_v = SV_v;
classifier.alpha_v = alpha_v;
classifier.SV_w = SV_w;
classifier.alpha_w = alpha_w;

fprintf(1,'The number of mistakes = %d\n', err_count);
run_time = toc;
