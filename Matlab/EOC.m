function  EOC(dataset_name)
% EOC: Effect of C
%--------------------------------------------------------------------------
% Input: 
%      dataset_name, name of the dataset, e.g. 'books_dvd'
%
% Output: a figure for the effect of parameter C on the dataset
%--------------------------------------------------------------------------

%load dataset
load(sprintf('data/%s',dataset_name));
[n,d]       = size(data);

%% set parameters: 
options.C      = 5;             % penalty parameter for PA-I 
options.Period      = 30;           % the number of one period for CDOL
options.lambda = 10;            % parameter lambda for ShiftPE
options.t_tick =  round(n/20);  %'t_tick'(step size for plotting figures)
options.sigma = 8;

%%  
% options.Number_old=n-m;
Y=data(1:n,1);
Y=full(Y);
X = data(1:n,2:d);


P = sum(X.*X,2);
P = full(P);
disp('Pre-computing kernel matrix...');
K = exp(-(repmat(P',n,1) + repmat(P,1,n)- 2*X*X')/(2*options.sigma^2));

Vector_C =2.^[-10:10];

for ix =1:length(Vector_C),
     options.C   = Vector_C(ix);
     %% run experiments:
     for i=1:size(ID_ALL,1),
         fprintf(1,'running on the %d-th trial...\n',i);
         ID = ID_ALL(i, :);      
    
     %1. PE
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = PE_K_M(Y,K,options,ID);
    nSV_PE(i) = length(classifier.SV);
    err_PE(i) = err_count;
    time_PE(i) = run_time;
    mistakes_list_PE(i,:) = mistakes;
    SVs_PE(i,:) = SVs;
    TMs_PE(i,:) = TMs;
    
    %2. PA-I
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = PA1_K_M(Y,K,options,ID);
    nSV_PA1(i) = length(classifier.SV);
    err_PA1(i) = err_count;
    time_PA1(i) = run_time;
    mistakes_list_PA1(i,:) = mistakes;
    SVs_PA1(i,:) = SVs;
    TMs_PA1(i,:) = TMs;
    
    %3. ShiftPE
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = ShiftPE_K_M(Y,K,options,ID);
    nSV_SPE(i) = length(classifier.SV);
    err_SPE(i) = err_count;
    time_SPE(i) = run_time;
    mistakes_list_SPE(i,:) = mistakes;
    SVs_SPE(i,:) = SVs;
    TMs_SPE(i,:) = TMs;
    
    %4. ModiPE
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = ModiPE_K_M(Y,K,options,ID);
    nSV_MPE(i) = length(classifier.SV);
    err_MPE(i) = err_count;
    time_MPE(i) = run_time;
    mistakes_list_MPE(i,:) = mistakes;
    SVs_MPE(i,:) = SVs;
    TMs_MPE(i,:) = TMs;
     
    
    %5. CDOLfix
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = CDOLfix_K_M(Y,K,options,ID);
    nSV_CF(i) = length(classifier.SV_v) + length(classifier.SV_w);
    err_CF(i) = err_count;
    time_CF(i) = run_time;
    mistakes_list_CF(i,:) = mistakes;
    SVs_CF(i,:) = SVs;
    TMs_CF(i,:) = TMs;
    
     %6. CDOL
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = CDOL_K_M(Y,K,options,ID);
    nSV_CD(i) = length(classifier.SV_v) + length(classifier.SV_w);
    err_CD(i) = err_count;
    time_CD(i) = run_time;
    mistakes_list_CD(i,:) = mistakes;
    SVs_CD(i,:) = SVs;
    TMs_CD(i,:) = TMs;
     end
     
     ERR_PE(ix)  = mean(err_PE)/n*100;
     ERR_PA1(ix) = mean(err_PA1)/n*100;
    ERR_SPE(ix) = mean(err_SPE)/n*100;
    ERR_MPE(ix)  = mean(err_MPE)/n*100;
    ERR_CF(ix) = mean(err_CF)/n*100;
    ERR_CD(ix)= mean(err_CD)/n*100;
     
end
 %% print and plot results
 mistakes_idx = [-10:10];
figure
plot(mistakes_idx, ERR_PE,'k-+');
hold on
plot(mistakes_idx, ERR_PA1,'g-*');
plot(mistakes_idx, ERR_MPE,'b.-');
plot(mistakes_idx, ERR_SPE,'m-d');
plot(mistakes_idx, ERR_CF,'r-x');
plot(mistakes_idx, ERR_CD,'r-o');
legend('PE','PA-I','ModiPE','ShiftPE','CDOL(fixed)','CDOL');
xlabel('log_2(C)');
ylabel('Average rate of mistakes')
grid
