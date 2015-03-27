function  Experiment_OTL_K_M(dataset_name)
% Experiment_OTL_K_M: the main function used to compare all the online
% algorithms
%==========================================================================
% Input:
%      dataset_name, name of the dataset, e.g. 'usenet1'
%
% Output:
%      a table containing the accuracies, the numbers of support vectors,
%      the running times of all the online learning algorithms on the
%      inputed datasets
%      a figure for the online average accuracies of all the online
%      learning algorithms
%      a figure for the online numbers of SVs of all the online learning
%      algorithms
%      a figure for the online running time of all the online learning
%      algorithms
%==========================================================================


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
 %% print and plot results
figure
mean_mistakes_PE = mean(mistakes_list_PE);
plot(mistakes_idx, mean_mistakes_PE,'k-+');
hold on
mean_mistakes_PA1 = mean(mistakes_list_PA1);
plot(mistakes_idx, mean_mistakes_PA1,'g-*');
mean_mistakes_MPE = mean(mistakes_list_MPE);
plot(mistakes_idx, mean_mistakes_MPE,'b.-');
mean_mistakes_SPE = mean(mistakes_list_SPE);
plot(mistakes_idx, mean_mistakes_SPE,'m-d');
mean_mistakes_CF = mean(mistakes_list_CF);
plot(mistakes_idx, mean_mistakes_CF,'r-x');
mean_mistakes_CD = mean(mistakes_list_CD);
plot(mistakes_idx, mean_mistakes_CD,'r-o');
legend('PE','PA-I','ModiPE','ShiftPE','CDOL(fixed)','CDOL');
xlabel('Number of samples');
ylabel('Online average rate of mistakes')
grid

figure
mean_SV_PE = mean(SVs_PE);
plot(mistakes_idx, mean_SV_PE,'k-+');
hold on
mean_SV_PA1 = mean(SVs_PA1);
plot(mistakes_idx, mean_SV_PA1,'g-*');
mean_SV_MPE = mean(SVs_MPE);
plot(mistakes_idx, mean_SV_MPE,'b.-');
mean_SV_SPE = mean(SVs_SPE);
plot(mistakes_idx, mean_SV_SPE,'m-d');
mean_SV_CF = mean(SVs_CF);
plot(mistakes_idx, mean_SV_CF,'r-x');
mean_SV_CD = mean(SVs_CD);
plot(mistakes_idx, mean_SV_CD,'r-o');
legend('PE','PA-I','ModiPE','ShiftPE','CDOL(fixed)','CDOL','Location','NorthWest');
xlabel('Number of samples');
ylabel('Online average number of support vectors')
grid

figure
mean_TM_PE = log(mean(TMs_PE))/log(10);
plot(mistakes_idx, mean_TM_PE,'k-+');
hold on
mean_TM_PA1 = log(mean(TMs_PA1))/log(10);
plot(mistakes_idx, mean_TM_PA1,'g-*');
mean_TM_MPE = log(mean(TMs_MPE))/log(10);
plot(mistakes_idx, mean_TM_MPE,'b.-');
mean_TM_SPE = log(mean(TMs_SPE))/log(10);
plot(mistakes_idx, mean_TM_SPE,'m-d');
mean_TM_CF = log(mean(TMs_CF))/log(10);
plot(mistakes_idx, mean_TM_CF,'r-x');
mean_TM_CD = log(mean(TMs_CD))/log(10);
plot(mistakes_idx, mean_TM_CD,'r-o');
legend('PE','PA-I','ModiPE','ShiftPE','CDOL(fixed)','CDOL','Location','NorthWest');
xlabel('Number of samples');
ylabel('average time cost (log_{10} t)')
grid


fprintf(1,'-------------------------------------------------------------------------------\n');
fprintf(1,'number of mistakes, size of support vectors, cpu running time\n');
fprintf(1,'PE:          %.4f \t %.4f \t  %.4f \t %.4f \t %.4f \t %.4f \n', mean(err_PE)*100/n,   std(err_PE)*100/n,  mean(nSV_PE),  std(nSV_PE),  mean(time_PE),   std(time_PE));
fprintf(1,'PA-I:        %.4f \t %.4f \t  %.4f \t %.4f \t %.4f \t %.4f \n', mean(err_PA1)*100/n,  std(err_PA1)*100/n, mean(nSV_PA1), std(nSV_PA1), mean(time_PA1),  std(time_PA1));
fprintf(1,'ShiftPE:     %.4f \t %.4f \t  %.4f \t %.4f \t %.4f \t %.4f \n', mean(err_SPE)*100/n,  std(err_SPE)*100/n, mean(nSV_SPE), std(nSV_SPE), mean(time_SPE),  std(time_SPE));
fprintf(1,'ModiPE:      %.4f \t %.4f \t  %.4f \t %.4f \t %.4f \t %.4f \n', mean(err_MPE)*100/n,  std(err_MPE)*100/n, mean(nSV_MPE), std(nSV_MPE), mean(time_MPE),  std(time_MPE));
fprintf(1,'CDOL(fixed): %.4f \t %.4f \t  %.4f \t %.4f \t %.4f \t %.4f \n', mean(err_CF)*100/n,   std(err_CF)*100/n,  mean(nSV_CF),  std(nSV_CF),  mean(time_CF), std(time_CF));
fprintf(1,'CDOL:        %.4f \t %.4f \t  %.4f \t %.4f \t %.4f \t %.4f \n', mean(err_CD)*100/n,   std(err_CD)*100/n,  mean(nSV_CD),  std(nSV_CD), mean(time_CD), std(time_CD));
fprintf(1,'-------------------------------------------------------------------------------\n');

