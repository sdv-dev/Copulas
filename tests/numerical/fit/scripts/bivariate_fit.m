% ======= estimation ============================

clc
FLAG_copulaparam = true;  
CopulaFamily = {'Clayton','Frank','Gumbel'};
datasets = {'1','2'};
strMsg = sprintf('Dataset,        Copula, Theta, Tau');
disp(strMsg);
strMsg = sprintf('-----------------------------------');
disp(strMsg);
for d = 1:length(datasets)
    strFile = sprintf('datasets/%s.csv',datasets{d});
    T = readtable(strFile);
    T = table2array(T);
    tau = corr(T(:,1),T(:,2),'Type','Kendall');
    for k=1:length(CopulaFamily)
        if FLAG_copulaparam,
            theta = copulaparam(CopulaFamily{k},tau);
        else
            theta = copulafit(CopulaFamily{k},T);
        end
        strMsg = sprintf('%s, %s, %2.2f, %2.2f',strFile,CopulaFamily{k},theta,tau);
        disp(strMsg);
    end
end
