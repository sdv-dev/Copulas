% ======= densities ============================
clc
strMsg = sprintf('\n\nType, Theta, x0, x1, pdf, cdf');
disp(strMsg);
strMsg = sprintf('--------------------------------------------');
disp(strMsg);
x0 = [0.33, 0.47, 0.61];
x1 = [0.2, 0.33, 0.71, 0.9];
for k0=1:length(x0),
    for k1=1:length(x1),
        %--clayton--
        copulafamily = 'Clayton';
        theta = [0.7, 1.6, 3.4];
        for kt = 1:length(theta),
            pdf_value = copulapdf(copulafamily,[x0(k0) x1(k1)],theta(kt));
            cdf_value = copulacdf(copulafamily,[x0(k0) x1(k1)],theta(kt));
            strMsg = sprintf('%s, %1.1f, %2.2f, %2.2f, %2.5f, %2.5f',...
                copulafamily,theta(kt),x0(k0),x1(k1),pdf_value,cdf_value);
            disp(strMsg);
        end
        %--frank--
        copulafamily = 'Frank';
        theta = [-1.6, 0.7, 3.4];
        for kt = 1:length(theta),
            pdf_value = copulapdf(copulafamily,[x0(k0) x1(k1)],theta(kt));
            cdf_value = copulacdf(copulafamily,[x0(k0) x1(k1)],theta(kt));
            strMsg = sprintf('%s, %1.1f, %2.2f, %2.2f, %2.5f, %2.5f',...
                copulafamily,theta(kt),x0(k0),x1(k1),pdf_value,cdf_value);
            disp(strMsg);
        end
        %--gumbel--
        copulafamily = 'Gumbel';
        theta = [1.6, 3.4];
        for kt = 1:length(theta),
            pdf_value = copulapdf(copulafamily,[x0(k0) x1(k1)],theta(kt));
            cdf_value = copulacdf(copulafamily,[x0(k0) x1(k1)],theta(kt));
            strMsg = sprintf('%s, %1.1f, %2.2f, %2.2f, %2.5f, %2.5f',...
                copulafamily,theta(kt),x0(k0),x1(k1),pdf_value,cdf_value);
            disp(strMsg);
        end
    end
end
