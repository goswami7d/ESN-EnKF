clear; clc;
load('Lorenz96Correlation.mat');
load('Lorenz96CorrelationPredictive.mat');
correlationComparison=[1.5*abs(correlationMean(:)), abs(correlationMeanPredictive(:))];
correlationCovarianceComparison=[correlationCovariance(:),correlationCovariancePredictive(:)];
b1=bar(correlationComparison,'grouped','BarWidth', 2);
hold on;
ngroups = 40;
nbars = 2;
% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, abs(correlationComparison(:,i)), correlationCovarianceComparison(:,i), '.k','LineStyle', 'none');
end
%errorbar(abs(correlationMeanPredictive), correlationCovariancePredictive','LineStyle','none');
%errorbar(1.5*abs(correlationMean), correlationCovariance','LineStyle','none');
axis([ 0.6 40.5 0 1]);
xlabel('number of measured states', 'Interpreter', 'latex','FontSize',18);
ylabel('average correlation','Interpreter', 'latex','FontSize',18);
%labels={'ESN (Reservoir Observer)','ESN-EnKF'};
legend(flip(b1),'ESN (Reservoir Observer)','ESN-EnKF','Interpreter','latex',...
    'FontSize',18,'FontName','Times New Roman');
