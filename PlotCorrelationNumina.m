clear; clc;
load('NuminaRCEnKFCorrelation.mat');
load('NuminaRCCorrelationPredictive.mat');
correlationComparison=[abs(correlationMean(:)),abs(correlationMeanPredictive(:))];
correlationCovarianceComparison=[correlationCovariance(:),correlationCovariancePredictive(:)];
b1=bar(correlationComparison,'grouped','BarWidth', 2);
hold on;
ngroups = 5;
nbars = 2;
% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, abs(correlationComparison(:,i)), correlationCovarianceComparison(:,i), '.k');
end
%errorbar(abs(correlationMeanPredictive), correlationCovariancePredictive','LineStyle','none');
%errorbar(abs(correlationMean), correlationCovariance','LineStyle','none');
axis([0.5 5.5 0 1]);
xlabel('number of available sensors', 'Interpreter', 'latex','FontSize',18);
ylabel('average correlation','Interpreter', 'latex','FontSize',18);
legend(flip(b1),'ESN (Reservoir Observer)','ESN-EnKF','Interpreter','latex',...
    'FontSize',18,'FontName','Times New Roman');