%%%% ComputeCorrelationNuminaESN.m %%%%%
%%% by D. Goswami, 2020 %%%
%%% Compute correlation between true and estimated traffic data with 
%%% reservoir observer for different number of observable nodes.
%%% time-resolution is 15 minutes
%%% A pre-trained ESN is loaded.


clear; clc;
%%% Load the pre-trained ESN
load('TrainedRCNumina.mat');
psi=@(x)0.5*(1+tanh(x)); %activation function


parfor nodeindex=1:outSize %number of nodes visible
    
for sampleindex=1:20 %sample for each Monte-Carlo
Y = zeros(outSize,testLen);
u = data(trainLen+1,:);
%u=10*ones(1,outSize);
%x = zeros(resSize,1);
xtemp=x;
for t = 1:testLen 
	[xtemp,y]=reservoirupdate(u',xtemp,Win,W,Wout,a,psi);
    y(y<0)=0;
	Y(:,t) = y;
	% generative mode:
	u = y';
    
	% this would be a predictive mode:
	%u = data(trainLen+t+1);
    
    % replace the observed nodes with actual data
    for i=1:nodeindex
        u(i)=data(trainLen+t+1,i);
    end
    
end
Y(Y<0)=0;

temp1=reshape(data(trainLen+2:trainLen+testLen+1,:),[1 testLen*outSize]);
temp2=reshape(Y',[1 testLen*outSize]);
R=corrcoef(temp1, temp2);
correlationMeasurementPredictive(sampleindex,nodeindex)=R(1,2);
end
end
correlationCovariancePredictive=diag(cov(correlationMeasurementPredictive));
correlationMeanPredictive=mean(correlationMeasurementPredictive);

save('NuminaRCCorrelationPredictive.mat','correlationMeasurementPredictive', ...
    'correlationMeanPredictive', 'correlationCovariancePredictive');