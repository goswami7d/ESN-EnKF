%%%% ComputeCorrelationRCEnKFNumina.m %%%%%
%%% by D. Goswami, 2020 %%%
%%% Compute correlation between true and estimated traffic data with ESN-EnKF for different
%%% number of observable nodes.
%%% time-resolution is 15 minutes
%%% A pre-trained ESN is loaded.


clear; clc;
%%% Load the pre-trained ESN
load('TrainedRCNumina.mat');
dataStateSize=outSize;
psi=@(x)0.5*(1+tanh(x)); %activation function

parfor nodeindex=1:outSize
Hdiag=zeros(1,outSize);
for i=1:nodeindex
    Hdiag(i)=1;
end
H=diag(Hdiag);
h=@(z) H*z;
%H=[1 1 1];
EnsembleCovariance=640*eye(outSize);
ObservationCovariance=160*eye(outSize);
EnsembleSize=200;

for sampleindex=1:20
Y = zeros(outSize,testLen);
%u=1000*ones(1,outSize);
u = data(trainLen+1,:);
%u_ensemble=randgen(EnsembleSize,u,EnsembleCovariance);
u_ensemble=transpose(mvnrnd(u,EnsembleCovariance,EnsembleSize));
%u_ensemble(u_ensemble<0)=0;
%x = zeros(resSize,1);
xtemp=x;
for t = 1:testLen 
    ObservationTrue=H*data(trainLen + t +1,:)'...
        +transpose(mvnrnd(zeros(dataStateSize,1),ObservationCovariance));
    [y_estbar,y_est,xtemp]...
    =EnKFESN(h,u_ensemble,ObservationTrue,ObservationCovariance,EnsembleSize,xtemp,Win,W,Wout,a,psi);
    Y(:,t) = y_estbar;
    y_est(y_est<0)=0;
    Y(Y<0)=0;
    u_ensemble=y_est;
   
end
temp1=reshape(data(trainLen+2:trainLen+testLen+1,:),[1 testLen*outSize]);
temp2=reshape(Y',[1 testLen*outSize]);
R=corrcoef(temp1, temp2);
correlationMeasurement(sampleindex,nodeindex)=R(1,2);
end
end
correlationCovariance=diag(cov(correlationMeasurement));
correlationMean=mean(correlationMeasurement);

save('NuminaRCEnKFCorrelation.mat','correlationMeasurement', 'correlationMean', 'correlationCovariance');