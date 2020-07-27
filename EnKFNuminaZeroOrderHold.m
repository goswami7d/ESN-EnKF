function [thist,xhist,Phist,yhist] = EnKFNuminaZeroOrderHold(dataNodes, dataVals, times,...
    ObserveddataNode, timeLength, reservoirStates, a, W, Win, Wout,...
    activation, alpha, L, H, ObservationCovariance, x0Ensemble, EnsembleSize, EnsembleCovariance, decayFactor)

N = length(L);
M=size(H,1);
outSize=length(dataNodes);

% discrete time ODE transition matrix
dt=3*60;
F = decayFactor*expm(-alpha*L*dt);
xEnsemble=x0Ensemble;
thist(1)=0;
xhist(:,1) = max(0,mean(x0Ensemble,2));
Phist(:,1) = diag(EnsembleCovariance);
yhist(:,1) = 0*dataVals(1,:)';
% xhist=zeros(N,timeLength);
% Phist=zeros(N,timeLength);
% yhist=zeros(M,timeLength);
ESNUpdatedEnsemble=xEnsemble(dataNodes,:);
diagH_ESN=zeros(1,outSize); 
diagH_ESN(ObserveddataNode)=1;
H_ESN=diag(diagH_ESN);
h=@(z) H_ESN*z;
for t=1:timeLength-1
 %[~,ESNUpdatedEnsemble,reservoirStates]=EnKFESN(h,ESNUpdatedEnsemble,dataVals(t,ObserveddataNode)',...
 %   ObservationCovariance*eye(outSize),EnsembleSize,reservoirStates,Win,W,Wout,a,activation);
 %    xEnsemble(dataNodes,:)=ESNUpdatedEnsemble;
     fprintf('Percent Complete %3.3f\n', t/(timeLength-1).*100);
    % do motion update only
    motionUpdateTimes = times(t):dt:times(t+1);
    Nsteps = length(motionUpdateTimes)-1; %ignore first and last
    simStep = length(thist);
   for k= simStep+1:1:simStep+Nsteps
    xEnsembleForecast=F*xEnsemble;
    yEnsembleForecast=H*xEnsembleForecast;
    xEnsembleMean=mean(xEnsembleForecast,2);
    yEnsembleMean=mean(yEnsembleForecast,2);
    
    for j=1:N
        Ex(j,:)=xEnsembleForecast(j,:)-xEnsembleMean(j);
    end
    
%     for j=1:M
%         Ey(j,:)=yEnsembleForecast(j,:)-yEnsembleMean(j);
%     end
    
   %Pxy=Ex*Ey'/(EnsembleSize-1);
   Pxx=Ex*Ex'/(EnsembleSize-1);
   Pxy=Pxx*H';
   Pyy=H*Pxx*H' + ObservationCovariance;
   %Pyy=Ey*Ey'/(EnsembleSize-1)+ObservationCovariance;
   K=Pxy*inv(Pyy);
   y=dataVals(t,ObserveddataNode)';
   xEnsembleForecastTemp=max(0,xEnsembleForecast);
   yEnsembleForecastTemp=max(0,yEnsembleForecast<0);
   xEnsemble=xEnsembleForecast+K*(y-yEnsembleForecast);
   xEnsembleTemp=xEnsembleForecastTemp+K*(y-yEnsembleForecastTemp);
   xEnsemble(xEnsemble<0)=0;
   xhist(:,k)=max(0,mean(xEnsembleTemp,2));
   Phist(:,k)=diag((eye(N,N)-K*H)*Pxx);
   yhist(:,k)=y;
   thist(k) = thist(k-1)+dt;
  end
    
end

