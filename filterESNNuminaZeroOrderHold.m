function [thist,xhist,Phist,yhist] = filterESNNuminaZeroOrderHold(dataNodes, dataVals,ObserveddataVals, times,...
    ObserveddataNodes, timeLength, reservoirStates, a, W, Win, Wout,activation,EnsembleSize,...
    alpha, L, H, Q, R, x0, Pinit, decayFactor)

N = length(L);
M=size(H,1);
outSize=length(dataNodes);

% discrete time ODE transition matrix
dt=3*60;
F = decayFactor*expm(-alpha*L*dt);
thist(1)=0;
xhist(:,1) = x0;
Phist(:,1) = diag(Pinit);
yhist(:,1) = 0*dataVals(1,:)';

%%% Initial conditions
x=x0;
P=Pinit;

xESN=x0(dataNodes);
xESNEnsemble=transpose(mvnrnd(xESN',Pinit(dataNodes,dataNodes),EnsembleSize));
xESNEnsemble(xESNEnsemble<0)=0;
H_ESN=zeros(1,outSize); 
H_ESN(ObserveddataNodes)=1;
h=@(z) H_ESN*z;
for t=1:timeLength-1
 [xESN,xESNEnsemble,reservoirStates]=EnKFESN(h,xESNEnsemble,ObserveddataVals(t,:)',...
   R,EnsembleSize,reservoirStates,Win,W,Wout,a,activation);
   xESNEnsemble(xESNEnsemble<0)=0;
   xESN(xESN<0)=0;
 %    xEnsemble(dataNodes,:)=ESNUpdatedEnsemble;
     fprintf('Percent Complete %3.3f\n', t/(timeLength-1).*100);
    % do motion update only
    motionUpdateTimes = times(t):dt:times(t+1);
    Nsteps = length(motionUpdateTimes)-1; %ignore first and last
    simStep = length(thist);
   for k = simStep+1:1:simStep+Nsteps
    xForecast=F*x;
    xForecast(dataNodes,:)=xESN;
    yForecast=H*xForecast;
    P=F*P*F'+Q;
    S=H*P*H'+R;
    K=P*H'/S;
    y=ObserveddataVals(t,:)';
    x=xForecast+K*(y-yForecast);
    P=(eye(N,N)-K*H)*P;
    xhist(:,k)=x;
    Phist(:,k)=diag(P);
    yhist(:,k)= dataVals(t,:)'; 
    thist(k) = thist(k-1)+dt;
    
  end
    
end