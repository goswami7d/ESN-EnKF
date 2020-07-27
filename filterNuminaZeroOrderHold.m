function [xhist,thist,Phist,yhist] = filterNuminaZeroOrderHold(dataNodes, dataVals, times, alpha, L, H, Q, R, Pinit, x0, dt, decayFactor)

N = length(L);

% discrete time ODE transition matrix
F = decayFactor*expm(-alpha*L*dt);

%
Ob = obsv(F,H);
if ( rank(Ob) == N)
    disp('System is observable');
else
    fprintf('System is NOT observable, rank( obsv(F,H) ) = %d, whereas N = %d \n',rank(Ob),N);
end
%
thist(1) = 0;
xhist(:,1) = x0;
Phist(:,1) = diag(Pinit);
yhist(:,1) = 0*dataVals(1,:)';

%% for initial update
xkm1_km1 = x0;
yk = dataVals(1,:)';
Pkm1_km1 = Pinit;
% compute time update
xk_km1 = F*xkm1_km1; % predicted state
yk_km1 = H*xk_km1; % predicted measurement
Pk_km1 = F*Pkm1_km1*F' + Q; % predicted covariance

% compute measurement update
S = H*Pk_km1*H' + R; % innovation
K = Pk_km1*H'/S; % Kalman gain
xk_k = xk_km1  + K*(yk - yk_km1); % posterior state stimation
%Pk_k = (eye(n,n)-K*H)*P_prior*(eye(n,n)-K*H)' + K*R*K'; % Joseph stabilized version
Pk_k = (eye(N,N)-K*H)*Pk_km1; % posterior covariance
thist(2) = 0;
xhist(:,2) = xk_k;
Phist(:,2) = diag(Pk_k);
yhist(:,2) = yk;
Pkm1_km1 = Pk_k;

%% for later updates
for j = 1:1:length(times)-1
    fprintf('Percent Complete %3.3f\n', j/(length(times)-1).*100);
    % do motion update only
    motionUpdateTimes = times(j):dt:times(j+1);
    Nsteps = length(motionUpdateTimes)-1; %ignore first and last
    simStep = length(thist);
    for k = simStep+1:1:simStep+Nsteps
        %         xhist(:,k)= F*xhist(:,k-1); % predicted state
        %         yk_km1 = H*xhist(:,k); % predicted measurement
        %         Phist{k} = F*Phist{k-1}*F' + Q; % predicted covariance
        %thist(k) = thist(k-1)+dt;
        %     end
        %     k = length(thist)+1;
        %     % do motion and measurement update
        %Pkm1_km1 = Phist{k-1};
        xk_km1 = F*xhist(:,k-1); % predicted state
        yk_km1 = H*xk_km1; % predicted measurement
        Pk_km1 = F*Pkm1_km1*F' + Q; % predicted covariance
        S = H*Pk_km1*H' + R; % innovation
        K = Pk_km1*H'/S; % Kalman gain
        yk = dataVals(j+1,:)';
        yhist(:,k) = yk;
        xhist(:,k) = xk_km1  + K*(yk - yk_km1); % posterior state stimation
        %Pk_k = (eye(n,n)-K*H)*P_prior*(eye(n,n)-K*H)' + K*R*K'; % Joseph stabilized version
        Pkm1_km1 = (eye(N,N)-K*H)*Pk_km1;
        Phist(:,k) = diag(Pkm1_km1); % posterior covariance
        thist(k) = thist(k-1)+dt;
    end
end

end