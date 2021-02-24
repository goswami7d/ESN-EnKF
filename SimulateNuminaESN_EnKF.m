%%%% SimulateNuminaESN_EnKF.m %%%%%
%%% by D. Goswami, 2020 %%%
%%% Train and test an ESN-EnKF to predict the traffic data obtained from Numina sensor
%%% time-resolution is 15 minutes


clear; clc;

% read the Numina CSV files
nd1 =readNuminaCSV('NuminaData/Numina-umd-umd-2-2019-12-02T00_00_00-2019-12-08T23_59_59_CampusPresidential.csv');
nd2 =readNuminaCSV('NuminaData/Numina-umd-umd-6-2019-12-02T00_00_00-2019-12-08T23_59_59_CampusPaint.csv');
nd3 =readNuminaCSV('NuminaData/Numina-umd-umd-9-2019-12-02T00_00_00-2019-12-08T23_59_59_RegentsStadium.csv');
nd4 =readNuminaCSV('NuminaData/Numina-umd-umd-4-2019-12-02T00_00_00-2019-12-08T23_59_59_SouthGate.csv');
nd5 =readNuminaCSV('NuminaData/Numina-umd-umd-1-2019-12-02T00_00_00-2019-12-08T23_59_59_UniversityPaint.csv');

trainLen = 1000;
testLen = 170;
initLen = 100;

% load the data
 %data=double([nd1.pedestrians, nd1.bicyclists, nd1.cars, nd1.buses, nd1.trucks]);
 data=double([nd1.cars, nd2.cars, nd3.cars, nd4.cars, nd5.cars ]);
 %data=double(nd1.cars);
dataStateSize=size(data,2);
resSize = 4000; %reservoir size
a = 0.7; % leaking rate
reg=1e-5; %regularization factor
psi=@(x)0.5*(1+tanh(x)); %activation function
%%% Create and train the ESN
[Wout, W, Win, x]=trainESN(data, data, psi, trainLen, initLen, resSize, a, reg);
fprintf('Training complete');

%% 
%x is initialized with training data and we continue from there.
%H=[1 1;0 0];
H=diag([0, 0, 0, 0, 1]);
h=@(z1) H*z1;
EnsembleCovariance=6400*eye(dataStateSize);
ObservationCovariance=1600*eye(dataStateSize);
EnsembleSize=200;
Y = zeros(dataStateSize,testLen);
u=0*ones(1,dataStateSize);
%u = data(trainLen+1,:);
u_ensemble=transpose(mvnrnd(u,EnsembleCovariance,EnsembleSize));
u_ensemble(u_ensemble<0)=0;
%x = zeros(resSize,1);
for t = 1:testLen 
    ObservationTrue=max(0,H*data(trainLen + t +1,:)'+transpose(mvnrnd(zeros(dataStateSize,1),ObservationCovariance)));
    [y_estbar,y_est,x]...
    =EnKFESN(h,u_ensemble,ObservationTrue,ObservationCovariance,EnsembleSize,x,Win,W,Wout,a,psi);
    Y(:,t) = y_estbar;
    y_est(y_est<0)=0;
    Y(Y<0)=0;
    u_ensemble=y_est;
end

errorLen = testLen;
%mse = sum(vecnorm(data(trainLen+2:trainLen+errorLen+1,:)'-Y(:,1:errorLen)))./errorLen;
mse = sqrt(immse(data(trainLen+2:trainLen+errorLen+1,:)',Y(:,1:errorLen)));
disp( ['MSE = ', num2str( mse )] );

%% Plotting the results
for i=1:dataStateSize
    
  subplot(dataStateSize,1,i)
  plot( data(trainLen+2:trainLen+testLen+1,i), 'color', [0,0.75,0],'linewidth',1 );
  hold on;
  plot( Y(i,:), 'b-.', 'linewidth',1);
  hold off;
  axis tight;
  xlabel('time', 'Interpreter', 'latex','FontSize',16);
  ylabel('$x$','Interpreter', 'latex','FontSize',16);
%title('Target and generated signals $x_1(n)$ starting at n=0','Interpreter', 'latex');
legend('True congestion', 'Estimated congestion','Interpreter','latex');
end

ymax=max(vecnorm(data(trainLen+2:trainLen+errorLen+1,:)'));
ymin=min(vecnorm(data(trainLen+2:trainLen+errorLen+1,:)'));

figure(2);
bar( Wout' )
title('Output weights $W^{out}$','Interpreter', 'latex');

figure(3);
plot( vecnorm(data(trainLen+2:trainLen+errorLen+1,:)'-Y(:,1:errorLen))/(ymax), 'b', 'linewidth',1);
hold off;
axis tight;
xlabel('time', 'Interpreter', 'latex');
ylabel('$L_2$ error','Interpreter', 'latex');
title('$L_2$ error in estimation','Interpreter', 'latex');

MSETimeSeries=vecnorm(data(trainLen+2:trainLen+errorLen+1,:)'-Y(:,1:errorLen))/(ymax-ymin);
save('RCEnKFNuminaAllNodesError.mat', 'MSETimeSeries');