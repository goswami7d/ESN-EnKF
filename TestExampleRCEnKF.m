clear;clc;

trainLen = 1000;
testLen = 200;
initLen = 200;

%%% Load the data from the file
%data = load('MackeyGlass_t17.txt');
load('dataLorenz.mat','Xf');
data=Xf;


dataStateSize=size(data,2);
a=0.7; %leakage rate
resSize=2000; %reservoir size
reg=1e-5; %regularization factor
psi1=@(x)tanh(x); %activation function
%%% Create and train the ESN
[Wout, W, Win, x]=trainESN(data, data, psi1, trainLen, initLen, resSize, a, reg);

%% 
%x is initialized with training data and we continue from there.
%H=[1 1;0 0];
H=[0 0 0;0 1 0;0 0 0];
h=@(z1) H*z1;
EnsembleCovariance=0.01*eye(dataStateSize);
ObservationCovariance=0.01*eye(dataStateSize);
EnsembleSize=100;
Y = zeros(dataStateSize,testLen);
u=30*ones(1,dataStateSize);
%u = data(trainLen+1,:);
%u_ensemble=randgen(EnsembleSize,u,EnsembleCovariance);
u_ensemble=transpose(mvnrnd(u,EnsembleCovariance,EnsembleSize));
%x = zeros(resSize,1);
for t = 1:testLen 
    ObservationTrue=H*data(trainLen + t +1,:)'+transpose(mvnrnd(zeros(dataStateSize,1),ObservationCovariance));
    [y_estbar,y_est,x]...
    =EnKFESN(h,u_ensemble,ObservationTrue,ObservationCovariance,EnsembleSize,x,Win,W,Wout,a,psi1);
    Y(:,t) = y_estbar;
    u_ensemble=y_est;
end

errorLen = testLen;
mse = sum(vecnorm(data(trainLen+2:trainLen+errorLen+1,:)'-Y(:,1:errorLen)))./errorLen;
disp( ['MSE = ', num2str( mse )] );

for i=1:dataStateSize
    
  subplot(dataStateSize,1,i)
  plot( data(trainLen+2:trainLen+testLen+1,i), 'color', [0,0.75,0],'linewidth',1 );
  hold on;
  plot( Y(i,:), 'b-.', 'linewidth',1);
  hold off;
  axis tight;
  xlabel('time', 'Interpreter', 'latex');
  ylabel('$x_1$','Interpreter', 'latex');
%title('Target and generated signals $x_1(n)$ starting at n=0','Interpreter', 'latex');
 legend('Target signal', 'Predicted signal','Interpreter','latex','FontSize',18);

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

MSETimeSeries=vecnorm(data(trainLen+2:trainLen+errorLen+1,:)'-Y(:,1:errorLen))/(ymax);
save('testRCEnKFError.mat', 'MSETimeSeries');