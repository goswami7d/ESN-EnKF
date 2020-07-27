clear;clc;

trainLen = 2000;
testLen = 200;
initLen = 200;

%%% Load the data from the file
%data = load('MackeyGlass_t17.txt');
load('dataLorenz.mat','Xf');
data=Xf;


dataStateSize=size(data,2);
a=0.3; %leakage rate
resSize=1000; %reservoir size
reg=1e-6; %regularization factor
psi=@(x)tanh(x); %activation function
%%% Create and train the ESN
[Wout, W, Win, x]=trainESN(data, data, psi, trainLen, initLen, resSize, a, reg);


%% Testing the trained ESN
Y = zeros(dataStateSize,testLen);
u = data(trainLen+1,:);
for t = 1:testLen 
	%x = (1-a)*x + a*psi( Win*[1;u'] + W*x );
	%y = Wout*[1;u';x];
    [x,y]=reservoirupdate(u',x,Win,W,Wout,a,psi);
	Y(:,t) = y;
	% generative mode:
	u = y';
	% this would be a predictive mode:
	%u = data(trainLen+t+1);
end

errorLen = testLen;
mse = sum(vecnorm(data(trainLen+2:trainLen+errorLen+1,:)'-Y(:,1:errorLen)))./errorLen;
disp( ['MSE = ', num2str( mse )] );

% plot some signals
figure(1);

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
save('testESNError.mat', 'MSETimeSeries');