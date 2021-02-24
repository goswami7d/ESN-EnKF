%%%% SimulateNuminaESN.m %%%%%
%%% by D. Goswami, 2020 %%%
%%% Train and test an open-loop ESN to predict the traffic data obtained from Numina sensor
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
a = 0.3; % leaking rate
reg=1e-5; %regularization factor
psi=@(x)0.5*(1+tanh(x)); %activation function
%%% Create and train the ESN
[Wout, W, Win, x]=trainESN(data, data, psi, trainLen, initLen, resSize, a, reg);

%% Testing the trained ESN
Y = zeros(dataStateSize,testLen);
u = data(trainLen+1,:);
for t = 1:testLen 
% 	x = (1-a)*x + a*psi( Win*[1;u'] + W*x );
% 	y = Wout*[1;u';x];
    [x,y]=reservoirupdate(u',x,Win,W,Wout,a,psi);
	Y(:,t) = y;
    y(y<0)=0;
	Y(:,t) = y;
	% generative mode:
	u = y';
	% this would be a predictive mode:
	%u = data(trainLen+t+1,:);
end

errorLen = testLen;
mse = sum(vecnorm(data(trainLen+2:trainLen+errorLen+1,:)'-Y(:,1:errorLen)))./errorLen;
disp( ['MSE = ', num2str( mse )] );

% plot some results
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
save('testESNErrorNumina.mat', 'MSETimeSeries');
