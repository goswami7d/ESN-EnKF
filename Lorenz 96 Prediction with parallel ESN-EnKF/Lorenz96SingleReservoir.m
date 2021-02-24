clear;clc;

% load the data
trainLen = 4000;
testLen = 500;
initLen = 200;
%data = load('MackeyGlass_t17.txt');
 %load('dataVanderPol.mat', 'Xf');
 %load('dataLorenz.mat','Xf');
 load('dataLorenz96.mat','Xf');
 data=Xf;

% plot some of it
figure(10);
plot(data(1:1000));
title('A sample of data');

% generate the ESN reservoir
inSize = 40; outSize = 40;
resSize = 2000;
a = 0.7;%0.7; % leaking rate
rand( 'seed', 42 );
Win = (rand(resSize,1+inSize)-0.5) .* 1;
% dense W:
%W = rand(resSize,resSize)-0.5;
% sparse W:
 W = sprand(resSize,resSize,0.01);
 W_mask = (W~=0); 
 W(W_mask) = (W(W_mask)-0.5);

% normalizing and setting spectral radius
disp 'Computing spectral radius...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
disp 'done.'
W = W .* ( 1.25 /rhoW);

W=triu(W); W=W-W';

% allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize,trainLen-initLen);
% set the corresponding target matrix directly
Yt = data(initLen+2:trainLen+1,:)';

% run the reservoir with the data and collect X
x = zeros(resSize,1);
for t = 1:trainLen
	u = data(t,:);
	x = (1-a)*x + a*tanh( Win*[1;u'] + W*x );
	if t > initLen
		X(:,t-initLen) = [1;u';x];
	end
end

% train the output by ridge regression
reg = 1e-5;  % regularization coefficient
% % direct equations from texts:
% X_T = X'; 
% Wout = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
% using Matlab solver:
Wout = ((X*X' + reg*eye(1+inSize+resSize)) \ (X*Yt'))'; 

% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(outSize,testLen);
u = data(trainLen+1,:);
%x = zeros(resSize,1);
for t = 1:testLen 
	x = (1-a)*x + a*tanh( Win*[1;u'] + W*x );
	y = Wout*[1;u';x];
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
plot( data(trainLen+2:trainLen+testLen+1,1), 'color', [0,0.75,0],'linewidth',1 );
hold on;
plot( Y(1,:), 'b-.', 'linewidth',1);
hold off;
axis tight;
xlabel('time', 'Interpreter', 'latex');
ylabel('$x_1$','Interpreter', 'latex');
title('Target and generated signals $x_1(n)$ starting at n=0','Interpreter', 'latex');
legend('Target signal', 'Predicted signal');

figure(2);
plot( data(trainLen+2:trainLen+testLen+1,2), 'color', [0,0.75,0],'linewidth',1 );
hold on;
plot( Y(2,:), 'b-.', 'linewidth',1);
hold off;
axis tight;
ylabel('$x_2$','Interpreter', 'latex');
title('Target and generated signals $x_2(n)$ starting at n=0','Interpreter', 'latex');
legend('Target signal', 'Predicted signal');

figure(3);
plot( data(trainLen+2:trainLen+testLen+1,3), 'color', [0,0.75,0],'linewidth',1 );
hold on;
plot( Y(3,:), 'b-.', 'linewidth',1);
hold off;
axis tight;
xlabel('time', 'Interpreter', 'latex');
ylabel('$x_3$','Interpreter', 'latex');
title('Target and generated signals $x_3(n)$ starting at n=0','Interpreter', 'latex');
legend('Target signal', 'Predicted signal');

figure(4);
plot( X(1:20,1:100)' );
title('Some reservoir activations $r(n)$','Interpreter', 'latex');

figure(5);
bar( Wout' )
title('Output weights $W^{out}$','Interpreter', 'latex');