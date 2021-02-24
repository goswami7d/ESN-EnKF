clear;clc;

% load the data
trainLen = 2000;
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
stateSize=40;
inSize = 4; outSize = 1;
resSize = 1000;
a = 0.8; % leaking rate
rand( 'seed', 42 );
Win = (rand(resSize,inSize)-0.5) .* 1;
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
       X = zeros(1+inSize+resSize,trainLen-initLen,stateSize);
       %X = zeros(resSize,trainLen-initLen,stateSize);
for i=1:stateSize
    
  if i==1
      index=[stateSize-1 stateSize 1 2];
      
  elseif i==2
          index=[stateSize 1 2 3];
          
  elseif i==stateSize
          index=[stateSize-2 stateSize-1 stateSize 1];
  else 
      index=[i-2 i-1 i i+1];
         
  end
       
   % set the corresponding target matrix directly
       Yt = data(initLen+2:trainLen+1,i)';

% run the reservoir with the data and collect X
x = zeros(resSize,stateSize);
for t = 1:trainLen
	u = data(t,index);
	x(:,i) = (1-a)*x(:,i) + a*tanh( Win*[u'] + W*x(:,i) );
%     x(:,i) = (1-a)*x(:,i) + a*( Win*[u'] + W*x(:,i) );
%      x(:,i) = x(:,i)/norm(x(:,i));
	if t > initLen
		X(:,t-initLen,i) = [1;u';x(:,i)];
        %X(:,t-initLen,i) = [x(:,i)];
	end
end

% train the output by ridge regression
reg = 1e-6;  % regularization coefficient
% using Matlab solver:
Wout(:,:,i) = ((X(:,:,i)*X(:,:,i)' + reg*eye(1+inSize+resSize)) \ (X(:,:,i)*Yt'))'; 
%Wout(:,:,i) = ((X(:,:,i)*X(:,:,i)' + reg*eye(resSize)) \ (X(:,:,i)*Yt'))'; 



end
% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(outSize,testLen,stateSize);


%x = zeros(resSize,1);
for t = 1:testLen 
  for i=1:stateSize
   if i==1
      index=[stateSize-1 stateSize 1 2];
      
  elseif i==2
          index=[stateSize 1 2 3];
          
  elseif i==stateSize
          index=[stateSize-2 stateSize-1 stateSize 1];
  else 
      index=[i-2 i-1 i i+1];
         
  end  
  if t==1 
     u = data(trainLen+1,index);
  else
     u=(reshape(Y(:,t-1,index),1,inSize));
  end
	x(:,i) = (1-a)*x(:,i) + a*tanh( Win*[u'] + W*x(:,i) );
%     x(:,i) = (1-a)*x(:,i) + a*( Win*[u'] + W*x(:,i) );
%     x(:,i) = x(:,i)/norm(x(:,i));
	y = Wout(:,:,i)*[1;u';x(:,i)];
    %y = Wout(:,:,i)*[x(:,i)];
	Y(:,t,i) = y;
	
  end
end

errorLen = testLen;
%mse = sum(vecnorm(data(trainLen+2:trainLen+errorLen+1,:)'-transpose(reshape(Y(:,1:errorLen,:),[errorLen stateSize]))))./errorLen;
mse=sqrt(immse(data(trainLen+2:trainLen+errorLen+1,:)',transpose(reshape(Y(:,1:errorLen,:),[errorLen stateSize]))));
disp( ['MSE = ', num2str( mse )] );

for i=1:stateSize
    R=corrcoef(data(trainLen+2:trainLen+testLen+1,i),Y(:,:,i));
    correlation(i)=R(1,2);
end
save('Lorenz96RC.mat', 'correlation');

% plot some signals
figure(1);
plot( data(trainLen+2:trainLen+testLen+1,1), 'color', [0,0.75,0],'linewidth',1 );
hold on;
plot( Y(:,:,1), 'b-.', 'linewidth',1);
hold off;
axis tight;
xlabel('time', 'Interpreter', 'latex');
ylabel('$x_1$','Interpreter', 'latex');
title('Target and generated signals $x_1(n)$ starting at n=0','Interpreter', 'latex');
legend('Target signal', 'Predicted signal');

figure(2);
plot( data(trainLen+2:trainLen+testLen+1,2), 'color', [0,0.75,0],'linewidth',1 );
hold on;
plot( Y(:,:,2), 'b-.', 'linewidth',1);
hold off;
axis tight;
ylabel('$x_2$','Interpreter', 'latex');
title('Target and generated signals $x_2(n)$ starting at n=0','Interpreter', 'latex');
legend('Target signal', 'Predicted signal');

figure(3);
plot( data(trainLen+2:trainLen+testLen+1,3), 'color', [0,0.75,0],'linewidth',1 );
hold on;
plot( Y(:,:,3), 'b-.', 'linewidth',1);
hold off;
axis tight;
xlabel('time', 'Interpreter', 'latex');
ylabel('$x_3$','Interpreter', 'latex');
title('Target and generated signals $x_3(n)$ starting at n=0','Interpreter', 'latex');
legend('Target signal', 'Predicted signal');

figure(4);
plot( X(1:20,1:100,1)' );
title('Some reservoir activations $r(n)$','Interpreter', 'latex');

figure(5);
bar( Wout(:,:,1)' )
title('Output weights $W^{out}$','Interpreter', 'latex');
ymax=max(vecnorm(data(trainLen+2:trainLen+errorLen+1,:)'));
ymin=min(vecnorm(data(trainLen+2:trainLen+errorLen+1,:)'));

MSETimeSeries=vecnorm(data(trainLen+2:trainLen+errorLen+1,:)'-Y(:,1:errorLen))/(ymax);
save('Lorenz96RCError.mat', 'MSETimeSeries');