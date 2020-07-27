%%%%%Function trainESN.m  %%%%%%%%%%

%% Written by Debdipta Goswami, UMD
% [Wout, X]=trainESN(data, trainLen, initLen, resSize, a, reg)
% Trains the output weights of an eco-state network from a set of input and output data-stream
% The reservoir is chosen as an Erdos-Renyi graph of p=0.01.
% Inputs:
% dataIn = data stream with rows as multidimensional input data and columns as time steps
% dataOut = data stream with rows as multidimensional output data and columns as time steps
% psi = activation function default function: tanh()
% trainLen = training time length, default value = 10000
% initLen = initial time length discarded during the training, dafault value = 100
% resSize = reservoir size, default value = 1000
% a = leaking rate, default value = 0.7
% reg = Tikhonov regularization coffecient, default value = 1e-6
% Outputs:
% Wout = output weights
% W = reservoir adjacency matrix (reservoir weights)
% Win = input weights
% X = final state of the reservoir


function [Wout, W, Win, x]=trainESN(dataIn, dataOut, activation, trainLen, initLen, resSize, a, reg)

switch nargin
    case 1
        dataOut=dataIn;
        activation= @(x) tanh(x);
        trainLen=10000;
        initLen=100;
        resSize=1000;
        a=0.7;
        reg=1e-6;
    case 2
        activation= @(x) tanh(x);
        trainLen=10000;
        initLen=100;
        resSize=1000;
        a=0.7;
        reg=1e-6;
    case 3
        trainLen=10000;
        initLen=100;
        resSize=1000;
        a=0.7;
        reg=1e-6;
    case 4
        initLen=100;
        resSize=1000;
        a=0.7;
        reg=1e-6;
    case 5
        resSize=1000;
        a=0.7;
        reg=1e-6;  
    case 6
        a=0.7;
        reg=1e-6;
    case 7
        reg=1e-6;
end



% generate the ESN reservoir
inSize = size(dataIn,2); outSize = size(dataOut,2);
rand( 'seed', 42 );
Win = (rand(resSize,1+inSize)-0.5) .* 1;
% dense W:
%W = rand(resSize,resSize)-0.5;
% sparse W:
 W = sprand(resSize,resSize,0.01);
 W_mask = (W~=0); 
 W(W_mask) = (W(W_mask)-0.5);
 W=triu(W); W=W-W';
% normalizing and setting spectral radius
disp 'Computing spectral radius...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
disp 'done.'
W = W .* ( 1.25 /rhoW);

% allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize,trainLen-initLen);
% set the corresponding target matrix directly
Yt = dataOut(initLen+2:trainLen+1,:)';

% run the reservoir with the data and collect X
x = zeros(resSize,1);
for t = 1:trainLen
	u = dataIn(t,:);
	x = (1-a)*x + a*activation( Win*[1;u'] + W*x );
	if t > initLen
		X(:,t-initLen) = [1;u';x];
	end
end

% train the output by ridge regression
% % direct equations from texts:
% X_T = X'; 
% Wout = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
% using Matlab solver:
Wout = ((X*X' + reg*eye(1+inSize+resSize)) \ (X*Yt'))'; 
