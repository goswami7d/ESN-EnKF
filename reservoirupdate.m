function [x,y]=reservoirupdate(u,x,Win,W,Wout,a,activation)

if nargin<5
    error('Not enough input argument')
elseif nargin<6
    activation=@(x) tanh(x); 
end

x = (1-a)*x + a*activation( Win*[1;u] + W*x );
y = Wout*[1;u;x];