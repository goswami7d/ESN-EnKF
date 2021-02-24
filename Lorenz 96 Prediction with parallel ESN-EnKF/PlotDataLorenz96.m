Error.RC=load('Lorenz96RCError.mat','MSETimeSeries');
Error.RCEnKF=load('Lorenz96RCEnKFError.mat','MSETimeSeries');

figure;
plot(Error.RC.MSETimeSeries, 'r', 'linewidth',1);
hold on;
plot(Error.RCEnKF.MSETimeSeries, 'b', 'linewidth',1);
xlabel('time-step(k)', 'Interpreter', 'latex','FontSize',18);
ylabel('$\mathcal{L}_2$ error','Interpreter', 'latex','FontSize',18);
axis tight;
%title('$L_2$ error in estimation','Interpreter', 'latex','FontSize',18);
legend('Free Running ESN Prediction', 'Estimation by ESN with Kalman Filter', 'Interpreter','latex','FontSize',15,'FontName','Times New Roman');