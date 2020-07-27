clear; clc;

% read the Numina CSV files
% nd1 =readNuminaCSV('NuminaData/Numina-umd-umd-2-2019-12-02T00_00_00-2019-12-08T23_59_59_CampusPresidential.csv');
% nd2 =readNuminaCSV('NuminaData/Numina-umd-umd-6-2019-12-02T00_00_00-2019-12-08T23_59_59_CampusPaint.csv');
% nd3 =readNuminaCSV('NuminaData/Numina-umd-umd-9-2019-12-02T00_00_00-2019-12-08T23_59_59_RegentsStadium.csv');
% nd4 =readNuminaCSV('NuminaData/Numina-umd-umd-4-2019-12-02T00_00_00-2019-12-08T23_59_59_SouthGate.csv');
% nd5 =readNuminaCSV('NuminaData/Numina-umd-umd-1-2019-12-02T00_00_00-2019-12-08T23_59_59_UniversityPaint.csv');

nd1 =readNuminaCSV('NuminaDataWeekly/Numina-umd-umd-2-2019-12-02T00_00_00-2019-12-08T23_59_59_CampusPresidential.csv');
nd2 =readNuminaCSV('NuminaDataWeekly/Numina-umd-umd-6-2019-12-02T00_00_00-2019-12-08T23_59_59_CampusPaint.csv');
nd3 =readNuminaCSV('NuminaDataWeekly/Numina-umd-umd-9-2019-12-02T00_00_00-2019-12-08T23_59_59_RegentsStadium.csv');
nd4 =readNuminaCSV('NuminaDataWeekly/Numina-umd-umd-4-2019-12-02T00_00_00-2019-12-08T23_59_59_SouthGate.csv');
nd5 =readNuminaCSV('NuminaDataWeekly/Numina-umd-umd-1-2019-12-02T00_00_00-2019-12-08T23_59_59_UniversityPaint.csv');

trainLen = 380;
testLen = 96;
initLen = 100;

% load the data
 %data=double([nd1.pedestrians, nd1.bicyclists, nd1.cars, nd1.buses, nd1.trucks]);
 data=double([nd1.cars, nd2.cars, nd3.cars, nd4.cars, nd5.cars ]);
 dataSec = nd1.elapsedTimeSec;
 %data=double(nd1.cars);

dataStateSize=size(data,2);
resSize = 4000; %reservoir size
a = 0.7; % leaking rate
reg=1e-5; %regularization factor
psi=@(x)0.5*(1+tanh(x)); %activation function
%%% Create and train the ESN
[Wout, W, Win, r]=trainESN(data, data, psi, trainLen, initLen, resSize, a, reg);

%%
% Numina nodes 
load NuminaData/UMD_campus_data;
numina(1,1) = 11; % campus dr./president ave.
numina(2,1) = 780; % campus dr./paint branch
numina(3,1) = 670; % regents dr./stadium dr.
numina(4,1) = 299; % south gate
numina(5,1) = 1168; % university blvd/paint branch

% user inputs
alpha = 0.1; % factor multiplying laplacian inside exponential
decayFactor =0.99; % to prevent global warming
proccessNoiseStdev = 80;
measurementNoiseStdev = 40;


skipFrames = 5; % for movie
xlimVec = [1250 2000];
ylimVec = [500 1250];
dt = 60*3;

% plot background
imgXData = [0 imgWidthMeters];
imgYData = [0 imgHeightMeters];
fig_img = image(flipud(img), 'XData', imgXData, 'YData', imgYData );
fig_img.AlphaData=0.7;
set(gca,'YDir','Normal')
axis equal;
hold on;
xlim([imgXData])
ylim([imgYData])
xlabel('East (m)')
ylabel('North (m)');
set(gca,'FontSize',16)

% graph laplacian
A = (A + A')/2;
G = graph(A);
D = diag(degree(G));
L = D - A; % graph laplacian

% L(L>0)=0;
% L(L<0)=0;

% highlight the nodes
for i = 1:1:length(numina)
    plot( xy(numina(i),1) , xy(numina(i),2),'ks','MarkerSize',30,'MarkerFaceColor','c')
end

% plot graph over top
plot(G, 'XData', xy(:,1), 'YData', xy(:,2),'MarkerSize',4)
xlim(xlimVec)
ylim(ylimVec)

% define observation matrix y = Cx
% x is numNodes x 1 column vector
% C is 2 x numNodes matrix
%nodeWidth = 4;
ObserveddataNodes=5;
H=zeros(length(ObserveddataNodes),numnodes(G));
%ObservedNode=numina(ObserveddataNode,:);
for i=1:length(ObserveddataNodes)
H(i,numina(ObserveddataNodes(i)))=1;
end

simVar = data;

numNodes = numnodes(G);
x0 = zeros(numNodes,1);
Q = eye(numNodes)*proccessNoiseStdev*proccessNoiseStdev;
R = eye(length(ObserveddataNodes))*measurementNoiseStdev*measurementNoiseStdev;
Pinit = 10*Q;
EnsembleSize=200;

disp('Simulating...')
[thist,xhist,Phist,yhist] = filterESNNuminaZeroOrderHold(numina,data(trainLen+2:trainLen+testLen+1,:),...
    data(trainLen+2:trainLen+testLen+1,ObserveddataNodes), dataSec(trainLen+2:trainLen+testLen+1,:),...
    ObserveddataNodes, testLen, r, a, W, Win, Wout,psi,EnsembleSize,...
    alpha, L, H, Q, R, x0, Pinit, decayFactor);


%[thist,xhist,Phist,yhist] = EnKFNuminaZeroOrderHold(numina,...
%data(trainLen+2:trainLen+testLen+1,:), dataSec(trainLen+2:trainLen+testLen+1,:),...
%    ObserveddataNodes,testLen, r,a, W, Win, Wout, psi, alpha, L, H, R,...
%    x0Ensemble, EnsembleSize, Q, decayFactor);
%  [xhist,thist,Phist,yhist] = filterNuminaZeroOrderHold(numina, data(trainLen+2:trainLen+testLen+1,:),...
%      dataSec(trainLen+2:trainLen+testLen+1,:), alpha, L, H, Q, R, Pinit, x0, dt, decayFactor);

node_var = Phist;
%datehist = datetime( posixtime(nd1.dateTime(1))+thist, 'ConvertFrom', 'posixtime');

% change what to plot
plotVar = simVar;
offset_x = [70 150 100 100 -180];
offset_y = [-120 0 -150 0 50];
animVar = xhist;
animVar2 = node_var;


%%
% animation

vidname = 'TrafficEstimation';
vidFlag=1;
if vidFlag
    vid1 = VideoWriter([vidname '.avi']);
    vid1.Quality = 75;
    vid1.FrameRate = 5;
    open(vid1);
    
end

figh = figure;
set(gcf, 'Position',  [100, 100, 1200, 800])
subplot(2,2,1)
fig_img2 = image(flipud(img), 'XData', imgXData, 'YData', imgYData );
fig_img2.AlphaData=0.7;
set(gca,'YDir','Normal')
axis equal;
hold on;
xlim([imgXData])
ylim([imgYData])
xlabel('East (m)')
ylabel('North (m)');
set(gca,'FontSize',14,'FontName','Arial')
for i = 1:1:length(numina)
    plot( xy(numina(i),1) , xy(numina(i),2),'ks','MarkerSize',20,'MarkerFaceColor','c')
    text( xy(numina(i),1)+offset_x(i) , xy(numina(i),2)+offset_y(i), num2str(i),'FontSize',16,'Color','b')
end
figG1 = plot(G, 'XData', xy(:,1), 'YData', xy(:,2),'MarkerSize',4,'NodeCData',animVar(:,1),'EdgeColor','k','LineWidth',2);
colormap(hot)
h = colorbar;
ylabel(h, 'Traffic Per 15 min.')
set(h,'FontSize',14,'FontName','Arial')
set(gcf,'Color','w')
caxis([0 3/4*max(max(animVar))])
%xlim(xlimVec)
%ylim(ylimVec)

subplot(2,2,2)
fig_img2 = image(flipud(img), 'XData', imgXData, 'YData', imgYData );
fig_img2.AlphaData=0.7;
set(gca,'YDir','Normal')
axis equal;
hold on;
xlim([imgXData])
ylim([imgYData])
xlabel('East (m)')
ylabel('North (m)');
set(gca,'FontSize',14,'FontName','Arial')
for i = 1:1:length(numina)
    plot( xy(numina(i),1) , xy(numina(i),2),'ks','MarkerSize',20,'MarkerFaceColor','c')
    text( xy(numina(i),1)+offset_x(i) , xy(numina(i),2)+offset_y(i), num2str(i),'FontSize',16,'Color','b')
end
figG2 = plot(G, 'XData', xy(:,1), 'YData', xy(:,2),'MarkerSize',4,'NodeCData',sqrt(animVar2(:,1)),'EdgeColor','k','LineWidth',2);
colormap(parula)
h = colorbar;
ylabel(h, 'Traffic Per 15 min.')
set(h,'FontSize',14,'FontName','Arial')
set(gcf,'Color','w')
caxis(sqrt([min(min(animVar2)) max(max(animVar2(:,end/2:end)))]))
%xlim(xlimVec)
%ylim(ylimVec)

subplot(2,2,[3 4])
pIndivNodes = plot(thist(1),yhist(:,1));
hold on;
pIndivNodes = plot(thist(1:i)/60/60,xhist(numina,1));
hold off;
xlim([0 24]);
ylim([0 max(max(yhist))]);
legend({'1) Campus Dr./Paint Branch','2) Regents Dr./Stadium Dr.'},'Location','Northwest','FontSize',12);
grid on;
title('Weekday Averaged Numina Data: Bicycles');
xlabel('Time of Day (hours)')
set(gca,'FontSize',14,'FontName','Arial')
ylabel('Traffic Per 15 min.')
disp('Press Key to Animate')
pause;

for i = 1:skipFrames:size(animVar,2)
    subplot(2,2,1)
    set(figG1,'NodeCData',animVar(:,i));
    titleString = sprintf('Estimate, Time: %3.2f hours',thist(i)/60/60);
    %titleString = datestr(datehist(i),'mmmm dd, yyyy HH:MM');
    title(titleString)
    %xlim(xlimVec)
    %ylim(ylimVec)
    set(gca,'FontSize',14)
    
    subplot(2,2,2)
    set(figG2,'NodeCData',sqrt(animVar2(:,i)));
    %titleString = datestr(datehist(i),'mmmm dd, yyyy HH:MM');
    title('Standard Deviation')
    %xlim(xlimVec)
    %ylim(ylimVec)
    set(gca,'FontSize',14)    
    
    subplot(2,2,[3 4])
    %ind = interp1(nd1.dateTime,[1:1:length(nd1.dateTime)],datehist(i),'previous');% determine index based on time
    % doesnt work with datetime
    %set(pIndivNodes,'XData',nd1.dateTime(1:ind),'YData',dataTotal(1:ind,:))
    pIndivNodes = plot(thist(1:i)/60/60,yhist(:,1:i),'.-','MarkerSize',5);
    hold on;
    pIndivNodes = plot(thist(1:i)/60/60,xhist(numina,1:i),'.-','MarkerSize',5);
    hold off;
    xlim([0 24]);
    %xlim([nd1.dateTime(1) nd1.dateTime(end)]);
    ylim([0 max(max(yhist))]);
    %legend({'1) Campus Dr./Paint Branch','2) Regents Dr./Stadium Dr.'},'Location','Northwest','FontSize',12);
	legend({'1) Campus Dr./President Ave.','1) Campus Dr./President Ave. Prediction',...
        '2) Campus Dr./Paint Branch','2) Campus Dr./Paint Branch Prediction',...
        '3) Regents Dr./Stadium Dr.', '3) Regents Dr./Stadium Dr. Prediction','4) South Gate','4) South Gate Prediction',...
        '5) University Blvd./Paint Branch','5) University Blvd./Paint Branch Prediction'},...
        'Location','Northeast','FontSize',12);
    grid on;
    title('Weekday Averaged Numina Data: Motor Vehicles');
    ylabel('Traffic Per 15 min.')
    
    xlabel('Time of Day (hours)')
    set(gca,'FontSize',14,'FontName','Arial')
    %pause(0.1);
    drawnow;
    if vidFlag==1
       writeVideo(vid1, getframe(figh));
     end
end

if vidFlag==1
 close(vid1);
end

