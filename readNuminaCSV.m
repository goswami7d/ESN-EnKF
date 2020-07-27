function numinaData =readNuminaCSV(csvFileName)
% Inputs: 
%
% csvFileName e.g., Numina-umd-umd-1-2019-12-02T00_00_00-2019-12-08T23_59_59_UniversityPaint.csv
%
% Outputs:
% 
% numinaData.t 
% numinaData.pedestrians =
% numinaData.bicyclists
% numinaData.cars 
% numinaData.buses
% numinaData.trucks 

fid = fopen(csvFileName);
% header: time,pedestrians,bicyclists,cars,buses,trucks
% 2019-12-02T00:00:00-05:00,1,0,62,3,0
C = textscan(fid,'%d-%d-%dT%d:%d:%d-%d:%d,%d,%d,%d,%d,%d', 'headerLines', 1);
fclose(fid);

% year month day
Y = C{1};
M = C{2};
D = C{3};
H = C{4};
MI = C{5};
S = C{6};

% create a datetime array
numinaData = struct;
numinaData.dateTime = datetime(Y,M,D,H,MI,S);
numinaData.elapsedTimeSec = posixtime(numinaData.dateTime) - posixtime(numinaData.dateTime(1));
numinaData.pedestrians = C{9};
numinaData.bicyclists = C{10};
numinaData.cars = C{11};
numinaData.buses = C{12};
numinaData.trucks = C{13};
numinaData.total = C{7} + C{8} + C{9} + C{10} + C{11};


end
