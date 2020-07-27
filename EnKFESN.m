function [stateEstimate,UpdatedEnsemble,reservoirStates]...
    =EnKFESN(h,Ensemble,Observation,ObservationCovariance,EnsembleSize,reservoirStates,Win,W,Wout,a,activation)

stateSize=size(Ensemble,1);
outSize=length(Observation);
UpdatedEnsemble=zeros(stateSize,EnsembleSize);
ObservationForecast=zeros(outSize,EnsembleSize);
for i=1:EnsembleSize
	[reservoirStates,UpdatedEnsemble(:,i)]=reservoirupdate(Ensemble(:,i),reservoirStates,Win,W,Wout,a,activation);
    ObservationForecast(:,i)=h(UpdatedEnsemble(:,i));
end 
   EnsembleMean=mean(UpdatedEnsemble,2);
   ObservationForecastMean=mean(ObservationForecast,2);
   for j=1:stateSize
       Ex(j,:)=UpdatedEnsemble(j,:)-EnsembleMean(j);
   end
   for j=1:outSize
       Ey(j,:)=ObservationForecast(j,:)-ObservationForecastMean(j);
   end
   
   Pxy=Ex*Ey'/(EnsembleSize-1);
   Pyy=Ey*Ey'/(EnsembleSize-1)+ObservationCovariance;
   K=Pxy*inv(Pyy);
   UpdatedEnsemble=UpdatedEnsemble+K*(Observation-ObservationForecast);
   stateEstimate=mean(UpdatedEnsemble,2);