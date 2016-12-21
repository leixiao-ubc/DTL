function [] = reset_parpool(M)

%if(isdeployed)
   delete(gcp);
   myCluster = parcluster('local');
   myCluster.NumWorkers = M;
   parpool(myCluster, M)
%end
    
end