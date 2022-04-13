function [z, out]=FeatureSelectionCost(u,nf,data)
    % Read Data Elements
    x=data.x;
    t=data.t;
    % Create Permutation uinsg Random Keys
    [~, q]=sort(u);
    % Selected Features
    S=q(1:nf);
    % Ratio of Selected Features
    rf=nf/numel(q);
    % Selecting Features
    xs=x(S,:);
    % Weights of Train and Test Errors
    wTrain=0.8;
    wTest=1-wTrain;
    % Number of Runs
    nRun=3;
    EE=zeros(1,nRun);
    for r=1:nRun
        % Create and Train ANN
        results=CreateAndTrainANN(xs,t);

        % Calculate Overall Error
        EE(r) = wTrain*results.TrainData.E + wTest*results.TestData.E;
    end
    E=mean(EE);
    % Calculate Final Cost
    z=E;
    % Set Outputs
    out.S=S;
    out.nf=nf;
    out.rf=rf;
    out.E=E;
    out.z=z;
end