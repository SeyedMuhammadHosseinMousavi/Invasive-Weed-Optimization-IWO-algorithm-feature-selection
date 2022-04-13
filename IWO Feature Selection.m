% Invasive Weed Optimization (IWO) algorithm feature selection
% You can use your own data or change number of features by "nf"
% "nVar" should be equal with total number of features
%----------------------------------------------------------------------
% Quote: "If you set your goals ridiculously high and it's a failure,
% you will fail above everyone else's success
%                                                  James Cameron
% ---------------------------------------------------------------------

clc;
clear;
close all;
warning('off');

%% Basics 
data=Loading();
ddd=data.x';
ttt=data.t';
origindata=[ddd ttt];

nf=5;   % Desired Number of Selected Features

CostFunction=@(u) FeatureSelectionCost(u,nf,data);        % Cost Function
nVar = 13;          % Number of Decision Variables
VarSize = [1 nVar]; % Decision Variables Matrix Size
VarMin = -10;       % Lower Bound of Decision Variables
VarMax = 10;        % Upper Bound of Decision Variables

%% IWO Params
MaxIt = 10;     % Maximum Number of Iterations
nPop0 = 2;      % Initial Population Size
nPop = 6;       % Maximum Population Size
Smin = 2;       % Minimum Number of Seeds
Smax = 6;       % Maximum Number of Seeds
Exponent = 2;           % Variance Reduction Exponent
sigma_initial = 0.5;    % Initial Value of Standard Deviation
sigma_final = 0.001;	% Final Value of Standard Deviation

%% Intro
% Empty Plant Structure
empty_plant.Position = [];
empty_plant.Cost = [];
empty_plant.out = [];
pop = repmat(empty_plant, nPop0, 1);    % Initial Population Array
for i = 1:numel(pop)
% Initialize Position
pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
% Evaluation
[pop(i).Cost, pop(i).out]= CostFunction(pop(i).Position);
end
% Best Solution Ever Found
BestSol = pop(1);
% Initialize Best Cost History
BestCosts = zeros(MaxIt, 1);


%% IWO Main Body
for it = 1:MaxIt
% Update Standard Deviation
sigma = ((MaxIt - it)/(MaxIt - 1))^Exponent * (sigma_initial - sigma_final) + sigma_final;
% Get Best and Worst Cost Values
Costs = [pop.Cost];
BestCost = min(Costs);
WorstCost = max(Costs);
% Initialize Offsprings Population
newpop = [];
% Reproduction
for i = 1:numel(pop)
ratio = (pop(i).Cost - WorstCost)/(BestCost - WorstCost);
S = floor(Smin + (Smax - Smin)*ratio);
for j = 1:S
% Initialize Offspring
newsol = empty_plant;
% Generate Random Location
newsol.Position = pop(i).Position + sigma * randn(VarSize);
% Apply Lower/Upper Bounds
newsol.Position = max(newsol.Position, VarMin);
newsol.Position = min(newsol.Position, VarMax);
% Evaluate Offsring
[newsol.Cost, newsol.out] = CostFunction(newsol.Position);
% Add Offpsring to the Population
newpop = [newpop
newsol];  
end
end
% Merge Populations
pop = [pop
newpop];
% Sort Population
[~, SortOrder] = sort([pop.Cost]);
pop = pop(SortOrder);
% Competitive Exclusion (Delete Extra Members)
if numel(pop)>nPop
pop = pop(1:nPop);
end
% Store Best Solution Ever Found
BestSol = pop(1);
% Store Best Cost History
BestCosts(it) = BestSol.Cost;
% Display Iteration Information
disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCosts(it))]);
end

%% Plot IWO Train
figure;
% plot(BestCosts, 'LineWidth', 2);
semilogy(BestCosts, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;

%% Extracting Data
RealData=data.x';
% Extracting Labels
RealLbl=data.t';
FinalFeaturesInd=BestSol.out.S;
% Sort Features
FFI=sort(FinalFeaturesInd);
% Select Final Features
IWO_Features=RealData(:,FFI);
% Adding Labels
IWO_FeaturesM=IWO_Features;
IWO_FeaturesM(:,end+1)=RealLbl;

%% KNN on Original
origindat=origindata(:,1:end-1);
originlbl=origindata(:,end);
sizenet=size(origindat);
sizenet=sizenet(1,1);
Mdl = fitcknn(origindat,originlbl,'NumNeighbors',5,'Standardize',1);
rng(1); % For reproducibility
origindat = crossval(Mdl);
classErroraco = kfoldLoss(origindat);
% Predict the labels of the training data.
predictedknn = resubPredict(Mdl);
preknn=predictedknn;
ctknnaco=0;
for i = 1 : sizenet(1,1)
if originlbl(i) ~= predictedknn(i)
    ctknnaco=ctknnaco+1;
end;end;
finknn=ctknnaco*100/ sizenet;
OriginalACC=(100-finknn)-classErroraco;

%% KNN On IWO Features
Mdl = fitcknn(IWO_Features,RealLbl,'NumNeighbors',5,'Standardize',1);
rng(1); % For reproducibility
knndat = crossval(Mdl);
classError = kfoldLoss(knndat);
% Predict the labels of the training data.
predictedknn = resubPredict(Mdl);
ctknnsurf=0;
for i = 1 : sizenet(1,1)
if RealLbl(i) ~= predictedknn(i)
    ctknnsurf=ctknnsurf+1;
end;
end;
finknn=ctknnsurf*100/ sizenet;
IWOACC=(100-finknn)-classError;

%% Confusion Matrixes
figure
set(gcf, 'Position',  [150, 150, 1000, 350])
subplot(1,2,1)
cmknn = confusionchart(originlbl,preknn);
cmknn.Title = (['KNN on All Data =  ' num2str(OriginalACC) '%']);
cmknn.RowSummary = 'row-normalized';
cmknn.ColumnSummary = 'column-normalized';
subplot(1,2,2)
cmknn1 = confusionchart(originlbl,predictedknn);
cmknn1.Title = (['KNN on IWO Features =  ' num2str(IWOACC) '%']);
cmknn1.RowSummary = 'row-normalized';
cmknn1.ColumnSummary = 'column-normalized';
% Final Accuracy 
fprintf('The (KNN Accuracy on All Data) is =  %0.4f.\n',OriginalACC)
fprintf('The (KNN Accuracy on IWO Features) is =  %0.4f.\n',IWOACC)

