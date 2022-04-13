function data=Loading()
    dataset=load('iwodata');
    data.x=dataset.Inputs;
    data.t=dataset.Targets;
    data.nx=size(data.x,1);
    data.nt=size(data.t,1);
    data.nSample=size(data.x,2);
end