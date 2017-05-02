M = csvread('data/observation_sample.csv');
s = size(M);
M1= M(2:s,:);
score = zeros(30,1);
for i = 5:30
    disp(i)
    tic
    idx = kmeans(M1,i,'Distance','sqEuclidean');
    toc   
    tic
    [s,h]=silhouette(M1,idx,'sqEuclidean');
    toc
    score(i)=mean(s);
end    
disp(score);