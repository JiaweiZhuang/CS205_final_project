M = csvread('data/observation_train_full.csv');
s = size(M);
M1= M(2:s,:);
tic
idx = kmeans(M1,23);
toc 