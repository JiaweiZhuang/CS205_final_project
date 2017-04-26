nt = 365*49-7;
levels = 18;
fileID = fopen('data/SSWdata.bin');
X=fread(fileID,[nt*levels*14 1],'double');
X=reshape(swapbytes(X),[levels*14 nt]);
size(X)
tic
idx = kmeans(X',3,'Distance','correlation');
toc
tic
[s,h]=silhouette(X',idx,'correlation');
toc
