
% Algebra para Ciencias de los Datos
% Maestr�a en Ciencias de los Datos y Anal�tica
% M�DULO DE M�TRICAS
% PORFESOR: Henry Laniado
% Universidad Eafit
% Referencias:
% http://www.ehu.eus/~mtwmastm/TEM0910.pdf




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Outlier identification with distance based on in  norm2, norm 1 and norm Infty.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
N=100000;
x=A1%randn(N,1)   
y=A2%randn(N,1);    
p=1; %1 for distance 1, 2 for euclidean distance and inf for distance infty

Data=[x y];
N=length(x)
MeanD=mean(Data);
DC=Data-ones(N,1)*MeanD;
%D=[];
for i=1:N
    D(i)=norm(DC(i,:),Inf);
end
Cut=prctile(D,95);
I=find(D>Cut);
plot(Data(:,1),Data(:,2),'o');
hold on;
plot(Data(I,1),Data(I,2),'or', 'LineWidth', 3);
hold on;
plot(MeanD(1),MeanD(2),'ok','LineWidth', 3);
figure;
subplot(2,1,1);
hist(D);
subplot(2,1,2);
boxplot(D,'orientation','horizontal');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%M�trica Binaria
Me = imresize(imread("9.jpg"), [1280, 960]);
MeGray = rgb2gray(Me);
%figure;
%imshow(MeGray);
for i = 1:26
    A(:,:,:,i) = imresize(imread(i + ".jpg"), [1280,960]);
    I(:,:,i) = rgb2gray(A(:,:,:,i));
end
MeanGrays = mean(I, 3);
for i = 1:26
    Distance(i) = norm(cast(MeGray - I(:,:,i), "double"));
end
figure;
imshow(cast(MeanGrays, "uint8"));
[minDist, minImages] = mink(Distance, 26);

figure;
imshow(A(:,:,:,minImages(3)));

for i = 1:26
    DistanceMean(i) = norm(cast(MeanGrays - I(:,:,i)));
    
end

%I = rgb2gray(A);
%A1 = imread('9.jpeg');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%I1 = rgb2gray(A1);
%figure
%imshow(I1)
%A2 = imread('11.jpeg');
%I2 = rgb2gray(A2); %figure imshow(I) title('Original Image')
mask1 = false(size(I1));
mask1(10:end-10,10:end-10) = true;
BW1 = activecontour(I1, mask1, 300);
mask2 = false(size(I2));
mask2(10:end-10,10:end-10) = true;
BW2 = activecontour(I2, mask2, 300);
similarity = jaccard(BW1,BW2)
D = pdist(BW1-BW2,'jaccard')

