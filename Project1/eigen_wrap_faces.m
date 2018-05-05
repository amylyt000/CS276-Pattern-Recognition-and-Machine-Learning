clear;close all;clc;
%% Load all data from file landmark_87
all_data = zeros(87*2,177);
for i = 1:177
    if i < 104
        dtpath = sprintf('./face_data/landmark_87/face%03d_87pt.dat',i-1);
    end
    if i >= 104
        dtpath = sprintf('./face_data/landmark_87/face%03d_87pt.dat',i);
    end
    data = textread(dtpath);
    all_data(:,i) = [data(2:88,1);data(2:88,2)];
end
train_data = all_data(:, 1:150);
test_data = all_data(:, 151:177);

%% Calculate and display the mean warpping
[r,c] = size(train_data);
mean = zeros(r, 1);
for i = 1:r
    mean(i) = sum(train_data(i,:))/c;
end
figure;
[train_img, test_img] = load_data();
[r_img, c_img] = size(train_img);
mean_img = zeros(r_img, 1);
for i = 1:r_img
    mean_img(i) = sum(train_img(i,:))/c_img;
end
imshow(reshape(uint8(mean_img),[256,256]));
hold on;
plot(mean(1:87),mean(88:174),'.','MarkerSize',8);
hold off;


%% Calculate and display the first 5 eigen warpping of the landmarks
diff_matrix = zeros(r,c);
for i = 1:c
    diff_matrix(:,i) = train_data(:,i) - mean(:);
end
scatter_matrix = diff_matrix' * diff_matrix; % size(scatter_matrix)=[150,150]
% [U, S, V] = svd(scatter_matrix);
% e = diff_matrix * U;
[V, D] = eigs(scatter_matrix, 149);
% [D, I] = sort(diag(D), 'descend');
% V = V(:,I);
e = diff_matrix * V;
% for i = 1:150 
%     e(:,i) = e(:,i)/norm(e(:,i));
% end
for i = 1:149
    e(:,i) = e(:,i)/norm(e(:,i));
end 
%{
figure;
for i = 1:5
    subplot(2,3,i);
    eigenwarpping = e(:,i) + mean;
    plot(255-eigenwarpping(1:87), 255-eigenwarpping(88:174),'b.','MarkerSize',8);
end
%}
figure;imshow(reshape(uint8(mean_img),[256,256]));
hold on;
eigenwarpping = 10*sqrt(e(:, 1)) + mean;
plot(eigenwarpping(1:87), eigenwarpping(88:174),'b.','MarkerSize',8);
hold on;
eigenwarpping = 10*sqrt(e(:, 2)) + mean;
plot(eigenwarpping(1:87), eigenwarpping(88:174),'g.','MarkerSize',8);
hold on;
eigenwarpping = 10*sqrt(e(:, 3)) + mean;
plot(eigenwarpping(1:87), eigenwarpping(88:174),'r.','MarkerSize',8);
hold on;
eigenwarpping = 10*sqrt(e(:, 4)) + mean;
plot(eigenwarpping(1:87), eigenwarpping(88:174),'c.','MarkerSize',8);
hold on;
eigenwarpping = 10*sqrt(e(:, 5)) + mean;
plot(eigenwarpping(1:87), eigenwarpping(88:174),'m.','MarkerSize',8);
hold off;


%% Reconstruct landmarks for test faces
[r_test, c_test] = size(test_data);
diff_matrix_test = zeros(r_test,c_test);
for i = 1:c_test
    diff_matrix_test(:,i) = test_data(:,i) - mean(:);
end
b = e(:,1:5)'*diff_matrix_test;
reconstructed_lm= repmat(mean,1,c_test) + e(:,1:5) * b;
figure; title('reconstructed landmarks');
for i = 1:27
    subplot(5,6,i);
    path = sprintf('./face_data/face/face%03d.bmp',i+150);
    imshow(imread(path));
    hold on;
    re_lm = reconstructed_lm(:,i);
    lm = test_data(:,i);
    plot(re_lm(1:87),re_lm(88:174),'b.','MarkerSize',8);
    hold on;
    plot(lm(1:87),lm(88:174),'r.','MarkerSize',8);   
end
hold off;

%% Calculate and display the test error
error = zeros(149,1); % 5
recon_lm = zeros(174, 27);
for i = 1:149  % 5
    b = e(:,1:i)'*diff_matrix_test;
    recon_lm = repmat(mean,1,c_test) + e(:,1:i) * b;
    error(i) = sqrt(sum(sum((test_data - recon_lm).^2))/27);
%     error(i) = sqrt(sum(sum((test_data - recon_lm).^2)))/27;
    
%     for j = 1:27
%         per_error = sum(sqrt(abs(sum(reshape(test_data(:,j),[87,2])-reshape(recon_lm(:,j),[87,2]).^2,1))),2)/87;
%         error(i) = error(i) + per_error;
%     end
%     error(i) = error(i)/27;
end
figure;
plot(1:149, error); % 5
title('Total Reconstruction Error of Landmarks');
xlabel('No. of Eigen-warppings k');
ylabel('Total Reconstruction Error');
 
    
    

