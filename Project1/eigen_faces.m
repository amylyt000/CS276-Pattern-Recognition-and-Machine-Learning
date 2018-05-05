%% Calculate the mean face
clear;clc;close all;
[train, test] = load_data();
[r, c] = size(train);
mean = zeros(r, 1);
for i = 1:r
    mean(i) = sum(train(i,:))/c;
end
figure;imshow(reshape(uint8(mean),[256,256])); title('mean face');

%% Scatter matrix
diff_matrix = zeros(r,c);
for i = 1:c
    diff_matrix(:,i) = train(:,i) - mean;
end
scatter_matrix = diff_matrix' * diff_matrix; % size(scatter_matrix)=[150,150]
%??
[U, S, V] = svd(scatter_matrix);
% [V, D] = eigs(scatter_matrix,149);
%[V, D] = eigs(scatter_matrix,149);
% e = diff_matrix * U;
% [D, I] = sort(diag(D), 'descend');
% V = V(:,I);
% for i = 1:149
%     e(:,i) = diff_matrix * V(:,i);
% end

e = diff_matrix * U(:,1:149);
for i = 1:149 %149
    e(:,i) = e(:,i)/norm(e(:,i));
end

%% Show first 20 eigen faces
figure;title('20 eigen faces');
for i = 1:20
    subplot(4,5,i);
    revise_img = uint8(255*mat2gray(e(:,i)));
    imshow(reshape(revise_img',[256,256]));
end

%% Reconstruct test images
[r_test, c_test] = size(test);
diff_matrix_test = zeros(r_test,c_test);
for i = 1:c_test
    diff_matrix_test(:,i) = test(:,i) - mean(:);
end

a = e'*diff_matrix_test;
reconstructed_img= repmat(mean,1,c_test) + e * a;
figure; title('reconstructed images');
for i = 1:27
    subplot(5,6,i);
    recon_imgs = uint8(255*mat2gray(reconstructed_img(:,i)));
    imshow(reshape(recon_imgs,[256,256]));
end

%% Calculate test error
error = zeros(149,1);
recon_img = zeros(65536, 27);
for i = 1:149
    a = e(:,1:i)'*diff_matrix_test;
    recon_img = repmat(mean,1,c_test) + e(:,1:i) * a;
    error(i) = sum(sum((test - recon_img).^2))/27/65536;
end
figure;
plot(1:149, error);
title('total reconstruction error');
xlabel('eigen-faces k');
ylabel('total reconstruction error');
 
figure;
plot(1:20, error(1:20));
title('reconstruction error for 20 eigen-faces');
xlabel('eigen-faces k');
ylabel('reconstruction error');