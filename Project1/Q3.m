% close all; clear; clc;
%% ------------------------------------------------------------------------
%  ------------------------------------------------------------------------
%% For the training images
% we first align the images by warping their landmarks into the mean 
% position, and then compute the eigen-faces (appearance) from these 
% aligned images.

%% Load all data from file landmark_87
all_landmark = zeros(87*2,177);
for i = 1:177
    if i < 104
        dtpath = sprintf('./face_data/landmark_87/face%03d_87pt.dat',i-1);
    end
    if i >= 104
        dtpath = sprintf('./face_data/landmark_87/face%03d_87pt.dat',i);
    end
    data = textread(dtpath);
    all_landmark(:,i) = [data(2:88,1);data(2:88,2)];
end
train_lm = all_landmark(:, 1:150);
test_lm = all_landmark(:, 151:177);

%% Calculate the mean-warpping and eigen-warpping of landmarks
[r_lm_train,c_lm_train] = size(train_lm);
mean_lm = sum(train_lm,2)/c_lm_train;
diff_lm_train = train_lm - mean_lm;
scatter_lm = diff_lm_train' * diff_lm_train; % size(scatter_lm)=[150,150]

[V_lm, D_lm] = eigs(scatter_lm,10); % can take top 10 eigen-vectors of warpping
eigen_lm = diff_lm_train * V_lm;
for i = 1:10 % 149
    eigen_lm(:,i) = eigen_lm(:,i)/norm(eigen_lm(:,i));
end

%% Read images and warp them
[train_img, test_img] = load_data();
[r_img_train, c_img_train] = size(train_img);
warp_img_train = zeros(r_img_train, c_img_train);
figure;
for i = 1 : c_img_train
    warp_img_train(:,i) = double(reshape(warpImage_new(reshape(train_img(:,i),[256,256]), ...
        reshape(train_lm(:,i),[87,2]),...
        reshape(mean_lm, [87,2])),[65536,1])); % type is uint8
    subplot(10,15,i);
    imshow(reshape(uint8(warp_img_train(:,i)), [256,256]));
end


mean_warp_img_train = zeros(65536, 1);
for i = 1:65536
    mean_warp_img_train(i) = sum(warp_img_train(i,:))/150;
end

diff_warp_img_train = zeros(65536,150);
for i = 1:150
    diff_warp_img_train(:,i) = warp_img_train(:,i) - mean_warp_img_train;
end
scatter_warp_img_train = diff_warp_img_train' * diff_warp_img_train; % size(scatter_matrix)=[150,150]

[U_warp_img_train, S_warp_img_train, V_warp_img_train] = svd(scatter_warp_img_train);


e_warp_img_train = diff_warp_img_train * U_warp_img_train(:,1:149);
for i = 1:149 %149
    e_warp_img_train(:,i) = e_warp_img_train(:,i)/norm(e_warp_img_train(:,i));
end

%% Show first 20 eigen faces
figure;title('10 eigen faces');
for i = 1:10
    subplot(2,5,i);
    revise_warp_img_train = uint8(255*mat2gray(e_warp_img_train(:,i)));
    imshow(reshape(revise_warp_img_train',[256,256]));
end



%% ------------------------------------------------------------------------
% -------------------------------------------------------------------------
%% For each testing image:
%% (i) project its landmarks to the top 10 eigen-warppings, you get the
% reconstructed landmarks.
[r_lm_test, c_lm_test] = size(test_lm);
diff_lm_test = test_lm - mean_lm;

b = eigen_lm(:,1:10)'*diff_lm_test;
% reconstructed_lm= repmat(mean_lm,1,c_lm_test) + eigen_lm(:,1:10) * b;
reconstructed_lm= eigen_lm(:,1:10) * b;
for i = 1:10
    subplot(2,5,i);
    eigenwarpping = eigen_lm(:,i) + mean_lm;
    plot(255-eigenwarpping(1:87), 255-eigenwarpping(88:174),'b.','MarkerSize',8);
end
%% (ii) warp the face image to the mean position 
[r_img_test, c_img_test] = size(test_img);
warp_img_test = zeros(r_img_test, c_img_test);
for i = 1 : c_img_test
    warp_img_test(:,i) = double(reshape(warpImage_new(reshape(test_img(:,i),[256,256]), ...
        reshape(test_lm(:,i),[87,2]),...
        reshape(mean_lm, [87,2])),[65536,1])); % type is uint8
end
%  and then project to the top 10 eigen-faces, you get the reconstructed 
% images at mean postition.
[r_img_train, c_img_train] = size(train_img);
mean_img = sum(train_img,2)/c_img_train;
diff_img_train = train_img - mean_img;

scatter_img = diff_img_train' * diff_img_train; 

[V_img, D_img] = eigs(scatter_img,10);
eigen_img = diff_img_train * V_img;
for i = 1:10 % 149
    eigen_img(:,i) = eigen_img(:,i)/norm(eigen_img(:,i));
end

% diff_img_test = test_img - mean_img;
% a = eigen_img'*diff_img_test;
a = eigen_img'*(warp_img_test - mean_img);
% reconstructed_img= repmat(mean_img,1,c_img_test) + eigen_img * a; % 65536*27 double
reconstructed_img= eigen_img * a; % 65536*27 double
% % % 
% reconstructed_img = 255*mat2gray(reconstructed_img);
figure;
for i = 1:27
    subplot(5,6,i);
    imshow(uint8(reshape(mean_img + reconstructed_img(:,i),[256,256])));
end

%% (iii) warp the reconstructed faces in step(ii) to the positions
% reconstructed in step(i). note that this new image is constructed from 20
% numbers. 
warp_img_test_lm = zeros(r_img_test, c_img_test);
for i = 1 : c_img_test
    warp_img_test_lm(:,i) = double(reshape(warpImage_new(...
        reshape(mean_img + reconstructed_img(:,i),[256,256]), ...
        reshape(mean_lm,[87,2]),...
        reshape(mean_lm + reconstructed_lm(:,i), [87,2])),[65536,1])); % type is uint8
end
% % %
% warp_img_test_lm = 255*mat2gray(warp_img_test_lm);
% Then compare the reconstructed faces against the original testing images.
figure;title('reconstructed faces')
for i = 1:27
    subplot(5,6,i);
    imshow(uint8(reshape(warp_img_test_lm(:,i),[256,256])));
end

figure;title('original testing faces')
for i = 1:27
    subplot(5,6,i);
    imshow(uint8(reshape(test_img(:,i),[256,256])));
end

%% (iv) Plot the reconstruction errors per pixel against the number of
% eigen-faces k.
reconstructed_err = zeros(149,1);
reconstructed_test_img = zeros(65536, 27);
warp_img_test_again = zeros(65536,27);
[V_img, D_img] = eigs(scatter_img,149);
eigen_img = diff_img_train * V_img;
for i = 1:149 % 149
    eigen_img(:,i) = eigen_img(:,i)/norm(eigen_img(:,i));
end
for i = 1:149
    %a = eigen_img(:,1:i)'*(warp_img_test-mean_img); % shoule be warp_img_test,
    %finally knows where is wrong!!
    %reconstructed_test_img = repmat(mean_img,1,c_img_test) + eigen_img(:,1:i) * a;
    reconstructed_test_img = reconstructed_test_img + eigen_img(:,i)*(eigen_img(:,i)'*(warp_img_test - mean_img));
    for j = 1:27
         warp_img_test_again(:,j) = double(reshape(warpImage_new(...
            reshape(mean_img + reconstructed_test_img(:,j),[256,256]), ...
            reshape(mean_lm,[87,2]),...
            reshape(mean_lm + reconstructed_lm(:,j), [87,2])),[65536,1]));
    end
    reconstructed_err(i) = sum(sum((test_img - warp_img_test_again).^2))/27/65536;
    disp(sprintf('%d eigen-faces error is %d ',i, reconstructed_err(i)));
end
figure;
plot(1:149, reconstructed_err);

title('Reconstruction Error per Pixel vs no. of Eigen-faces');
xlabel('no. of Eigen-faces');
ylabel('Reconstruction Error per Pixel');

%% Q4 - Synthesize random faces by a random sampling of the landmarks
% top 10 eigen-values and eigen-vectors of landmarks
[V_img, D_img] = eigs(scatter_img,149);
eigen_img = diff_img_train * V_img;
a = eigen_img'*(warp_img_test - mean_img);
for i = 1:149 % 149
    eigen_img(:,i) = eigen_img(:,i)/norm(eigen_img(:,i));
end
D_img = diag(D_img);

% top 10 eigen-values and eigen-vectors of images
[V_lm, D_lm] = eigs(scatter_lm,149); % can take top 10 eigen-vectors of warpping
eigen_lm = diff_lm_train * V_lm;
b = eigen_lm(:,1:10)'*diff_lm_test;
for i = 1:149 % 149
    eigen_lm(:,i) = eigen_lm(:,i)/norm(eigen_lm(:,i));
end
%recon_lm = eigen_lm(:,1:10) * b + mean_lm;
D_lm = diag(D_lm);

% Synthesize 20 test faces
recon_img = zeros(65536,20);
recon_lm = zeros(174, 20);
syn_img = zeros(65536,20);

for i = 1:20
    for j = 1:10
        recon_img_eigen = normrnd(0.0, 1.0) * sqrt(D_img(j)/150);
        recon_lm_eigen = normrnd(0.0, 1.0) * sqrt(D_lm(j)/150);
        recon_img(:,i) = recon_img(:,i) + eigen_img(:,j) * recon_img_eigen;
        recon_lm(:,i) = recon_lm(:,i) + eigen_lm(:,j) * recon_lm_eigen;
    end
    recon_img(:,i) = recon_img(:,i) + mean_img;
    recon_lm(:,i) = recon_lm(:,i) + mean_lm;
end
figure;
for i = 1:20
    subplot(4,5,i);
    syn_img(:,i) = double(reshape(warpImage_new(reshape(recon_img(:,i),[256,256]),...
        reshape(mean_lm,[87,2]), reshape(recon_lm(:,i), [87,2])), [65536,1]));
    imshow(uint8(reshape(syn_img(:,i),[256,256])));
end
% figure;
% for i = 1:20
%     syn_img(:,i) = double(reshape(warpImage_new(recon_img(:,i), mean_lm, ...
%         reshape(recon_lm, [256,256])), [65536,1]));
%     subplot(4,5,i);
%     imshow(uint8(reshape(syn_img(:,i),[256,256])));
% end
%}
    