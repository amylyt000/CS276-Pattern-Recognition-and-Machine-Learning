close all; clear; clc;
%% fisher face (appearance)
[m_train, m_test, f_train, f_test, u_train, train, test] = load_fisher();
mean_male = zeros(65536, 1);
for i = 1:65536
    mean_male(i) = sum(m_train(i,:))/78;
end
mean_female = zeros(65536, 1);
for i = 1:65536
    mean_female(i) = sum(f_train(i,:))/75;
end

%% fisher face (geometric)
[lm_male_train, lm_male_test, lm_female_train, lm_female_test, lm_unknown_train, train_lm, test_lm] = load_fisher_align();
mean_male_lm = zeros(174, 1);
for i = 1:174
    mean_male_lm(i) = sum(lm_male_train(i,:))/78;
end
mean_female_lm = zeros(174, 1);
for i = 1:174
    mean_female_lm(i) = sum(lm_female_train(i,:))/75;
end

% Compute the Fisher faces over reduced dimensions
all_train = [m_train, f_train];
[r, c] = size(all_train);
mean = zeros(r, 1);
for i = 1:r
    mean(i) = sum(all_train(i, :))/c;
end

all_lm = [lm_male_train, lm_female_train];
[r1, c1] = size(all_lm);
mean_lm = zeros(r1, 1);
for i = 1:r1
    mean_lm(i) = sum(all_lm(i, :))/c1;
end

all_train_align = zeros(65536, 153);
for i = 1 : c
    all_train_align(:,i) = double(reshape(warpImage_new(reshape(all_train(:,i),[256,256]), ...
        reshape(train_lm(:,i),[87,2]),...
        reshape(mean_lm, [87,2])),[65536,1])); % type is uint8
end

% diff_matrix = zeros(r,c);
diff_matrix = all_train_align - mean;
% for i = 1:c
%     diff_matrix(:,i) = all_train(:, i) - mean;
% end
scatter_matrix = diff_matrix' * diff_matrix;
[U, S, V] = svd(scatter_matrix);
e = diff_matrix * U(:, 1:50);

m_train_align = all_train_align(:, 1:78);
f_train_align = all_train_align(:, 79:153);
a = e'*(m_train_align - mean);
% reconstructed_male= e * a;
b = e'*(f_train_align - mean);
% reconstructed_female= e * a;
[w2, m_mean, f_mean] = Reduced_fisher(a, b);
% w2 = Reduced_fisher(reconstructed_male, reconstructed_female);
c = e'*(m_test - mean);
% reconstructed_male= e * a;
d = e'*(f_test - mean);
%----------------------------------------------------------------
% all_lm = [lm_male_train, lm_female_train];
% [r1, c1] = size(all_lm);
% mean_lm = zeros(r1, 1);
% for i = 1:r1
%     mean_lm(i) = sum(all_lm(i, :))/c1;
% end

diff_matrix_lm = all_lm - mean_lm;
scatter_matrix_lm = diff_matrix_lm' * diff_matrix_lm;
[U_lm, S_lm, V_lm] = svd(scatter_matrix_lm);
e_lm = diff_matrix_lm * U_lm(:, 1:50);

a_lm = e_lm'*(lm_male_train - mean_lm);
% reconstructed_male= e * a;
b_lm = e_lm'*(lm_female_train - mean_lm);
% reconstructed_female= e * a;
[w2_lm, m_mean_lm, f_mean_lm] = Reduced_fisher(a_lm, b_lm);
% w2 = Reduced_fisher(reconstructed_male, reconstructed_female);
c_lm = e_lm'*(lm_male_test - mean_lm);
% reconstructed_male= e * a;
d_lm = e_lm'*(lm_female_test - mean_lm);

figure;
hold on;
scatter(w2' * a /(10^8), w2_lm' * a_lm /(10^4), 'b+');
hold on;
scatter(w2' * b/(10^8), w2_lm' * b_lm/(10^4), 'r+');
hold off;
legend('male train', 'female train')

figure;
hold on;
scatter(w2' * c/(10^8), w2_lm' * c_lm/(10^4), 'bo');
hold on;
scatter(w2' * d/(10^8), w2_lm' * d_lm/(10^4), 'ro');
hold off;
legend('male test', 'female test')
%{
% Plot
boundary = (w2' * m_mean + w2' * f_mean)/2;
figure;
show_male_train = w2' * a;
plot(show_male_train, 'b+')    
hold on;
show_female_train = w2' * b;
plot(show_female_train, 'r+');
hold on;
show_male_test = w2' * c;
plot(show_male_test, 'bo');
hold on;
show_female_test = w2' * d;
plot(show_female_test, 'ro');
hold on;
plot(ones(1, 78) * boundary, 'cyan-', 'linewidth',1);
hold off;
legend('male train', 'female train', 'male test', 'female test');
%}


