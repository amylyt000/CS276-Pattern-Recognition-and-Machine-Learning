[m_train, m_test, f_train, f_test, u_train, train, test] = load_fisher();
% Find the FLD or Fisher face that distinguishes male from female
% using training sets
% Compute mean value for male training sets and female training sets
% [r, c] = size(m_train);
mean_male = zeros(65536, 1);
for i = 1:65536
    mean_male(i) = sum(m_train(i,:))/78;
end
mean_female = zeros(65536, 1);
for i = 1:65536
    mean_female(i) = sum(f_train(i,:))/75;
end
% Compute eigen-vectors and eigen-values of within-class scatter matrix
w1 = fisher(m_train, f_train, mean_male, mean_female);
% Plot
boundary = (w1' * mean_male + w1' * mean_female)/2;
figure;
show_male_train = w1' * m_train;
plot(show_male_train, 'b+')    
hold on;
show_female_train = w1' * f_train;
plot(show_female_train, 'r+');
hold on;
show_male_test = w1' * m_test;
plot(show_male_test, 'bo');
hold on;
show_female_test = w1' * f_test;
plot(show_female_test, 'ro');
hold on;
plot(ones(1, 78) * boundary, 'y-');
hold off;

% Compute the Fisher faces over reduced dimensions
all_train = [m_train, f_train];
[r, c] = size(all_train);
mean = zeros(r, 1);
for i = 1:r
    mean(i) = sum(all_train(i, :))/c;
end
% diff_matrix = zeros(r,c);
diff_matrix = all_train - mean;
% for i = 1:c
%     diff_matrix(:,i) = all_train(:, i) - mean;
% end
scatter_matrix = diff_matrix' * diff_matrix;
[U, S, V] = svd(scatter_matrix);
e = diff_matrix * U(:, 1:50);


a = e'*(m_train - mean);
% reconstructed_male= e * a;
b = e'*(f_train- mean);
% reconstructed_female= e * a;
[w2, m_mean, f_mean] = Reduced_fisher(a, b);
% w2 = Reduced_fisher(reconstructed_male, reconstructed_female);
c = e'*(m_test - mean);
% reconstructed_male= e * a;
d = e'*(f_test - mean);
% Plot
% boundary = (w2' * m_mean + w2' * f_mean)/2;
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
% hold on;
% plot(ones(1, 78) * boundary, 'cyan-', 'linewidth',1);
hold off;
legend('male train', 'female train', 'male test', 'female test');
