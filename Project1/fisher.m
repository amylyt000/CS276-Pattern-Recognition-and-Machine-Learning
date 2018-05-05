function [w] = fisher(data1, data2, mean1, mean2)
C = [data1, data2];
B = C' * C;
[V, D] = eig(B);
[r, c] = size(C);
A = zeros(r, 153);
for i = 1: 153
    A(:, i) = sqrt(D(i,i)) * C * V(:, i)/norm(C * V(:,i));
end
y = A' * (mean1 - mean2);
z = inv((D^2 * V')) * y;
w = C * z;
w = w/norm(w);
end





