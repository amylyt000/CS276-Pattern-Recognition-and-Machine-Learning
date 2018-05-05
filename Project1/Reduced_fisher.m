function [w, m_mean, f_mean] = Reduced_fisher(male, female)

[r1, c1] = size(male);
[r2, c2] = size(female);

m_mean = zeros(r1, 1);
for i = 1:r1
    m_mean(i) = sum(male(i, :))/c1;
end

f_mean = zeros(r2, 1);
for i = 1:r2
    f_mean(i) = sum(female(i, :))/c2;
end

S_m = (male - m_mean) * (male - m_mean)';
S_f = (female - f_mean) * (female - f_mean)';

w = inv(S_m + S_f) * (m_mean - f_mean);
w = w./norm(w);
end
