function [lm_male_train, lm_male_test, lm_female_train, lm_female_test,...
    lm_unknown_train, train_lm, test_lm] = load_fisher_align()
lm_male = zeros(87*2, 88);
lm_female = zeros(87*2, 85);
% male landmarks
for i = 1:88
    if i < 58
        dtpath = sprintf('./face_data/male_landmark_87/face%03d_87pt.txt',i-1);
    end
    if i >= 58
        dtpath = sprintf('./face_data/male_landmark_87/face%03d_87pt.txt',i);
    end
    lm = textread(dtpath);
    lm_male(:,i) = reshape(lm, [87*2, 1]);
end
lm_male_train = lm_male(:, 1:78);
lm_male_test = lm_male(:, 79:88);
% female landmarks
for i = 1:85
    dtpath = sprintf('./face_data/female_landmark_87/face%03d_87pt.txt',i-1);
    lm = textread(dtpath);
    lm_female(:,i) = reshape(lm, [87*2, 1]);
end
lm_female_train = lm_female(:, 1:75);
lm_female_test = lm_female(:, 76:85);
% unknown landmarks
for i = 1:4
    dtpath = sprintf('./face_data/unknown_landmark_87/face%03d_87pt.txt',i-1);
    lm = textread(dtpath);
    lm_unknown(:,i) = reshape(lm, [87*2, 1]);
end
lm_unknown_train = lm_unknown;

train_lm = [lm_male_train, lm_female_train];
test_lm = [lm_male_test, lm_female_test];
end
