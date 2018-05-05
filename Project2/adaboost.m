%function pos_rec = adaboost()
% clear;clc;close all;
% flags
flag_data_subset = 1;
flag_extract_features = 1;
flag_parpool = 1;
flag_boosting = 1;

% parpool
if flag_parpool
    delete(gcp('nocreate'));
    parpool(4);
end

% unit tests
test_sum_rect();
test_filters();

% constants
if flag_data_subset
    N_pos = 100;
    N_neg = 100;
else
    %N_pos = 11838;
    %N_neg = 45356;
    
    N_pos = 11838;
    N_neg = 25356;
end
N = N_pos + N_neg;
w = 16;
h = 16;

% (1) haar filter

% load images
if flag_extract_features
    tic;
    I = zeros(N, h, w);
    for i=1:N_pos
        I(i,:,:) = rgb2gray(imread(sprintf('newface16/face16_%06d.bmp',i), 'bmp'));
    end
    for i=1:N_neg
        I(N_pos+i,:,:) = rgb2gray(imread(sprintf('nonface16/nonface16_%06d.bmp',i), 'bmp'));
    end
    fprintf('Loading images took %.2f secs.\n', toc);
end

% construct filters
A = filters_A();
B = filters_B();
C = filters_C();
D = filters_D();
if flag_data_subset
    filters = [A(1:250,:); B(1:250,:); C(1:250,:); D(1:250,:)];
else
    filters = [A; B; C; D];
end

% extract features
if flag_extract_features
    tic;
    I = normalize(I);
    II = integral(I);
    features = compute_features(II, filters);
    clear I;
    clear II;
    save('features.mat', '-v7.3', 'features');
    fprintf('Extracting %d features from %d images took %.2f secs.\n', size(filters, 1), N, toc);
else
    load('features.mat','features');
end

% perform boosting
if(flag_boosting == 1)
    fprintf('Running AdaBoost with %d features from %d images.\n', size(filters, 1), N);
    tic;
    %% implement this  
    % AdaBoost
    new_feature = features;
    clear features;
    [r,c] = size(new_feature);
    % init_alpha = zeros(1,r);
    desiredOutput = [ones(r, N_pos), -1*ones(r, N_neg)];
    T = 20;
%     T = [1 10 50 100];
    pos_rec = zeros(1, T);
    bestError_rec = zeros(1, T);
    filter_rec = zeros(T, (N_pos+N_neg));
    alpha_rec = zeros(1, T);
    ht_rec = zeros(1, T);
    ht_x_rec = zeros(T, (N_pos+N_neg));
%     weak_classifiers = [];
    err_rate = zeros(1, T);
    bestError_Tlist = zeros(1, r);   % bestError_Tlist = zeros(4, T);
    %sum_atht_rec = zeros(length(T), (N_pos+N_neg));
    sum_atht_rec = zeros(4, (N_pos+N_neg));
    iter = 0;
    init_weight = (1/c) * ones(1, c);
    weights = init_weight;
%     for T =  [1 10 50 100]
    for T =  20
        iter = iter + 1;
        for t = 1 : T
            [bestError, h, s] = FindBestClassifiers(new_feature, weights, N_pos, N_neg); %?????bestError?1000*1?

            minError = 100;
            for x = 1 : size(bestError, 1)
                if t == 1 
                    pos = find(bestError == min(bestError));
                end
                if (bestError(x) < minError)  
                    %if ~ismember(bestError(i), bestError_rec)
                    pos = find(bestError == bestError(x));
                    if ~ismember(pos, pos_rec)
                        minError = bestError(x);
                    end
                end
            end
            if t ~= 1
                pos = find(bestError == minError);
            end
            disp(pos);
            disp(bestError(pos));

            ht = h(pos);
%             st = s(pos);
%             weak_classifiers = [weak_classifiers; new_feature(pos, :)];

            % Assign voting weights
            alpha = 0.5 * log((1-bestError(pos))/bestError(pos));
            % Update weights
            
            ht_x = find_hx(new_feature(pos, :), ht, c, s(pos));
            weights = weights .* exp(-desiredOutput(1,:) * alpha .* ht_x);
            weights = weights/sum(weights);

            pos_rec(t) = pos; % 1*20
            bestError_rec(t) = bestError(pos);
            filter_rec(t, :) = new_feature(pos,:);
            alpha_rec(t) = alpha;   % 1*20
            ht_rec(t) = ht; % 1*20
            ht_x_rec(t, :) = ht_x; % 20*200
            %new_feature = [new_feature((1:pos-1), :); new_feature((pos+1):end, :)];   

            sum_at = sum(alpha_rec);
            % sum_atht = alpha_rec * ht_x_rec
            count = 0;
            H = zeros(1, N_pos + N_neg);

            for i = 1 : (N_pos + N_neg)
                sum_atht = 0;
                for k = 1 : T
                    sum_atht = sum_atht + alpha_rec(k) * ht_x_rec(k, i);
                end
                if sum_atht >= 0 %(0.5 * sum_at)
                    H(i) = 1;
                else
                    H(i) = 0;
                end
                sum_atht_rec(iter, i) = sum_atht;
            end

            for i = 1 : N_pos
                if H(i) ~= 1
                    count = count + 1;
                end
                if H(N_pos + i) ~= 0
                    count = count + 1;
                end
            end
            err_rate(t) = count / (N_pos + N_neg);

        end
        bestError_Tlist(iter, :) = bestError';      
    end

%    save('adaboost.mat', '-v7.3', 'alpha_rec', 'pos_rec', 'bestError_rec', 'filter_rec', 'ht_rec', 'ht_x_rec', 'err_rate', 'bestError_Tlist', 'sum_atht_rec');
    %save('adaboost.mat','-v7.3','alpha','index','theta','s','y_hat','eps','err','weak_err');
    fprintf('Running AdaBoost %d with features from %d images took %.2f secs.\n', size(filters, 1), N, toc);
else
    %load('adaboost.mat','alpha','index','theta','s','y_hat','eps','err','weak_err');
end

% (1) top-20 haar filters
%% implement this

figure;
for k = 1:20
    [rec1, rec2] = filters{pos_rec(k), :};
    subplot(4,5,k);
    grey_image = 0.5*ones(16,16);
    
    this_image = grey_image;

    if size(rec1,1) == 1
        x1 = rec1(1);
        y1 = rec1(2);
        w1 = rec1(3);
        h1 = rec1(4);
        this_image(x1:x1+w1-1, y1:y1+h1-1) = 0;
    end
    if size(rec1,1) == 2
        x1 = rec1(1,1);
        y1 = rec1(1,2);
        w1 = rec1(1,3);
        h1 = rec1(1,4);
        this_image(x1:x1+w1-1, y1:y1+h1-1) = 0;
        x2 = rec1(2,1);
        y2 = rec1(2,2);
        w2 = rec1(2,3);
        h2 = rec1(2,4);
        this_image(x2:x2+w2-1, y2:y2+h2-1) = 0;
    end
    if size(rec2,1) == 1
        x1 = rec2(1);
        y1 = rec2(2);
        w1 = rec2(3);
        h1 = rec2(4);
        this_image(x1:x1+w1-1, y1:y1+h1-1) = 1;
    end
    if size(rec2,1) == 2
        x1 = rec2(1,1);
        y1 = rec2(1,2);
        w1 = rec2(1,3);
        h1 = rec2(1,4);
        this_image(x1:x1+w1-1, y1:y1+h1-1) = 1;
        x2 = rec2(2,1);
        y2 = rec2(2,2);
        w2 = rec2(2,3);
        h2 = rec2(2,4);
        this_image(x2:x2+w2-1, y2:y2+h2-1) = 1;
    end
    imshow(this_image);

end

% (2) plot training error
%% implement this
figure; plot(1:T, err_rate);
xlabel('The number of steps T');
ylabel('Training error');
title('Training Error of strong classifier');


% (3) training errors of top-1000 weak classifiers
%% implement this
y1 = sort(bestError_Tlist(1,:));
y2 = sort(bestError_Tlist(2,:));
y3 = sort(bestError_Tlist(3,:));
y4 = sort(bestError_Tlist(4,:));
x = 1:1000;
figure;plot(x,y1(1:1000),x,y2(1:1000),x,y3(1:1000),x,y4(1:1000));
legend('k = 1', 'k = 10', 'k = 50', 'k = 100');
xlabel('Top kth weak classifier');
ylabel('Training Error of weak classifiers');

%[bestError_0, ~] = FindBestClassifiers(new_feature, weights, N_pos, N_neg);
% (4) negative positive histograms
%% implement this
real_feature = round(sum_atht_rec);
for i = 2:4
    figure;   
    [n1,x1] = hist(real_feature(i, 1:N_pos),[min(real_feature(i, :)) : 1 : max(real_feature(i, :))]);
    h1=bar(x1,n1,'hist');
    set(h1,'facecolor','c');
    hold on;
    [n2,x2] = hist(real_feature(i, N_pos+1:N_pos+N_neg),[min(real_feature(i, :)) : 1 : max(real_feature(i, :))]);
    h2=bar(x2,n2,'hist');
    set(h2,'facecolor','m')
    hold off;
    legend('Pos(face)', 'Neg(nonface)');
end
% (5) plot ROC curves
%% implement this
real_feature = sum_atht_rec;
figure;
for i = 2:4
    FP = 0;
    TN = 0;
    TP = 0;
    FN = 0;
    FPR = [];
    TPR = [];
    min_thres = min(real_feature(i, :));
    max_thres = max(real_feature(i, :));
    for threshold = min_thres:0.1:max_thres
        FP = sum(real_feature(i, 1+N_pos:end) >= threshold);
        TN = sum(real_feature(i, 1+N_pos:end) < threshold);
        TP = sum(real_feature(i, 1:N_pos) >= threshold);
        FN = sum(real_feature(i, 1:N_pos) < threshold);
        FPR = [FPR; FP/N_neg];
        TPR = [TPR; TP/N_pos];
%         for j = 1 : N_pos
%             if real_feature(j) >= threshold
%                 TP = TP + 1;
%             else
%                 FN = FN + 1;
%             end
%         end
%         for j = 1 : N_neg
%             if real_feature(N_pos + j) >= threshold
%                 FP = FP + 1;
%             else
%                 TN = TN + 1;
%             end
%         end
% %         FPR = [FPR; FP / (FP + TN)];
% %         TPR = [TPR; TP / (TP + FN)];
%         FPR = [FPR; FP/N_neg];
%         TPR = [TPR; TP/N_pos];
    end  
    hold on;
    plot(FPR, TPR);
    
end
hold off;

        
% (6) detect faces
%% implement this
load_test();
disp('Done.');

%end

function hx = find_hx(thres_row, threshold, c, polarity)
hx = zeros(1, c);
for k = 1:c
    if thres_row(k) < threshold
        hx(k) = -polarity;
    else
        hx(k) = polarity;
    end
end
end 

%% filters

function features = compute_features(II, filters)
features = zeros(size(filters, 1), size(II, 1));
for j = 1:size(filters, 1)
    [rects1, rects2] = filters{j,:};
    features(j,:) = apply_filter(II, rects1, rects2);
end
end

function I = normalize(I)
[N,~,~] = size(I);
for i = 1:N;
    image = I(i,:,:);
    sigma = std(image(:));
    I(i,:,:) = I(i,:,:) / sigma;
end
end

function II = integral(I)
[N,H,W] = size(I);
II = zeros(N,H+1,W+1);
for i = 1:N
    image = squeeze(I(i,:,:));
    II(i,2:H+1,2:W+1) = cumsum(cumsum(double(image), 1), 2);
end
end

function sum = apply_filter(II, rects1, rects2)
sum = 0;
% white rects
for k = 1:size(rects1,1)
    r1 = rects1(k,:);
    w = r1(3);
    h = r1(4);
    sum = sum + sum_rect(II, [0, 0], r1) / (w * h * 255);
end
% black rects
for k = 1:size(rects2,1)
    r2 = rects2(k,:);
    w = r2(3);
    h = r2(4);
    sum = sum - sum_rect(II, [0, 0], r2) / (w * h * 255);
end
end

function result = sum_rect(II, offset, rect)
x_off = offset(1);
y_off = offset(2);

x = rect(1);
y = rect(2);
w = rect(3);
h = rect(4);

a1 = II(:, y_off + y + h, x_off + x + w);
a2 = II(:, y_off + y + h, x_off + x);
a3 = II(:, y_off + y,     x_off + x + w);
a4 = II(:, y_off + y,     x_off + x);

result = a1 - a2 - a3 + a4;
end

function rects = filters_A()
count = 1;
w_min = 4;
h_min = 4;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:2:w_max
    for h = h_min:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w/2;
                r1_h = h;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x + r1_w;
                r2_y = r1_y;
                r2_w = w/2;
                r2_h = h;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                rects{count, 1} = r1; % white
                rects{count, 2} = r2; % black
                count = count + 1;
            end
        end
    end
end
end

function rects = filters_B()
count = 1;
w_min = 4;
h_min = 4;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:w_max
    for h = h_min:2:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w;
                r1_h = h/2;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x;
                r2_y = r1_y + r1_h;
                r2_w = w;
                r2_h = h/2;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                rects{count, 1} = r2; % white
                rects{count, 2} = r1; % black
                count = count + 1;
            end
        end
    end
end
end

function rects = filters_C()
count = 1;
w_min = 6;
h_min = 4;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:3:w_max
    for h = h_min:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w/3;
                r1_h = h;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x + r1_w;
                r2_y = r1_y;
                r2_w = w/3;
                r2_h = h;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                r3_x = r1_x + r1_w + r2_w;
                r3_y = r1_y;
                r3_w = w/3;
                r3_h = h;
                r3 = [r3_x, r3_y, r3_w, r3_h];
                
                rects{count, 1} = [r1; r3]; % white
                rects{count, 2} = r2; % black
                count = count + 1;
            end
        end
    end
end
end

function rects = filters_D()
count = 1;
w_min = 6;
h_min = 6;
w_max = 16;
h_max = 16;
rects = cell(1,2);
for w = w_min:2:w_max
    for h = h_min:2:h_max
        for x = 1:(w_max-w)
            for y = 1:(h_max-h)
                r1_x = x;
                r1_y = y;
                r1_w = w/2;
                r1_h = h/2;
                r1 = [r1_x, r1_y, r1_w, r1_h];
                
                r2_x = r1_x+r1_w;
                r2_y = r1_y;
                r2_w = w/2;
                r2_h = h/2;
                r2 = [r2_x, r2_y, r2_w, r2_h];
                
                r3_x = x;
                r3_y = r1_y+r1_h;
                r3_w = w/2;
                r3_h = h/2;
                r3 = [r3_x, r3_y, r3_w, r3_h];
                
                r4_x = r1_x+r1_w;
                r4_y = r1_y+r2_h;
                r4_w = w/2;
                r4_h = h/2;
                r4 = [r4_x, r4_y, r4_w, r4_h];
                
                rects{count, 1} = [r2; r3]; % white
                rects{count, 2} = [r1; r4]; % black
                count = count + 1;
            end
        end
    end
end
end

function test_sum_rect()
% 1
I = zeros(1,16,16);
I(1,2:4,2:4) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [2, 2, 3, 3]) == 9);
assert(sum_rect(II, [0, 0], [10, 10, 2, 2]) == 0);

% 2
I = zeros(1,16,16);
I(1,10:16,10:16) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [10, 10, 2, 2]) == 4);

% 3
I = zeros(1,16,16);
I(1,:,:) = 0;
I(1,3:6,3:6) = 1;
I(1,3:6,11:14) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [11, 3, 6, 6]) == 16);

% 4
I = zeros(1,16,16);
I(1,:,:) = 0;
I(1,3:6,3:6) = 1;
I(1,3:6,11:14) = 1;
%disp(squeeze(I(1,:,:)));
II = integral(I);
assert(sum_rect(II, [0, 0], [3, 4, 4, 4]) == 12);
assert(sum_rect(II, [0, 0], [7, 4, 4, 4]) == 0);
assert(sum_rect(II, [0, 0], [11, 4, 4, 4]) == 12);
assert(sum_rect(II, [0, 0], [3, 3, 4, 4]) == 16);
assert(sum_rect(II, [0, 0], [11, 3, 4, 4]) == 16);

end

function test_filters()

% A
I = zeros(1,16,16);
I(1,:,:) = 255;
I(1,5:8,5:8) = 0;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_A();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r2s(1,3) * r2s(1,4);
    if(and(f_sum > max_sum, f_size == 4*4*2))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [r1s, r2s];
    end
end
assert(max_sum == 1);
assert(max_size == 4*4*2);
assert(isequal(min_f, [1 5 4 4 5 5 4 4]));

% B
I = zeros(1,16,16);
I(1,:,:) = 255;
I(1,2:5,2:5) = 0;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_B();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r2s(1,3) * r2s(1,4);
    if(and(f_sum > max_sum, f_size == 4*4*2))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [r1s, r2s];
    end
end
assert(max_sum == 1);
assert(max_size == 4*4*2);
assert(isequal(min_f, [2 6 4 4 2 2 4 4]));

% C
I = zeros(1,16,16);
I(1,:,:) = 0;
I(1,3:6,3:6) = 255;
I(1,3:6,11:14) = 255;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_C();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r1s(2,3) * r1s(2,4) + r2s(1,3) * r2s(1,4);
    if(and(f_sum > max_sum, f_size == 4*4*3))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [reshape(r1s', [1,8]), r2s];
    end
end
assert(max_sum == 2);
assert(max_size == 4*4*3);
assert(isequal(min_f, [3 3 4 4 11 3 4 4 7 3 4 4]));

% D
I = zeros(1,16,16);
I(1,:,:) = 255;
I(1,2:5,2:5) = 0;
I(1,6:9,6:9) = 0;
II = integral(I);
%disp(squeeze(I(1,:,:)));
rects = filters_D();
max_size = 0;
max_sum = 0;
for i = 1:size(rects, 1)
    [r1s, r2s] = rects{i,:};
    f_sum = apply_filter(II, r1s, r2s);
    f_size = r1s(1,3) * r1s(1,4) + r1s(2,3) * r1s(2,4) + r2s(1,3) * r2s(1,4) + r2s(2,3) * r2s(2,4);
    if(and(f_sum > max_sum, f_size == 4*4*4))
        max_size = f_size;
        max_sum = f_sum;
        min_f = [reshape(r1s', [1,8]), reshape(r2s', [1,8])];
    end
end
assert(max_sum == 2);
assert(max_size == 4*4*4);
assert(isequal(min_f, [6 2 4 4 2 6 4 4 2 2 4 4 6 6 4 4]));

end