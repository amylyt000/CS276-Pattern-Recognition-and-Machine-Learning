path = '/Users/luyutong/Documents/MATLAB/project 2/Testing_Images/';
% N = N_pos + N_neg;

min_scale = 0.15;
overlap = 0.7;
for i = 1:6
    w_orig = 1280;
    h_orig = 960;
    J = zeros(1, h_orig, w_orig);
    if i <= 3
        impath = [path, sprintf('Face_%01d.jpg', i)];
    else 
        impath = [path, sprintf('Face_%01d.jpg', i-3)];
    end
%     J(1,:,:) = rgb2gray(imresize(imread(impath), scale));
    J(1,:,:) = rgb2gray(imread(impath));
    boxes = [];
    for scale_factor = 1:0.5:2
        hard_neg_boxes = [];
        scale = min_scale * 2 ^ scale_factor;
        scaled_face = imresize(J, scale);
        w = 16;
        h = 16;
        disp(scale_factor);
        for x = 1:(size(scaled_face, 3) - w)
            for y=1:(size(scaled_face,2) - h)
                I_crop = J(1, x:x+16, y:y+16);
                I_crop = normalize(I_crop);
                II_crop = integral(I_crop);
                test_fea = compute_features(II_crop, filters);
                clear I_crop;
                clear II_crop;
                
%                 ht_x_test_rec = zeros(20,1);
                sum_atht_test = 0;
%                 disp(sum_atht_test);
                T = 20;
                for t = 1:T
%                     ht_x_test = find_hx(test_fea(pos_rec(t), :), ht_rec(t), 1, s(pos_rec(t)));
% %                     ht_x_test_rec(t) = ht_x_test;
%                     sum_atht_test = sum_atht_test + alpha_rec(k) * ht_x_test;
                     ht_x_test = find_hx(test_fea(h(t,1), :), h(t,2), 1, h(t,4));
%                     ht_x_test_rec(t) = ht_x_test;
                    sum_atht_test = sum_atht_test + alpha(t) * ht_x_test;
                end

%                 if sum_atht_test >= 0 %(0.5 * sum_at)
%                     H = 1;
%         %             disp(H);
%                     rectangle('position',[row,col,16/scale,16/scale],'edgecolor','b');
%                 else
%                     H = 0;
%                 end 
                
                if sum_atht_test >= 0
                    if i > 3
                        hard_neg_box = [x y w h];
                        hard_neg_boxes = [hard_neg_boxes; [hard_boxes sum_atht_test]];
                        
                    end
                    box = [x y w h]/scale;
                    boxes = [boxes; [box sum_atht_test]];
                end
            end
        end
        hard_neg_candidates = nms(hard_neg_boxes, overlap);
        hard_neg_candidates = hard_neg_boxes;
        N_hard_neg = size(hard_neg_candidates, 1);
        for j=1:N_hard_neg
            I_crop = imcrop(scaled_face, hard_neg_candidates(j, :));
            I_crop = normalize(I_crop);
            II_crop = integral(I_crop);
            features = [features; compute_features(II_crop, filters)];
        end
%         fprintf('There are now %d negative examples (including %d hard negative ones)', N_neg + N_hard_neg, N_hard_neg);
    end
    
    candidates = nms(boxes, overlap);
    figure;imshow(imread(impath));
    for j=1:size(candidates, 1)
        rectangle('Position', candidates(j, 1:4), 'EdgeColor', 'r');
    end
end
%     print(gcf, '-djpeg', sprintf('./pictures/detection_Picture%d.jpg', i));
%     close all                
                        


%     for i = 1
%         imgs = zeros(w*h/256,16,16);
%         j = 0;
%         row = 1;
%         while (row+15 <= h)
%             col = 1;
%             while (col+15 <= w)
%                 j = j+1;
%                 img = J(i, row:row+15, col:col+15);
%                 imgs(j,:,:) = img; 
%                 col = col + 2;
%     %             disp(col);           
%             end
%             row = row + 2;
%         end
%     end
%     imgs = normalize(imgs);
%     imgs = integral(imgs);
%     test_fea = compute_features(imgs, filters);
% 
%     for i = 1
%         %figure;imshow(imresize(imread(impath), scale));
%         row = 1;
%         col = 1;
%         for j = 1:size(test_fea, 2)
%             test_f = test_fea(:, j);
%             ht_x_test_rec = zeros(20,1);
%             T = 20;
%             for t = 1:T
%                 ht_x_test = find_hx(test_f(pos_rec(t), :), ht_rec(t), 1);
%                 ht_x_test_rec(t) = ht_x_test;
%             end
% 
%             sum_atht_test = 0;
%             for k = 1:T
%                 sum_atht_test = sum_atht_test + alpha_rec(k) * ht_x_test_rec(k);
%             end
%             if sum_atht_test >= 0 %(0.5 * sum_at)
%                 H = 1;
%     %             disp(H);
%                 rectangle('position',[row,col,16/scale,16/scale],'edgecolor','b');
%             else
%                 H = 0;
%             end
%             col = col+2;
%             if col-1 == w
%                 row = row + 2;
%                 col = 1;
%             end
%         end
%     end


                    
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

