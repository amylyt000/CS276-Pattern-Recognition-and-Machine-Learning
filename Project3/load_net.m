clear; close all;
run_setup = 0;
if run_setup == 1
    Setup();
end

%% Set up directories, load images and pretrained net
fullpath = pwd; 
% ['/Users/luyutong/Documents/MATLAB/project3_code&data/code/fast_rcnn']
[folder, name, ext] = fileparts(mfilename('fullpath'));
pretrain_model_dir = fullfile(folder, '..', '..', 'data', 'models', ...
               'fast-rcnn-caffenet-pascal07-dagnn.mat');
raw_net = load(pretrain_model_dir);
net = preprocessNet(raw_net);
single_im = fullfile(folder, '..', '..', 'example.jpg');           
single_im_bbox = fullfile(folder, '..', '..', 'example_boxes.mat');

%% Part 1 

example_img = imread(single_im); % 406*500*3 uint8
example_bbox = load(single_im_bbox);
example_rois = single(example_bbox.boxes); % 2888*4 uint16

[im, RoIs] = resizeImBbox(single(example_img), example_rois);
im = im - net.meta.normalization.averageImage;
net.eval({'data', im, 'rois', RoIs'});
score = squeeze(net.getVar('cls_prob').value);
bbox_reg = squeeze(net.getVar('bbox_pred').value);

car_ids = 8; % no. 8 in net.meta.class.name is 'car'
nms_threshold = 0.3;
score_car_before = score(car_ids, :); % 1*2888 single
bbox_reg_car = bbox_reg(1+(car_ids-1)*4: car_ids*4, :); % 4*2888 single
reg_bbox_car_before = bbox_transform_inv(example_rois, bbox_reg_car'); % x1, x2, y1, y2
% rois: 2888*4 single, reg_bbox_car_before: 2888*4 single
chosen_detection = nms([reg_bbox_car_before, score_car_before'], nms_threshold);
reg_bbox_car = reg_bbox_car_before(chosen_detection, :); % 45*4 single
probs_car = score_car_before(chosen_detection)'; % 465*1 single

low = min(probs_car);
up = max(probs_car);
num_threds = length(probs_car); % = 465
threds = linspace(low, up, num_threds); % 1*465
positives = repmat(probs_car', num_threds, 1) > repmat(threds, num_threds, 1)';
num_positives = sum(positives, 2); % calculate the sum for each row
figure;
plot(threds, num_positives);
xlabel('thresholds');
ylabel('number of detections');
title('No. Detections vs Thresholds');

id_chosen = 280;
part1_final_thred = threds(id_chosen);
fprintf('Chosen threshold %f\n', part1_final_thred);
detections = find(positives(id_chosen, :));

figure; imshow(example_img);
hold on;
for detection = detections
    x = reg_bbox_car(detection, 1);
    y = reg_bbox_car(detection, 2);
    w = reg_bbox_car(detection, 3) - x + 1;
    h = reg_bbox_car(detection, 4) - y + 1;
    rectangle('Position', [x+1, y+1, w, h], 'EdgeColor', 'red', 'LineWidth', 1.5);
    rectangle('Position', [x+1, y-10, w, 11], 'FaceColor', 'red');
    str = sprintf('%0.2f', probs_car(detection));
    text(double(x+1), double(y+1),str,...
        'HorizontalAlignment', 'left',...
        'VerticalAlignment', 'bottom',...
        'Margin', 1, 'FontSize', 10);
end

%% Part 2
% part1_final_thred = 0.6906;
run_network = 0;
annotation_dir = '../../data/annotations';
a_files = dir(annotation_dir); % 4954x1 struct with 6 fields
img_dir = '../../data/images';
rois_dir = fullfile(folder,'..','..','data','SSW','SelectiveSearchVOC2007test.mat');
rois_all = load(rois_dir);
num_ims = length(rois_all.images); % 4952 images

all_annotations = cell(num_ims, 1);
count = 1;
for i = 1:numel(a_files)
    file = a_files(i);
    if length(file.name) < 4 % since names of the first two files are '.' and '..'
        continue;
    end
    annotations = PASreadrecord(fullfile(annotation_dir, file.name));
    all_annotations{count} = annotations;
    count = count + 1;
end

if run_network
    for i = 1: num_ims % =4952
        tic
        assert(strcmp(rois_all.images{i}, all_annotations{i}.filename(1: end-4)));
        im_path = fullfile(img_dir, all_annotations{i}.filename);
        im = imread(im_path);
        [im, Rois] = resizeImBbox(single(im), single(rois_all.boxes{i}));
        im = im - net.meta.normalization.averageImage;
        net.eval({'data', im, 'rois', Rois'});
        score = squeeze(net.getVar('cls_prob').value);
        bbox_reg = squeeze(net.getVar('bbox_pred').value);
        results(i).name = all_annotations{i}.filename;
        results(i).prob = score;
        results(i).bbox_reg = bbox_reg;
        fprintf('Process image %d in %d seconds\n', i, toc);
    end
    save('results.mat', '-v7.3', 'results');
else
    load('results.mat', 'results');
end

%% merge results by class
clear merge_results;
do_merge_results = 0;
num_class = 20;
nms_threshold = 0.3;
class_results(num_class) = struct('detections', [],...
                                'gts', [],...
                                'precision', [],...
                                'recall', [],...
                                'ap', []);
class_top_selection = 40;
total_merge_chosen = class_top_selection * num_ims; % = 198080
if do_merge_results
    merge_results(num_class) = struct('im_score_bbreg', []);
    for c = 1:num_class
        tic;
        class_id = c+1; % the first class is background
        for i = 1:num_ims
            score_before = results(i).prob(class_id, :); %1*1828 single
            bbox_reg = results(i).bbox_reg(1+(class_id - 1)*4:class_id*4, :); % 4*1828 single
            image_id = i*ones(1, size(score_before, 2)); % 1*1828 double 
            this_im_this_class = [image_id; score_before; 1:length(image_id)];
            merge_results(c).im_score_bbreg = ...
                [merge_results(c).im_score_bbreg; this_im_this_class'];
        end
        [~, I] = sort(merge_results(c).im_score_bbreg(:,2), 'descend');
        merge_results(c).im_score_bbreg = ...
            merge_results(c).im_score_bbreg(I(1:total_merge_chosen),:);
        fprintf('Merge class: %d takes %f seconds\n', c, toc);
    end
    save('merge_results.mat', '-v7.3', 'merge_results');
else
    load('merge_results.mat', 'merge_results');
end

%% get multi_classes detection
best_detect_num = 0;
best_detect_im = 0;
best_detect_boxes = 0;
for i = 1:num_ims % =4952
    current_bbox = [];
    class_included = 0;
    for c = 1:num_class
        chosen = (merge_results(c).im_score_bbreg(:,1) == i) & ...
            (merge_results(c).im_score_bbreg(:,2) > part1_final_thred); 
        detect_box_id = merge_results(c).im_score_bbreg(chosen, 3);
        detect_box_prob = merge_results(c).im_score_bbreg(chosen, 2);
        find_this_class_num = numel(detect_box_id); 
        if find_this_class_num > 0
            class_included = class_included + 1;
        end
        current_bbox = [current_bbox;...
            [c*ones(find_this_class_num, 1), detect_box_id, detect_box_prob]];
    end
    if class_included > best_detect_num
        best_detect_im = i; % =1275
        best_detect_boxes = current_bbox;
        best_detect_num = class_included;
    end
end

final_boxes = zeros(size(best_detect_boxes, 1), 6);
for b = 1:size(best_detect_boxes, 1) % no. of rows = 74
    box_cl = best_detect_boxes(b, 1);
    box_id = best_detect_boxes(b, 2);
    box_score = best_detect_boxes(b, 3);
    box_reg = results(best_detect_im).bbox_reg(1+box_cl*4:(box_cl+1)*4, box_id); 
    score = results(best_detect_im).prob(box_cl+1, box_id); 
    assert(box_score == score);
    box_propose = rois_all.boxes{best_detect_im}(box_id, :);
    reg_bbox = bbox_transform_inv(box_propose, box_reg.'); 
    final_boxes(b, :) = [box_cl, reg_bbox, box_score']; 
end

classes = unique(final_boxes(:, 1));
final_after_nms = [];
for ic = 1:numel(classes)
    c = classes(ic);
    this_class = final_boxes(final_boxes(:, 1) == c, 2:end);
    after_nms = nms(this_class, nms_threshold);
    final_after_nms = [final_after_nms; [c*ones(size(after_nms,1), 1),...
        this_class(after_nms, :)]];
end

colors = 'ycmrgbycmrgbycmrgbycmrgb';
figure;
imshow(fullfile(img_dir, all_annotations{best_detect_im}.filename));
hold on;
for b = 1:size(final_after_nms)
    reg_bbox = final_after_nms(b, :);
    class_name = net.meta.classes.name{reg_bbox(1) + 1}; 
    x = reg_bbox(2);
    y = reg_bbox(3);
    w = reg_bbox(4) - x + 1;
    h = reg_bbox(5) - y + 1;
    score = reg_bbox(6);
    color = colors(reg_bbox(1));
    rectangle('Position', [x+1, y+1, w, h], 'EdgeColor', color);
    rectangle('Position', [x+1, y-10, w, 11], 'FaceColor', color);
    str = sprintf('%s: %0.2f', class_name, score);
    text(double(x+1), double(y+1), str,...
        'HorizontalAlignment', 'left',...
        'VerticalAlignment', 'bottom',...
        'Margin', 1, 'FontSize', 10);
end

%% draw pr_curve
mAP = 0;
image_top_selection = 100;
for c = 1:num_class
    class_id = c + 1;
    class_name = net.meta.classes.name{class_id};
    detect_results(num_ims) = struct('Boxes', [], 'Scores', []);
    for i = 1:num_ims
        assert(strcmp(results(i).name, all_annotations{i}.filename)); 
        I = (merge_results(c).im_score_bbreg(:, 1) == i);
        score_before = merge_results(c).im_score_bbreg(I,2);
        bboxes_idx = merge_results(c).im_score_bbreg(I,3);
        if numel(bboxes_idx) ~= 0
            raw_bbox_reg = results(i).bbox_reg(1+c*4:(c+1)*4, bboxes_idx);
            reg_bbox_before = bbox_transform_inv(rois_all.boxes{i}(bboxes_idx, :),... 
                raw_bbox_reg');%14*4 single
            chosen_detection = nms([reg_bbox_before, score_before], nms_threshold);
            num_chosen_detection = min(image_top_selection, size(chosen_detection, 1));
            reg_bbox = reg_bbox_before(chosen_detection(1:num_chosen_detection), :);
            score = score_before(chosen_detection(1:num_chosen_detection));
            
            detect_results(i).Boxes = xyxy2xywh(reg_bbox) + [1,1,0,0];
            detect_results(i).Scores = score;
        else
            detect_results(i).Boxes = [];
            detect_results(i).Scores = [];
        end
        
        gt_bbox = [];
        for obj = all_annotations{i}.objects
            if strcmp(obj.class, class_name)
                xywh_gt = xyxy2xywh(obj.bbox) + [1,1,0,0];
                gt_bbox = [gt_bbox; xywh_gt];
            end
        end
        all_gts(i).Boxes = gt_bbox;
    end
    [ap, recall, precision] = evaluateDetectionPrecision(...
                            struct2table(detect_results), ...
                            struct2table(all_gts), 0.5);
    class_results(c).detection = detect_results;
    class_results(c).gts = all_gts;
    class_results(c).ap = ap;
    class_results(c).precision = precision;
    class_results(c).recall = recall;
    fprintf('Class %s, AP: %f\n', class_name, ap);
    mAP = mAP + ap;
end
save('class_results.mat', '-v7.3', 'class_results');

mAP = mAP/num_class;
figure;plot(class_results(7).recall, class_results(7).precision);
xlabel('Recall');
ylabel('Precision');
title('Precision-Recall curve for the car class, mAP = 0.5346');
  
%{

Class aeroplane, AP: 0.610109
Class bicycle, AP: 0.623567
Class bird, AP: 0.418176
Class boat, AP: 0.311831
Class bottle, AP: 0.348172
Class bus, AP: 0.613599
Class car, AP: 0.597605
Class cat, AP: 0.686555
Class chair, AP: 0.620031
Class cow, AP: 0.520160
Class diningtable, AP: 0.496266
Class dog, AP: 0.589480
Class horse, AP: 0.649905
Class motorbike, AP: 0.632312
Class person, AP: 0.513134
Class pottedplant, AP: 0.509296
Class sheep, AP: 0.427564
Class sofa, AP: 0.545914
Class train, AP: 0.655892
Class tvmonitor, AP: 0.522837
         
%}          
   
        
    