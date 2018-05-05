function [img, Rois] = resizeImBbox(img, rois)
scale = 600/min(size(img,1), size(img,2));
img = imresize(img, scale);
Rois = cat(2, ones(size(rois, 1), 1), rois*scale);