function [FPR, TPR] = perfcurve(N_pos, N_neg, row)
FP = 0;
TN = 0;
TP = 0;
FN = 0;
FPR = [];
TPR = [];
min_thres = min(row);
max_thres = max(row);
for threshold = min_thres:0.1:max_thres
    FP = sum(row(1+N_pos:end) >= threshold);
    TN = sum(row(1+N_pos:end) < threshold);
    TP = sum(row(1:N_pos) >= threshold);
    FN = sum(row(1:N_pos) < threshold);
    FPR = [FPR; FP/N_neg];
    TPR = [TPR; TP/N_pos];
%     FPR = [FPR; FP / (FP + TN)];
%     TPR = [TPR; TP / (TP + FN)];
end  
