function [bestError, h, s] = FindBestClassifiers(new_feature, weight, N_pos, N_neg)
[r,c] = size(new_feature);
bestError = 100 * ones(r, 1);
desiredOutput = [ones(1, N_pos), -1*ones(1, N_neg)];
h = zeros(r, 1);
s = zeros(r, 1);
% weak_classifiers = zeros(r,2);
for k = 1:r
    minF = min(new_feature(k,:));
    maxF = max(new_feature(k,:));
    step = (maxF - minF)/c;

    for threshold = minF : step : maxF
        compare = zeros(1, c);
        for polarity = [1 -1]
            hx = find_hx(new_feature(k, :), threshold, c, polarity);
            for j = 1:c
                if desiredOutput(j) ~= hx(j)
                    compare(j) = 1;
                end
            end           
            Error = weight * compare'; %(1*c)*(c*1)
            if (Error < bestError(k))
                % disp(Error);
                % disp(bestError);
                bestError(k) = Error;
                h(k) = threshold;
                s(k) = polarity;
            end
        end
        
    end
    
%{
    for threshold = minF : step : maxF
        for s = [1 -1]
            Error = 0;
            % iterate m images
            for i = 1:c
                % h(x) = - s_j
                if new_feature(k,i) <= threshold
                    Error = Error + weight(i) * pred_err(-s, desiredOutput(i));
                % h(x) = s_j
                else
                    Error = Error + weight(i) * pred_err(s, desiredOutput(i));
                end
            end
            if Error < bestError(k)
                bestError(k) = Error;
                weak_clf = [threshold, s];
%                 h(k) = threshold;
            end
        end   
    end
    weak_classifiers(k,1) = weak_clf(1);
    weak_classifiers(k,2) = weak_clf(2);    
    %}
    % Assign voting weights
    % alpha(k) = 0.5 * log10((1-bestError(k))/bestError(k));
end
end

function hx = find_hx(thres_row, threshold, c, s)
hx = zeros(1, c);
for i = 1:c
    if thres_row(i) < threshold
        hx(i) = -s;
    else
        hx(i) = s;
    end
end
end
% 
% function err = pred_err(hx, y)
% if hx ~= y
%     err = 1;
% else
%     err = 0;
% end
% end
