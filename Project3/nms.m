function select = nms(bboxes, overlap)
if isempty(bboxes)
  select = [];
  return;
end

x1 = bboxes(:,1);
y1 = bboxes(:,2);
x2 = bboxes(:,3);
y2 = bboxes(:,4);
score = bboxes(:,end);

area = (x2-x1+1) .* (y2-y1+1);
[~, Index] = sort(score);

select = score*0;
counter = 1;
while ~isempty(Index)
  i = Index(end);
  select(counter) = i;
  counter = counter + 1;

  xx1 = max(x1(i), x1(Index(1:end-1)));
  yy1 = max(y1(i), y1(Index(1:end-1)));
  xx2 = min(x2(i), x2(Index(1:end-1)));
  yy2 = min(y2(i), y2(Index(1:end-1)));

  w = max(0.0, xx2-xx1+1);
  h = max(0.0, yy2-yy1+1);

  inter = w.*h;
  o = inter ./ (area(i) + area(Index(1:end-1)) - inter);

  Index = Index(find(o<=overlap));
end

select = select(1:(counter-1));



