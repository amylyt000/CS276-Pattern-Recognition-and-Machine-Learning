function new_box = xyxy2xywh(box)
x = box(:,1);
y = box(:,2);
x2 = box(:,3);
y2 = box(:,4);
w = x2 - x + 1;
h = y2 - y + 1;
new_box = [x y w h];


